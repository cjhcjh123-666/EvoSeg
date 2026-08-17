#!/usr/bin/env python
"""EvoSeg Faithfulness GRPO (image referring segmentation, v1).

Rewards:
  present : mask IoU (and Acc@0.5) against GT
  absent  : abstention reward (no mask -> 1.0; shaped penalty on mask area)
Update   : GRPO (group-relative advantage, PPO-clip ratio) on the LLM response
           token logprobs; KL to the frozen SFT (non-LoRA) policy.
"""
import argparse
import json
import math
import os
import random
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from PIL import Image
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from peft import LoraConfig, get_peft_model
from pycocotools import mask as mask_utils
from qwen_vl_utils import process_vision_info

REPO = '/9950backfile/chenjiahui/EvoSeg'
sys.path.insert(0, os.path.join(REPO, 'projects', 'evoseg', 'rl'))
from rewards import reward_present, reward_absent, format_reward  # noqa: E402

GREFS = '/9950backfile/chenjiahui/evo_artifacts/datasets/grefcoco/grefs_unc.json'
INSTANCES = '/9950backfile/chenjiahui/evo_artifacts/datasets/grefcoco/instances.json'
IMG_ROOT = '/9950backfile/chenjiahui/evo_artifacts/datasets/s4b/pixel_llm_data/glamm_data/images/coco2014/train2014'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('model_path', help='HF Sa2VA model dir (merged SFT checkpoint).')
    p.add_argument('--save-dir', default='/9950backfile/chenjiahui/evo_artifacts/results/s4b/evoseg_grpo_v1')
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--batch-prompts', type=int, default=4, help='prompts per rank per step')
    p.add_argument('--group-size', type=int, default=4, help='rollouts per prompt')
    p.add_argument('--max-prompts', type=int, default=None, help='cap total prompts (debug)')
    p.add_argument('--lr', type=float, default=2e-6)
    p.add_argument('--kl-coef', type=float, default=0.05)
    p.add_argument('--clip-eps', type=float, default=0.2)
    p.add_argument('--lora-r', type=int, default=32)
    p.add_argument('--lora-alpha', type=int, default=64)
    p.add_argument('--temperature', type=float, default=0.9)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--max-new-tokens', type=int, default=32)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--present-ratio', type=float, default=0.5)
    p.add_argument('--local_rank', '--local-rank', type=int, default=0)
    return p.parse_args()


def setup():
    dist.init_process_group('nccl')
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world


def load_manifest(max_prompts=None, present_ratio=0.5):
    """Build (present, absent) prompt lists from gRefCOCO source."""
    grefs = json.load(open(GREFS))
    instances = json.load(open(INSTANCES))
    anns = {a['id']: a for a in instances['annotations']}
    img_ids = {im['id']: im for im in instances['images']}

    present, absent = [], []
    for r in grefs:
        fn = os.path.join(IMG_ROOT, r['file_name'])
        if not os.path.exists(fn):
            continue
        sent = r['sentences'][0]['sent']
        if r['no_target'] or r['ann_id'] == [-1]:
            if r['split'] == 'val':
                absent.append({'file': fn, 'query': sent, 'kind': 'absent', 'gt': None})
        else:
            if r['split'] == 'train':
                present.append({'file': fn, 'query': sent, 'kind': 'present', 'image_id': r['image_id'],
                                'ann_ids': r['ann_id']})
    if max_prompts:
        rng = random.Random(0)
        rng.shuffle(present)
        rng.shuffle(absent)
        n_abs = int(max_prompts * (1 - present_ratio))
        n_pre = max_prompts - n_abs
        present, absent = present[:n_pre], absent[:n_abs]
    # decode GT masks only for kept present samples
    for s in present:
        h, w = img_ids[s['image_id']]['height'], img_ids[s['image_id']]['width']
        mask = np.zeros((h, w), dtype=np.uint8)
        for aid in s['ann_ids']:
            ann = anns.get(aid)
            if ann is None or len(ann['segmentation']) == 0:
                continue
            seg = ann['segmentation']
            if isinstance(seg, dict):
                counts = seg['counts']
                hh, ww = seg['size']
                flat = np.zeros(hh * ww, dtype=np.uint8)
                pos = 0
                for i, c in enumerate(counts):
                    if i % 2 == 1:
                        flat[pos:pos + c] = 1
                    pos += c
                m = flat.reshape(hh, ww)
            elif isinstance(seg[0], list):
                rle = mask_utils.frPyObjects(seg, h, w)
                m = mask_utils.decode(rle)
                m = m.sum(axis=2) if m.ndim == 3 else m
            else:
                m = mask_utils.decode([seg])
                m = m.sum(axis=2) if m.ndim == 3 else m
            mask = np.maximum(mask, m.astype(np.uint8))
        s['gt'] = mask
        del s['image_id'], s['ann_ids']
    return present, absent


def build_prompt(query):
    return f"<image>\n Please segment {query} in this image."


def build_mm_inputs(model, processor, image, text):
    """Qwen3VL processor path (same as predict_forward)."""
    messages = [{
        'role': 'user',
        'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': text}],
    }]
    processsed_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    mm_inputs = processor(
        text=[processsed_text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors='pt',
        min_pixels=model.min_pixels, max_pixels=model.max_pixels,
    ).to(image.device if isinstance(image, torch.Tensor) else next(model.parameters()).device)
    return mm_inputs


def build_g_pixel_values(model, image):
    g_img = np.array(image)
    g_img = model.extra_image_processor.apply_image(g_img)
    g_px = torch.from_numpy(g_img).permute(2, 0, 1).contiguous().to(model.torch_dtype)
    g_px = torch.stack([model.grounding_encoder.preprocess_image(g_px)]).to(model.torch_dtype)
    return g_px


def rollout(model, processor, image, prompt_text, gen_kwargs, device):
    """Sampled rollout (Qwen3VL path) -> (resp_ids, old_logprob, pred_masks, resp_text)."""
    model.processor = processor
    model.seg_token_idx = processor.tokenizer.convert_tokens_to_ids('[SEG]')
    mm_inputs = build_mm_inputs(model, processor, image, prompt_text)
    g_px = build_g_pixel_values(model, image)
    prompt_len = mm_inputs.input_ids.shape[1]
    with torch.no_grad():
        out = model.model.generate(
            **mm_inputs,
            max_new_tokens=gen_kwargs['max_new_tokens'],
            do_sample=True,
            temperature=gen_kwargs['temperature'],
            top_p=gen_kwargs['top_p'],
            output_hidden_states=True,
            output_scores=True,
            return_dict_in_generate=True,
        )
    seq = out.sequences[0]
    resp_ids = seq[prompt_len:]
    old_lp = 0.0
    for t, logits in enumerate(out.scores):
        lps = F.log_softmax(logits.float(), dim=-1)
        old_lp += lps[0, resp_ids[t]].item()
    # masks via [SEG] hidden states -> SAM2
    last_hs = torch.cat([item[-1][0] for item in out.hidden_states], dim=0)
    seg_hs = get_seg_hidden_states(last_hs, seq[:-1], model.seg_token_idx)
    all_seg_hs = model.text_hidden_fcs(seg_hs)
    pred_masks = []
    for shs in all_seg_hs:
        shs = shs.unsqueeze(0)
        sam_states = model.grounding_encoder.get_sam2_embeddings(g_px)
        pred = model.grounding_encoder.language_embd_inference(sam_states, [shs])
        w, h = image.size
        pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False)[:, 0]
        pred_masks.append((pred.sigmoid() > 0.5)[0].cpu().numpy())
    resp_text = processor.batch_decode(
        [seq[prompt_len:]], skip_special_tokens=False)[0].strip()
    return resp_ids, old_lp, pred_masks, resp_text


def compute_logprobs(model, processor, image, prompt_text, resp_ids, device):
    """Teacher-forced forward (Qwen3VL path): sum of logprobs of resp_ids."""
    model.processor = processor
    model.seg_token_idx = processor.tokenizer.convert_tokens_to_ids('[SEG]')
    mm_inputs = build_mm_inputs(model, processor, image, prompt_text)
    prompt_len = mm_inputs.input_ids.shape[1]
    full_ids = torch.cat([mm_inputs.input_ids, resp_ids.to(device).unsqueeze(0)], dim=1)
    attention_mask = torch.ones_like(full_ids, dtype=torch.bool)
    labels = torch.full_like(full_ids, -100)
    labels[0, prompt_len:] = resp_ids.to(device)
    out = model.model(
        input_ids=full_ids,
        attention_mask=attention_mask,
        pixel_values=mm_inputs.pixel_values,
        image_grid_thw=mm_inputs.image_grid_thw,
        labels=labels,
        use_cache=False,
    )
    logits = out.logits[0]
    lps = F.log_softmax(logits.float(), dim=-1)
    resp_tokens = resp_ids.to(device)
    lp = torch.zeros((), device=device)
    for i in range(len(resp_tokens)):
        lp = lp + lps[prompt_len - 1 + i, resp_tokens[i]]
    return lp


def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    if n_out == 0:
        return hidden_states[0:0]
    return hidden_states[-n_out:][seg_mask]


def main():
    args = parse_args()
    rank, world = setup()
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    device = f'cuda:{rank}'
    os.makedirs(args.save_dir, exist_ok=True)

    if rank == 0:
        print(f'loading model {args.model_path}', flush=True)
    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model.processor = processor
    model.seg_token_idx = processor.tokenizer.convert_tokens_to_ids('[SEG]')

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias='none',
        task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        modules_to_save=['lm_head', 'embed_tokens'],
    )
    model.model = get_peft_model(model.model, lora_cfg)
    model.model.print_trainable_parameters()
    trainable = [p for p in model.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    present, absent = load_manifest(args.max_prompts, args.present_ratio)
    if rank == 0:
        print(f'present={len(present)} absent={len(absent)}', flush=True)
    prompts = [dict(x) for x in present] + [dict(x) for x in absent]
    my_prompts = [p for i, p in enumerate(prompts) if i % world == rank]
    if rank == 0:
        print(f'rank{rank} prompts={len(my_prompts)}', flush=True)

    log_path = os.path.join(args.save_dir, 'grpo_train.log')
    logf = open(log_path, 'w') if rank == 0 else None
    step = 0
    rng = random.Random(args.seed + rank)
    while step < args.steps and len(my_prompts) > 0:
        chosen = rng.sample(my_prompts, min(args.batch_prompts, len(my_prompts)))
        total_loss = torch.tensor(0.0, device=device)
        stat = {'r_mean': 0.0, 'n': 0, 'halluc': 0, 'iou_present': 0.0, 'n_present': 0}
        for sample in chosen:
            image = Image.open(sample['file']).convert('RGB')
            prompt_text = build_prompt(sample['query'])
            resps, old_lps, rewards = [], [], []
            for g in range(args.group_size):
                resp_ids, old_lp, pred_masks, resp_text = rollout(
                    model, processor, image, prompt_text,
                    dict(max_new_tokens=args.max_new_tokens,
                         temperature=args.temperature, top_p=args.top_p),
                    device)
                if sample['kind'] == 'present':
                    r_iou, r_acc, nm = reward_present(pred_masks, sample['gt'])
                    r = 0.7 * r_iou + 0.3 * r_acc
                else:
                    r, area, nm = reward_absent(pred_masks)
                    if nm > 0:
                        stat['halluc'] += 1
                r += 0.05 * format_reward(resp_text, sample['kind'])
                resps.append(resp_ids); old_lps.append(old_lp); rewards.append(r)
            r_t = torch.tensor(rewards, device=device)
            mean_r = r_t.mean(); std_r = r_t.std().clamp_min(1e-4)
            adv = (r_t - mean_r) / (std_r + 1e-4)
            ref_lps, new_lps = [], []
            for resp_ids in resps:
                with torch.no_grad():
                    model.model.disable_adapter()
                    rlp = compute_logprobs(model, processor, image, prompt_text, resp_ids, device)
                    model.model.set_adapter('default')
                ref_lps.append(rlp)
            for resp_ids in resps:
                new_lps.append(compute_logprobs(model, processor, image, prompt_text, resp_ids, device))
            new_lp_t = torch.stack(new_lps)
            old_lp_t = torch.tensor(old_lps, device=device)
            ref_lp_t = torch.stack(ref_lps).detach()
            ratio = torch.clamp(torch.exp(new_lp_t - old_lp_t),
                                1.0 - args.clip_eps, 1.0 + args.clip_eps)
            grpo_loss = -torch.mean(adv * ratio)
            kl = torch.mean(torch.exp(new_lp_t - ref_lp_t) - (new_lp_t - ref_lp_t) - 1.0)
            total_loss = total_loss + grpo_loss + args.kl_coef * kl
            stat['r_mean'] += float(r_t.mean()); stat['n'] += 1
            if sample['kind'] == 'present':
                stat['iou_present'] += float(r_t.mean()); stat['n_present'] += 1
        total_loss = total_loss / max(len(chosen), 1)
        total_loss.backward()
        for p in trainable:
            if p.grad is not None:
                dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                p.grad /= world
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        opt.zero_grad()
        step += 1
        if rank == 0:
            r_mean = stat['r_mean'] / max(stat['n'], 1)
            iou_p = stat['iou_present'] / max(stat['n_present'], 1)
            msg = (f'[step {step}] loss={total_loss.item():.4f} r_mean={r_mean:.4f} '
                   f'iou_present={iou_p:.4f} halluc={stat["halluc"]}/{stat["n"]} kl={kl.item():.4f}')
            print(msg, flush=True)
            logf.write(msg + '\n'); logf.flush()
            if step % args.save_every == 0:
                ckpt = os.path.join(args.save_dir, f'grpo_step{step}.pt')
                torch.save({'step': step, 'lora': model.model.state_dict(),
                            'opt': opt.state_dict()}, ckpt)
                print(f'saved {ckpt}', flush=True)
    if rank == 0:
        logf.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
