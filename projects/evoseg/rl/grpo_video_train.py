#!/usr/bin/env python
"""EvoSeg Video Faithfulness GRPO (temporal absence / identity / global absence).

Rolls out on video faithfulness samples (Ref-YT-VOS train manifest built by
projects/evoseg/tools/build_video_faithfulness_train.py). The policy controls
the [SEG] vs refusal token; SAM2 produces per-frame masks when [SEG] is emitted.

Rewards (per rollout):
  r = 0.7 * reward_temporal_presence  (mean IoU on present frames)
      + lambda_absent * reward_temporal_absence  (negative, pixel area on
        absent frames -> the model must STOP propagating after disappearance)
      + 0.05 * format_reward

GRPO: group-relative advantage + PPO-clip ratio + KL to the frozen (non-LoRA)
reference policy, identical to the image trainer projects/evoseg/rl/grpo_train.py.

Usage (8 GPUs):
  torchrun --nproc_per_node=8 projects/evoseg/rl/grpo_video_train.py \
      /9950backfile/chenjiahui/evo_artifacts/models/EvoSeg-Qwen3-VL-4B-VideoFaithful \
      --manifest /9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/faithfulness_train.json \
      --save-dir /9950backfile/chenjiahui/evo_artifacts/results/s4b/evoseg_grpo_video
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
from qwen_vl_utils import process_vision_info

REPO = '/9950backfile/chenjiahui/EvoSeg'
sys.path.insert(0, os.path.join(REPO, 'projects', 'evoseg', 'rl'))
from rewards import (  # noqa: E402
    reward_temporal_presence, reward_temporal_absence, format_reward)

ANNROOT = ('/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/'
           'extracted/train/Annotations')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('model_path')
    p.add_argument('--manifest', required=True)
    p.add_argument('--save-dir', default='/9950backfile/chenjiahui/evo_artifacts/results/s4b/evoseg_grpo_video')
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--batch-prompts', type=int, default=1, help='videos per rank per step')
    p.add_argument('--group-size', type=int, default=4)
    p.add_argument('--max-cases', type=int, default=None)
    p.add_argument('--lr', type=float, default=2e-6)
    p.add_argument('--kl-coef', type=float, default=0.05)
    p.add_argument('--clip-eps', type=float, default=0.2)
    p.add_argument('--lora-r', type=int, default=32)
    p.add_argument('--lora-alpha', type=int, default=64)
    p.add_argument('--temperature', type=float, default=0.9)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--max-new-tokens', type=int, default=32)
    p.add_argument('--lambda-absent', type=float, default=0.7)
    p.add_argument('--max-frames', type=int, default=12, help='cap frames per video rollout (memory)')
    p.add_argument('--llm-max-pixels', type=int, default=401408, help='LLM video input max pixels (smaller = fewer tokens)')
    p.add_argument('--save-every', type=int, default=30)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--local_rank', '--local-rank', type=int, default=0)
    return p.parse_args()


def setup():
    dist.init_process_group('nccl', timeout=datetime_timedelta_minutes(120))
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world


def datetime_timedelta_minutes(mins):
    import datetime
    return datetime.timedelta(minutes=mins)


def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    if n_out == 0:
        return hidden_states[0:0]
    return hidden_states[-n_out:][seg_mask]


def load_cases(manifest, max_cases=None, seed=0):
    data = json.load(open(manifest))
    cases = data['cases']
    # prefer temporal_absence (the differentiator) + identity_swap + global_absence
    rng = random.Random(seed)
    rng.shuffle(cases)
    if max_cases:
        cases = cases[:max_cases]
    return cases


LLM_MAX_PIXELS = 401408  # overridden by --llm-max-pixels


def build_mm_inputs(model, processor, frames, text):
    messages = [{
        'role': 'user',
        'content': [{'type': 'image', 'image': im} for im in frames[:5]]
                   + [{'type': 'text', 'text': text}],
    }]
    processsed_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    mm_inputs = processor(
        text=[processsed_text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors='pt',
        min_pixels=model.min_pixels, max_pixels=LLM_MAX_PIXELS,
    ).to(next(model.parameters()).device)
    return mm_inputs


def build_g_pixel_values(model, frames):
    gpx = []
    for im in frames:
        g = np.array(im)
        g = model.extra_image_processor.apply_image(g)
        gpx.append(torch.from_numpy(g).permute(2, 0, 1).contiguous())
    gpx = torch.stack([model.grounding_encoder.preprocess_image(p) for p in gpx])
    return gpx.to(model.torch_dtype)


def rollout_video(model, processor, frames, prompt_text, gen_kwargs, device):
    """Sampled video rollout -> (resp_ids, old_logprob, pred_masks, resp_text)."""
    model.processor = processor
    model.seg_token_idx = processor.tokenizer.convert_tokens_to_ids('[SEG]')
    mm_inputs = build_mm_inputs(model, processor, frames, prompt_text)
    g_px = build_g_pixel_values(model, frames)
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
    last_hs = torch.cat([item[-1][0] for item in out.hidden_states], dim=0)
    seg_hs = get_seg_hidden_states(last_hs, seq[:-1], model.seg_token_idx)
    all_seg_hs = model.text_hidden_fcs(seg_hs)
    pred_masks = []
    if len(all_seg_hs) > 0:
        # one [SEG] embedding conditions every frame's mask (SAM2 path)
        shs = all_seg_hs[0].unsqueeze(0)
        sam_states = model.grounding_encoder.get_sam2_embeddings(g_px)
        pred = model.grounding_encoder.language_embd_inference(sam_states, [shs])
        for t in range(len(frames)):
            p = pred[t]
            w, h = frames[t].size
            p = F.interpolate(p.unsqueeze(0), size=(h, w), mode='bilinear',
                              align_corners=False)[:, 0]
            pred_masks.append((p.sigmoid() > 0.5)[0].cpu().numpy())
    resp_text = processor.batch_decode(
        [seq[prompt_len:]], skip_special_tokens=False)[0].strip()
    del out, g_px, mm_inputs
    torch.cuda.empty_cache()
    return resp_ids, old_lp, pred_masks, resp_text


def compute_logprobs_video(model, processor, frames, prompt_text, resp_ids, device):
    model.processor = processor
    model.seg_token_idx = processor.tokenizer.convert_tokens_to_ids('[SEG]')
    mm_inputs = build_mm_inputs(model, processor, frames, prompt_text)
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
    lp = torch.zeros((), device=device)
    for i in range(len(resp_ids)):
        lp = lp + lps[prompt_len - 1 + i, resp_ids[i]]
    return lp


def load_gt_masks(case, frames):
    """Per-frame GT masks (H,W) for the referenced instance (None if no instance)."""
    if case.get('instance') is None:
        return None
    inst = int(case['instance'])
    masks = []
    for f in frames:
        p = os.path.join(ANNROOT, case['video_id'], f + '.png')
        if not os.path.isfile(p):
            masks.append(None)
            continue
        a = np.array(Image.open(p).convert('L'))
        masks.append((a == inst).astype(np.uint8))
    return masks




def subsample_frames(frames, presence, max_frames):
    """Cap frames to max_frames while preserving the presence->absence boundary."""
    n = len(frames)
    if n <= max_frames:
        return list(range(n))
    idxs = list(np.linspace(0, n - 1, max_frames).round().astype(int))
    idxs = sorted(set(i for i in idxs if i < n))
    p = np.array(presence, dtype=bool)
    if np.any(p):
        t0 = int(np.where(p)[0][0]); t1 = int(np.where(p)[0][-1])
        if t1 < n - 1 and (n - 1) not in idxs:
            idxs.append(n - 1)
        if t0 > 0 and 0 not in idxs:
            idxs.append(0)
    return sorted(set(i for i in idxs if i < n))

def main():
    args = parse_args()
    global LLM_MAX_PIXELS
    LLM_MAX_PIXELS = args.llm_max_pixels
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
    # gradient checkpointing: video logprob backward otherwise explodes memory
    try:
        model.model.gradient_checkpointing_enable()
        model.model.enable_input_require_grads()
        print('gradient checkpointing enabled', flush=True)
    except Exception as e:
        print(f'gc enable failed: {e}', flush=True)

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

    cases = load_cases(args.manifest, args.max_cases, args.seed)
    my_cases = [c for i, c in enumerate(cases) if i % world == rank]
    if rank == 0:
        print(f'cases={len(cases)} per_rank={len(my_cases)}', flush=True)

    JPEGROOT = ('/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/'
                'extracted/train/JPEGImages')
    log_path = os.path.join(args.save_dir, 'grpo_video.log')
    logf = open(log_path, 'w') if rank == 0 else None
    step = 0
    rng = random.Random(args.seed + rank)
    while step < args.steps and len(my_cases) > 0:
        chosen = rng.sample(my_cases, min(args.batch_prompts, len(my_cases)))
        total_loss = torch.tensor(0.0, device=device)
        stat = {'r_mean': 0.0, 'n': 0, 'halluc_frames': 0, 'iou_present': 0.0,
                'n_present': 0}
        for case in chosen:
            vid = case['video_id']
            sub = subsample_frames(case['frames'], case['presence'], args.max_frames)
            sel_frames_list = [case['frames'][i] for i in sub]
            frames = [Image.open(os.path.join(JPEGROOT, vid, f + '.jpg')).convert('RGB')
                      for f in sel_frames_list]
            presence = np.array(case['presence'], dtype=bool)[sub]
            present_idx = list(np.where(presence)[0])
            absent_idx = list(np.where(~presence)[0])
            gt_masks = load_gt_masks(case, sel_frames_list)
            query = case['query'].lower().replace('.', '').strip()
            prompt_text = f'<image>\n Please segment {query} in this video.'
            records, rewards = [], []
            gen_kwargs = dict(max_new_tokens=args.max_new_tokens,
                              temperature=args.temperature, top_p=args.top_p)
            for g in range(args.group_size):
                resp_ids, old_lp, pred_masks, resp_text = rollout_video(
                    model, processor, frames, prompt_text, gen_kwargs, device)
                n_frames = len(frames)
                pred_presence = [False] * n_frames
                for t in range(min(n_frames, len(pred_masks))):
                    pred_presence[t] = bool((pred_masks[t] > 0).sum())
                r_iou, n_pres = reward_temporal_presence(
                    pred_masks, gt_masks, present_idx)
                r_abs, area_abs, n_abs = reward_temporal_absence(
                    pred_masks, absent_idx)
                r = 0.7 * r_iou + args.lambda_absent * r_abs
                r += 0.05 * format_reward(resp_text, 'video')
                # hallucinated frame count on absent frames
                halluc = sum(1 for t in absent_idx if pred_presence[t])
                stat['halluc_frames'] += halluc
                stat['iou_present'] += r_iou
                stat['n_present'] += max(n_pres, 1)
                records.append({'resp_ids': resp_ids, 'old_lp': old_lp,
                                'prompt': prompt_text})
                rewards.append(r)
            torch.cuda.empty_cache()
            r_t = torch.tensor(rewards, device=device)
            mean_r = r_t.mean(); std_r = r_t.std().clamp_min(1e-4)
            adv = (r_t - mean_r) / (std_r + 1e-4)
            ref_lps, new_lps = [], []
            for rec in records:
                with torch.no_grad():
                    model.model.disable_adapter()
                    rlp = compute_logprobs_video(
                        model, processor, frames, rec['prompt'], rec['resp_ids'], device)
                    model.model.set_adapter('default')
                ref_lps.append(rlp)
            for rec in records:
                new_lps.append(compute_logprobs_video(
                    model, processor, frames, rec['prompt'], rec['resp_ids'], device))
            new_lp_t = torch.stack(new_lps)
            old_lp_t = torch.tensor([rec['old_lp'] for rec in records], device=device)
            ref_lp_t = torch.stack(ref_lps).detach()
            ratio = torch.clamp(torch.exp(new_lp_t - old_lp_t),
                                1.0 - args.clip_eps, 1.0 + args.clip_eps)
            grpo_loss = -torch.mean(adv * ratio)
            kl = torch.mean(torch.exp(new_lp_t - ref_lp_t) - (new_lp_t - ref_lp_t) - 1.0)
            total_loss = total_loss + grpo_loss + args.kl_coef * kl
            stat['r_mean'] += float(r_t.mean()); stat['n'] += 1
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
                   f'iou_present={iou_p:.4f} halluc_frames={stat["halluc_frames"]} '
                   f'kl={kl.item():.4f}')
            print(msg, flush=True)
            logf.write(msg + '\n'); logf.flush()
            torch.cuda.empty_cache()
            if step % args.save_every == 0:
                ckpt = os.path.join(args.save_dir, f'grpo_video_step{step}.pt')
                torch.save({
                    'model': model.model.state_dict(),
                    'step': step,
                }, ckpt)
                print(f'saved {ckpt}', flush=True)
    if rank == 0:
        logf.close()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
