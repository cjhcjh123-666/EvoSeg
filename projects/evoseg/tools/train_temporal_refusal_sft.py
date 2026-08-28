"""HF LoRA SFT: temporal-aware refusal training (full seq, answer-only CE)."""
import argparse, json, os, random, time
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoProcessor, AutoTokenizer

MODEL = '/9950backfile/chenjiahui/evo_artifacts/models/EvoSeg-Qwen3-VL-4B-TEG'
JPEG = '/9950backfile/chenjiahui/evo_artifacts/datasets/ref_youtube_vos/extracted/train/JPEGImages'


class TRDataset(Dataset):
    def __init__(self, exs):
        self.exs = exs
    def __len__(self):
        return len(self.exs)
    def __getitem__(self, i):
        e = self.exs[i]
        ff = e['frames'][-2:] if len(e['frames']) > 2 else e['frames']  # last 2 frames
        frames = [Image.open(os.path.join(JPEG, e['vid'], f + '.jpg')).convert('RGB')
                  for f in ff]
        # answer is the assistant turn in conversations
        answer = e['conversations'][1]['content']
        query = e['conversations'][0]['content'][-1]['text']
        return frames, query, answer


def collate(batch, proc, min_px, max_px):
    texts, images, answers = [], [], []
    for frames, query, answer in batch:
        content = [{'type': 'image', 'image': fr} for fr in frames]
        content.append({'type': 'text', 'text': query})
        messages = [{'role': 'user', 'content': content},
                    {'role': 'assistant', 'content': answer}]
        texts.append(proc.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=False))
        images.append(frames)
        answers.append(answer)
    # training only needs the existence decision, not mask quality: cap res
    train_max_px = 768 * 28 * 28    # ~768x768 per frame
    mm = proc(text=texts, images=images, videos=None, padding=True,
              return_tensors='pt', min_pixels=min_px, max_pixels=train_max_px)
    return mm, answers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/9950backfile/chenjiahui/evo_artifacts/datasets/s4b/pixel_llm_data/temporal_refusal/annotations.json')
    ap.add_argument('--out-dir', default='/9950backfile/chenjiahui/evo_artifacts/checkpoints/tr_sft')
    ap.add_argument('--max-examples', type=int, default=40000)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--local-rank', '--local_rank', type=int, default=0)
    args = ap.parse_args()
    import torch.distributed as dist
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    torch.manual_seed(args.seed); random.seed(args.seed)
    if world > 1:
        dist.init_process_group('nccl', init_method='env://')
    torch.cuda.set_device(rank)

    data = json.load(open(args.data))['examples']
    pos = [e for e in data if e['presence']]
    neg = [e for e in data if not e['presence']]
    random.shuffle(pos); random.shuffle(neg)
    half = args.max_examples // 2
    exs = pos[:half] + neg[:half]
    random.shuffle(exs)
    if rank == 0:
        npos = sum(1 for e in exs if e['presence'])
        print(f'total examples: {len(exs)} (present={npos} absent={len(exs)-npos})', flush=True)

    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True, use_flash_attn=True, trust_remote_code=True).to(rank)
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    # LoRA on the underlying Qwen3VL LLM (Sa2VA's outer wrapper has no trainable forward)
    llm = model.model
    lora = LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, bias='none',
                      task_type='CAUSAL_LM',
                      target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    llm = get_peft_model(llm, lora)
    # llm.gradient_checkpointing_enable()
    llm.train()
    opt = torch.optim.AdamW([p for p in llm.parameters() if p.requires_grad], lr=args.lr)
    model_min_pixels = model.min_pixels
    model_max_pixels = model.max_pixels

    ds = TRDataset(exs)
    sampler = torch.utils.data.distributed.DistributedSampler(ds, shuffle=True) if world > 1 else None
    dl = DataLoader(ds, batch_size=args.batch, shuffle=(sampler is None), sampler=sampler,
                    collate_fn=lambda b: collate(b, proc, model_min_pixels, model_max_pixels))

    os.makedirs(args.out_dir, exist_ok=True)
    step = 0
    for ep in range(args.epochs):
        if sampler: sampler.set_epoch(ep)
        t0 = time.time(); tot = 0; nb = 0
        for mm, answers in dl:
            for k, v in mm.items():
                if isinstance(v, torch.Tensor):
                    mm[k] = v.to(rank)
            labels = mm['input_ids'].clone()
            # mask prompt tokens; keep only the assistant answer tokens
            # answer token count: tokenize the answers (without images)
            ans_ids = proc.tokenizer(answers, add_special_tokens=False)['input_ids']
            for i, a in enumerate(ans_ids):
                n = len(a)
                # last n tokens of this sequence (before pad) are the answer
                seq = mm['input_ids'][i]
                valid = (seq != proc.tokenizer.pad_token_id).sum().item()
                labels[i, :valid - n] = -100
            labels[labels == proc.tokenizer.pad_token_id] = -100
            out = llm(**mm, labels=labels)
            loss = out.loss
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1; step += 1
            if rank == 0 and step % 50 == 0:
                print(f'[ep{ep} step{step}] loss={tot/nb:.4f} {(time.time()-t0)/60:.1f}min', flush=True)
    if rank == 0:
        llm.save_pretrained(os.path.join(args.out_dir, 'lora'))
        print(f'DONE -> {args.out_dir}/lora', flush=True)


if __name__ == '__main__':
    main()
