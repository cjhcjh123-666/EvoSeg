#!/usr/bin/env python
"""Fuse a saved EvoSeg GRPO LoRA checkpoint into the base HF Sa2VA model.

The GRPO trainer (grpo_train.py) saves `model.model.state_dict()` of the
peft-wrapped Qwen3VLForConditionalGeneration (LoRA adapters on q/k/v/o/gate/
up/down + lm_head/embed_tokens via modules_to_save). This script:

  1. loads the base HF Sa2VA model,
  2. re-wraps `model.model` with the same LoraConfig,
  3. loads the saved adapter state (strict=False; the saved dict also carries
     untouched base weights, which are identical to the freshly loaded ones),
  4. merge_and_unload() -> plain merged LLM,
  5. saves a self-contained HF model dir that the existing eval harnesses
     (eval_no_object_halluc.py, sa2va_eval_refcoco.py, sa2va_eval_ref_vos.py)
     can consume directly.

Usage (8 GPUs recommended, memory: the 4B model fits on one A800 bf16):
  python projects/evoseg/rl/grpo_fuse_eval.py <base_hf_dir> \\
      --ckpt /path/to/grpo_step150.pt \\
      --save-dir /path/to/EvoSeg-GRPO-merged
"""
import argparse
import os

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoProcessor, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('model_path', help='base HF Sa2VA model dir (must match the '
                                      'checkpoint the GRPO trainer started from).')
    p.add_argument('--ckpt', required=True, help='saved grpo_step*.pt')
    p.add_argument('--save-dir', required=True)
    p.add_argument('--lora-r', type=int, default=32)
    p.add_argument('--lora-alpha', type=int, default=64)
    p.add_argument('--local_rank', '--local-rank', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(args.local_rank)
    os.makedirs(args.save_dir, exist_ok=True)

    model = AutoModel.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        use_flash_attn=True, trust_remote_code=True).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    # Mirror grpo_train.py exactly so the state-dict keys match.
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05, bias='none',
        task_type='CAUSAL_LM',
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                        'gate_proj', 'up_proj', 'down_proj'],
        modules_to_save=['lm_head', 'embed_tokens'],
    )
    model.model = get_peft_model(model.model, lora_cfg)

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = ckpt['lora'] if (isinstance(ckpt, dict) and 'lora' in ckpt) else ckpt
    missing, unexpected = model.model.load_state_dict(sd, strict=False)
    step = ckpt.get('step', '?') if isinstance(ckpt, dict) else '?'
    print(f'loaded ckpt step={step} missing={len(missing)} unexpected={len(unexpected)}')
    # Any unexpected key would mean the base architecture changed between the
    # GRPO run and this script -> refuse to silently produce a broken merge.
    if unexpected:
        print('unexpected keys (sample):', unexpected[:10])
        raise RuntimeError(
            f'{len(unexpected)} unexpected keys when loading LoRA state. '
            'Make sure --lora-r/--lora-alpha and the base model match the GRPO run.')
    if missing:
        print('missing keys (sample):', missing[:10])
        raise RuntimeError(
            f'{len(missing)} missing keys when loading LoRA state.')

    model.model = model.model.merge_and_unload()
    print('merged LoRA into base model', flush=True)

    model.save_pretrained(args.save_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.save_dir)
    processor.save_pretrained(args.save_dir)
    print('saved merged HF model to', args.save_dir, flush=True)
    print('next steps:')
    print(f'  python projects/evoseg/eval/eval_no_object_halluc.py {args.save_dir} '
          f'--save {args.save_dir}_halluc.json')
    print(f'  python projects/sa2va/evaluation/sa2va_eval_refcoco.py {args.save_dir} '
          f'--dataset grefcoco --split val ...')
    print(f'  python projects/sa2va/evaluation/sa2va_eval_refcoco.py {args.save_dir} '
          f'--dataset refcoco --split val ... (and refcoco_plus/refcocog)')
    print(f'  python projects/sa2va/evaluation/sa2va_eval_ref_vos.py {args.save_dir} '
          f'--dataset MEVIS_U ... / --dataset REFYTVOS ...')


if __name__ == '__main__':
    main()
