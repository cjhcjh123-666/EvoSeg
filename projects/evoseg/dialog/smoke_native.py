"""Strict public foundation loading and first-round mask decode, not a score table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from projects.samtok.models import DirectResize, SAM2Config, VQ_SAM2, VQ_SAM2Config
from projects.evoseg.restart.prepare_foundation import atomic_json
from .public_data import IMAGE, parse_conversation


def run(args):
    directory = Path(args.model_dir)
    assets = json.loads(Path(args.assets).read_text())
    if assets['status'] != 'ASSETS_DOWNLOADED':
        raise RuntimeError('public foundation inventory is not complete')
    model, loading = Qwen3VLForConditionalGeneration.from_pretrained(
        directory, dtype=torch.bfloat16, local_files_only=True,
        output_loading_info=True, attn_implementation='sdpa')
    problems = {key: loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')
                if loading.get(key)}
    if problems:
        raise RuntimeError(f'incomplete language foundation loading: {problems}')
    model = model.cuda().eval()
    processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
    config = VQ_SAM2Config(sam2_config=SAM2Config(ckpt_path=str(directory / 'sam2.1_hiera_large.pt')),
                           codebook_size=256, codebook_depth=2, shared_codebook=False, latent_dim=256)
    tokenizer = VQ_SAM2(config)
    weights = torch.load(directory / 'mask_tokenizer_256x2.pth', map_location='cpu', weights_only=True)
    tokenizer.load_state_dict(weights, strict=True)
    del weights
    tokenizer = tokenizer.cuda().eval().requires_grad_(False)
    records = json.loads(Path(args.dialogues).read_text())
    record = records[0]
    first = parse_conversation(record)[0]
    image_path = Path(args.coco_images) / Path(first['source_image']).name
    with Image.open(image_path) as original:
        image = original.convert('RGB')
    instruction = IMAGE.sub('', first['query']).strip()
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image},
                                           {'type': 'text', 'text': instruction}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors='pt').to(model.device)
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    answer = processor.decode(generated[0, inputs.input_ids.shape[1]:], skip_special_tokens=False)
    pairs = re.findall(r'<\|mt_start\|>\s*<\|mt_(\d{4})\|>\s*<\|mt_(\d{4})\|>\s*<\|mt_end\|>', answer)
    if len(pairs) != 1:
        raise RuntimeError('first-round native smoke expected one valid mask token pair: ' + answer)
    first_code, second_code = map(int, pairs[0])
    if not 0 <= first_code < 256 or not 256 <= second_code < 512:
        raise RuntimeError('mask codebook depth or token range mismatch')
    pixels = DirectResize(1024).apply_image(np.asarray(image))
    pixels = torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None].to('cuda', dtype=tokenizer.dtype)
    codes = torch.tensor([[first_code, second_code - 256]], device='cuda')
    with torch.inference_mode():
        logits = tokenizer.forward_with_codes(pixels, codes)
        masks = F.interpolate(logits, size=(image.height, image.width), mode='bilinear') > .5
    output = {'status': 'NATIVE_FIRST_ROUND_SMOKE_PASS', 'foundation_revision': assets['model_revision'],
              'language_loading_info': loading, 'pixel_tokenizer_strict_loading': True,
              'image': str(image_path), 'native_answer': answer, 'mask_shape': list(masks.shape),
              'mask_pixels': int(masks.sum()), 'gpu_peak_gb': torch.cuda.max_memory_allocated() / 1e9,
              'public_benchmark_complete': False, 'training_started': False,
              'next_gate': 'compile complete history, official metrics, and full multi-round baseline'}
    atomic_json(Path(args.output), output)
    print(json.dumps(output), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    base = Path('/9950backfile/chenjiahui/evo_artifacts')
    parser.add_argument('--model-dir', default=str(base / 'models/Qwen3-VL-8B-SAMTok-official'))
    parser.add_argument('--assets', default=str(base / 'results/evoseg_dialog_20261006/ASSETS.json'))
    parser.add_argument('--dialogues', default=str(base / 'datasets/SegLLM-official/conversations_folder/all_data_mix_val/mr_refcoco_val.json'))
    parser.add_argument('--coco-images', default=str(base / 'datasets/coco2014/train2014'))
    parser.add_argument('--output', default=str(base / 'results/evoseg_dialog_20261006/NATIVE_SMOKE.json'))
    run(parser.parse_args())


if __name__ == '__main__':
    main()
