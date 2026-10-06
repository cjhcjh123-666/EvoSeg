"""Strict native SAMTok runtime, reused for cache preparation and inference."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from projects.samtok.models import DirectResize, SAM2Config, VQ_SAM2, VQ_SAM2Config


MASK_PAIR = re.compile(r'<\|mt_start\|>\s*<\|mt_(\d{4})\|>\s*<\|mt_(\d{4})\|>\s*<\|mt_end\|>')


def token_string(codes):
    first, second = [int(code) for code in codes]
    if not 0 <= first < 256 or not 0 <= second < 256:
        raise ValueError('native mask codebook value out of range')
    return f'<|mt_start|><|mt_{first:04d}|><|mt_{second+256:04d}|><|mt_end|>'


def parse_mask_pair(text):
    pairs = MASK_PAIR.findall(text)
    if len(pairs) != 1:
        return None
    first, second = map(int, pairs[0])
    if not 0 <= first < 256 or not 256 <= second < 512:
        return None
    return [first, second - 256]


def adapter_fingerprint(directory):
    if directory is None:
        return 'released_native'
    digest = hashlib.sha256()
    files = sorted(Path(directory).glob('adapter*'))
    if not any(path.suffix == '.safetensors' for path in files):
        raise RuntimeError('adapter weight file missing')
    for path in files:
        digest.update(path.name.encode())
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


class NativeRuntime:
    def __init__(self, model_dir, assets_file, *, load_language=True, adapter_dir=None):
        directory = Path(model_dir)
        assets = json.loads(Path(assets_file).read_text())
        if assets['status'] != 'ASSETS_DOWNLOADED':
            raise RuntimeError('foundation inventory incomplete')
        for file in assets['model_inventory']:
            if (directory / file['name']).stat().st_size != file['size']:
                raise RuntimeError('foundation file size differs from pinned inventory')
        self.revision = assets['model_revision']
        self.adapter_fingerprint = adapter_fingerprint(adapter_dir)
        self.model = self.processor = None
        self.loading = {}
        if load_language:
            model, loading = Qwen3VLForConditionalGeneration.from_pretrained(
                directory, dtype=torch.bfloat16, local_files_only=True,
                output_loading_info=True, attn_implementation='sdpa')
            problems = {key: loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')
                        if loading.get(key)}
            if problems:
                raise RuntimeError(f'incomplete language loading: {problems}')
            if adapter_dir is not None:
                from peft import PeftModel
                # Bare 'cuda' in safetensors resolves to GPU 0 even when the
                # current rank uses another GPU. Load on CPU then move once.
                model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=False, torch_device='cpu')
            self.model = model.cuda().eval()
            self.processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
            self.loading = loading
        config = VQ_SAM2Config(sam2_config=SAM2Config(ckpt_path=str(directory / 'sam2.1_hiera_large.pt')),
                               codebook_size=256, codebook_depth=2, shared_codebook=False, latent_dim=256)
        tokenizer = VQ_SAM2(config)
        state = torch.load(directory / 'mask_tokenizer_256x2.pth', map_location='cpu', weights_only=True)
        tokenizer.load_state_dict(state, strict=True)
        del state
        self.tokenizer = tokenizer.cuda().eval().requires_grad_(False)
        self.image_path = None
        self.image = self.sam_states = None

    @torch.inference_mode()
    def set_image(self, path):
        if str(path) == self.image_path:
            return
        with Image.open(path) as image:
            self.image = image.convert('RGB')
        values = DirectResize(1024).apply_image(np.asarray(self.image))
        pixels = torch.from_numpy(values.copy()).permute(2, 0, 1)[None].to('cuda', dtype=self.tokenizer.dtype)
        pixels = torch.stack([self.tokenizer.model.preprocess_image(pixel) for pixel in pixels])
        self.sam_states = self.tokenizer.model.get_sam2_embeddings(pixels, expand_size=1)
        self.image_path = str(path)

    @torch.inference_mode()
    def encode(self, mask):
        if mask.shape != (self.image.height, self.image.width):
            raise ValueError('reference mask and image dimensions differ')
        yy, xx = np.nonzero(mask)
        if not len(xx):
            return None
        # Same inclusive pixel-extrema convention as torchvision.masks_to_boxes.
        boxes = torch.tensor([[xx.min()/self.image.width, yy.min()/self.image.height,
                               xx.max()/self.image.width, yy.max()/self.image.height]],
                              device='cuda', dtype=torch.float32)
        target = torch.from_numpy(mask.astype(np.float32))[None].cuda()
        embeddings = self.tokenizer.model.encode_mask_box_input(self.sam_states, [target], boxes)
        embeddings = embeddings.reshape(1, 1, -1)
        embeddings = self.tokenizer.concate_mask_embeds(embeddings)
        _, _, codes = self.tokenizer.quantizer(embeddings, freeze_codebook=True)
        result = codes.reshape(-1).cpu().tolist()
        if len(result) != 2:
            raise RuntimeError('native quantizer did not produce two mask codes')
        token_string(result)
        return result

    @torch.inference_mode()
    def decode(self, codes):
        if codes is None:
            return np.zeros((self.image.height, self.image.width), dtype=bool)
        token_string(codes)
        values = torch.tensor([codes], device='cuda')
        embeddings = self.tokenizer.quantizer.embed_code(values)[:, None]
        embeddings = self.tokenizer.deconcate_quant_embed(embeddings)
        embeddings = embeddings.reshape(1, self.tokenizer.num_mask_tokens, -1)
        logits = self.tokenizer.model.inject_language_embd(self.sam_states, embeddings, nf_nobj=(1, 1))
        resized = F.interpolate(logits, size=(self.image.height, self.image.width), mode='bilinear')
        return (resized[0, 0] > .5).cpu().numpy()

    @torch.inference_mode()
    def generate(self, messages):
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors='pt').to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        return self.processor.decode(output[0, inputs.input_ids.shape[1]:], skip_special_tokens=False)
