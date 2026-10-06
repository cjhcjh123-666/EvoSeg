"""Real released SAM2 weights: train-head parity and absent-gate gradients."""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from safetensors import safe_open
import torch
import torch.nn.functional as F

from projects.sasasa2va.hf.models.sam2 import SAM2
from .native_model import native_training_masks
from .prepare_foundation import atomic_json


def official_method():
    path = Path(__file__).resolve().parents[2] / 'sa2va/models/extension/sam2_base.py'
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'SAM2Base')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_forward_sam_heads')
    namespace = {'torch': torch, 'F': F, 'NO_OBJ_SCORE': -1024.}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['_forward_sam_heads']


def audit(model_dir):
    torch.cuda.set_device(0)
    directory = Path(model_dir)
    index = json.loads((directory / 'model.safetensors.index.json').read_text())['weight_map']
    prefix = 'grounding_encoder.'
    names = [key for key in index if key.startswith(prefix)]
    weights = {}
    for shard in sorted({index[key] for key in names}):
        with safe_open(str(directory / shard), framework='pt', device='cpu') as reader:
            for key in names:
                if index[key] == shard:
                    weights[key[len(prefix):]] = reader.get_tensor(key)
    encoder = SAM2(samurai_mode=False)
    encoder.load_state_dict(weights, strict=True)
    del weights
    encoder = encoder.cuda().to(torch.bfloat16).eval().requires_grad_(False)
    sam = encoder.sam2_model
    torch.manual_seed(42)
    features = torch.randn(2, 256, 64, 64, device='cuda', dtype=torch.bfloat16)
    high_res = [torch.randn(2, 32, 256, 256, device='cuda', dtype=torch.bfloat16),
                torch.randn(2, 64, 128, 128, device='cuda', dtype=torch.bfloat16)]
    language = torch.randn(2, 1, 256, device='cuda', requires_grad=True)

    def forced_negative_presence(module, inputs, outputs):
        # Replace only a diagnostic output, not model weights or saved data.
        return (*outputs[:-1], -torch.ones_like(outputs[-1]))

    hook = sam.sam_mask_decoder.register_forward_hook(forced_negative_presence)
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            reference = official_method()(sam, features, high_res_features=high_res,
                                           language_embd=language, multimask_output=True)[3]
            repaired = native_training_masks(sam, features, high_res, language, True)
            inference = sam._forward_sam_heads(features, high_res_features=high_res,
                                               language_embd=language, multimask_output=True)[3]
        torch.testing.assert_close(repaired, reference, rtol=0, atol=0)
        if not torch.all(inference == -1024):
            raise RuntimeError('diagnostic did not exercise the original inference hard gate')
        loss = F.binary_cross_entropy_with_logits(repaired, torch.ones_like(repaired))
        loss.backward()
        norm = float(language.grad.abs().sum())
        if not torch.isfinite(language.grad).all() or norm <= 0:
            raise RuntimeError('negative presence still erased training prompt gradients')
        return {'status': 'PASS', 'strict_released_sam_tensors': len(names),
                'author_training_max_difference': float((reference - repaired).abs().max()),
                'negative_presence_prompt_gradient_l1': norm,
                'inference_hard_gate_unchanged': True,
                'synthetic_tensors_scope': 'unit diagnostic only; not training data',
                'gpu_peak_gb': torch.cuda.max_memory_allocated() / 1e9}
    finally:
        hook.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = audit(args.model_dir)
    atomic_json(Path(args.output), output)
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
