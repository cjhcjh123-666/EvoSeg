"""CPU-only structural SAM/point-gradient smoke; NOT a quality experiment.

The visual token fixture and code pair are synthetic unit-test inputs, never
training data or predictions. The cached SAM features contain image evidence
only. This lets us test the real published decoder without interrupting GPUs.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from projects.samtok.models import SAM2Config, VQ_SAM2, VQ_SAM2Config
from projects.evoseg.restart.prepare_foundation import atomic_json
from .read_alignment import SasPConfig, similarity_as_points


def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    directory = Path(args.model_dir)
    config = VQ_SAM2Config(sam2_config=SAM2Config(ckpt_path=str(directory / 'sam2.1_hiera_large.pt')),
                          codebook_size=256, codebook_depth=2, shared_codebook=False, latent_dim=256)
    started = time.time()
    codec = VQ_SAM2(config).float().eval().requires_grad_(False)
    state = torch.load(directory / 'mask_tokenizer_256x2.pth', map_location='cpu', weights_only=True)
    codec.load_state_dict(state, strict=True)
    del state
    states = torch.load(args.sam_states, map_location='cpu', weights_only=True)
    states['current_vision_feats'] = [feature.float() for feature in states['current_vision_feats']]
    prompt = codec.deconcate_quant_embed(codec.quantizer.embed_code(torch.tensor([[1, 2]]))[:, None])
    prompt = prompt.reshape(1, codec.num_mask_tokens, -1)
    with torch.no_grad():
        native = codec.model.inject_language_embd(states, prompt, nf_nobj=(1, 1))
        explicit_none = codec.model.inject_language_embd(states, prompt, nf_nobj=(1, 1), point_inputs=None)
    if not torch.equal(native, explicit_none):
        raise RuntimeError('default native decoder is not exactly preserved')
    features = torch.randn(16, 8, requires_grad=True)
    semantic = torch.randn(8, requires_grad=True)
    points, audit = similarity_as_points(features, semantic, (4, 4), (24, 32),
                                        config=SasPConfig(downsample=1, max_points=2))
    logits = codec.model.inject_language_embd(states, prompt, nf_nobj=(1, 1), point_inputs=points)
    if not torch.isfinite(logits).all():
        raise RuntimeError('non-finite real SAM output with point prompts')
    logits.square().mean().backward()
    gradients = {'visual_fixture': float(features.grad.abs().sum()),
                 'semantic_fixture': float(semantic.grad.abs().sum())}
    if not all(v > 0 for v in gradients.values()):
        raise RuntimeError('pixel-loss gradient did not reach similarity features')
    report = {'status': 'STRUCTURAL_SMOKE_PASSED', 'device': 'cpu',
              'real_published_decoder_loaded_strictly': True, 'native_no_points_exact': True,
              'pixel_loss_to_similarity_gradients': gradients, 'point_audit': audit,
              'max_abs_mask_change': float((logits.detach() - native).abs().max()),
              'elapsed_seconds': time.time() - started, 'quality_measurement': False,
              'training_performed': False, 'synthetic_test_fixture_is_not_training_data': True}
    atomic_json(Path(args.output), report)
    print(json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', default='/9950backfile/chenjiahui/evo_artifacts/models/Qwen3-VL-8B-SAMTok-official')
    p.add_argument('--sam-states', required=True)
    p.add_argument('--output', required=True)
    run(p.parse_args())


if __name__ == '__main__':
    main()
