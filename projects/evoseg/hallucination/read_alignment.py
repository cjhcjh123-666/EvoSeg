"""READ-style SasP interface adaptation, not a reproduction of READ scores.

Reference: Qian et al., READ, CVPR 2025; Apache-2.0 upstream READ.py.
https://github.com/rui-qian/READ/blob/254924020ec4d33ace8575eb1e2a515390530594/model/READ.py

The upstream 7B implementation uses raw dot products, downsample=2, min-max
normalization, thresholds .8/.2, at most 30 positive-first points and an
exp(-squared-distance) weighted continuous interpolation. We retain these
rules, but adapt crop/padding geometry to Qwen's uncropped rectangular grid
and SAMTok's DirectResize. Native mask-end *prediction* states replace READ's
SEG prediction state. No learned module, synthetic label or absence gate is
added. Flat maps produce no points instead of upstream's division by zero.
"""
from dataclasses import asdict, dataclass
import re

import torch
import torch.nn.functional as F


UPSTREAM_REVISION = '254924020ec4d33ace8575eb1e2a515390530594'
ALIGNMENT_VERSION = 'read_sasp_qwen_samtok_v1'


@dataclass(frozen=True)
class SasPConfig:
    downsample: int = 2
    max_points: int = 30
    positive_threshold: float = .8
    negative_threshold: float = .2

    def __post_init__(self):
        if self.downsample < 1 or self.max_points < 1:
            raise ValueError('positive downsample and point budget required')
        if not 0 <= self.negative_threshold < self.positive_threshold <= 1:
            raise ValueError('invalid SasP thresholds')


def similarity_as_points(image_features, semantic, grid, image_shape, *, config=None, sam_size=1024):
    """Image/query-derived features only; no target, candidate mask or GT input.

    Integer selection is piecewise constant (as in READ); continuous point
    coordinates remain differentiable w.r.t. similarities and source features.
    """
    config = config or SasPConfig()
    height, width = map(int, grid)
    image_h, image_w = map(int, image_shape)
    if min(height, width, image_h, image_w, sam_size) < 1:
        raise ValueError('positive image/grid dimensions required')
    if image_features.ndim != 2 or len(image_features) != height * width:
        raise ValueError('image token count and raster grid disagree')
    if semantic.ndim != 1 or semantic.shape[0] != image_features.shape[1]:
        raise ValueError('one semantic state with matching hidden dimension required')
    similarity = image_features.float() @ semantic.float()
    if not torch.isfinite(similarity).all():
        raise ValueError('non-finite spatial similarity')
    dense = similarity.reshape(1, 1, height, width)
    coarse_shape = (max(1, height // config.downsample), max(1, width // config.downsample))
    coarse = F.interpolate(dense, size=coarse_shape, mode='bilinear', align_corners=False)[0, 0]
    span = coarse.max() - coarse.min()
    audit = {'config': asdict(config), 'grid': [height, width],
             'source': 'native_mask_end_prediction_state', 'sam_size': sam_size,
             'uses_ground_truth': False, 'positive': 0, 'negative': 0, 'neutral': 0}
    # Do not invent a foreground anchor when the map has no spatial contrast.
    if float(span.detach()) <= 1e-6 or float((similarity.max() - similarity.min()).detach()) <= 1e-6:
        return None, {**audit, 'skip_reason': 'flat_similarity'}
    values = ((coarse - coarse.min()) / span).flatten()
    positive = torch.where(values >= config.positive_threshold)[0]
    positive = positive[torch.argsort(values[positive], descending=True, stable=True)]
    negative = torch.where(values <= config.negative_threshold)[0]
    negative = negative[torch.argsort(values[negative], stable=True)]
    neutral = torch.where((values > config.negative_threshold) & (values < config.positive_threshold))[0]
    indices = torch.cat([positive, negative, neutral])[:config.max_points]
    labels = torch.cat([torch.ones_like(positive), torch.zeros_like(negative),
                        -torch.ones_like(neutral)])[:len(indices)]
    # Upstream rounds coarse cell centers down to original-image integer pixels.
    selected_x = (((indices % coarse_shape[1]).float() + .5) * image_w / coarse_shape[1]).clamp_max(image_w - 1).floor()
    selected_y = (((indices // coarse_shape[1]).float() + .5) * image_h / coarse_shape[0]).clamp_max(image_h - 1).floor()
    normalized = (dense - similarity.min()) / (similarity.max() - similarity.min())
    full = F.interpolate(normalized, size=(image_h, image_w), mode='bilinear', align_corners=False).flatten()
    yy, xx = torch.meshgrid(torch.arange(image_h, device=full.device, dtype=torch.float32),
                           torch.arange(image_w, device=full.device, dtype=torch.float32), indexing='ij')
    xx, yy = xx.flatten(), yy.flatten()
    points = []
    for x, y in zip(selected_x, selected_y):
        # Stable equivalent of exp(-d^2)*softmax(S), followed by renormalization.
        weights = (full - (xx - x).square() - (yy - y).square()).softmax(0)
        points.append(torch.stack([(xx * weights).sum() * sam_size / image_w,
                                   (yy * weights).sum() * sam_size / image_h]))
    coords = torch.stack(points)[None]
    audit.update(positive=int((labels == 1).sum()), negative=int((labels == 0).sum()),
                 neutral=int((labels == -1).sum()), skip_reason=None)
    return {'point_coords': coords, 'point_labels': labels[None].to(torch.int32)}, audit


def merged_image_features(hidden, inputs, model_config):
    """Validate single-image raster geometry before exposing any spatial map."""
    if inputs.image_grid_thw.shape != (1, 3) or inputs.input_ids.shape[0] != 1:
        raise ValueError('READ alignment requires exactly one image and one query')
    temporal, grid_h, grid_w = map(int, inputs.image_grid_thw[0].tolist())
    merge = model_config.vision_config.spatial_merge_size
    if temporal != 1 or grid_h % merge or grid_w % merge:
        raise ValueError('non-image or invalid merged grid')
    if hidden.shape[:2] != inputs.input_ids.shape:
        raise ValueError('prefill hidden states and input token positions disagree')
    features = hidden[0, inputs.input_ids[0] == model_config.image_token_id]
    grid = (grid_h // merge, grid_w // merge)
    if len(features) != grid[0] * grid[1]:
        raise ValueError('image token count and merged grid disagree')
    return features, grid


def native_prediction_states(generated, prefix_length, processor, codes):
    """Use causal states that predicted generated mt_end, NOT the next step.

    READ's released code similarly indexes states preceding the predicted SEG.
    SAMTok has two VQ codes rather than one SEG; the last pair-conditioned
    prediction state is an explicit, documented backbone-specific adaptation.
    Malformed/non-delimited codes never borrow a different target's state.
    """
    ids = generated.sequences[0, prefix_length:].tolist()
    pieces = processor.tokenizer.convert_ids_to_tokens(ids)
    pairs, end_steps = [], []
    start = None
    pending = []
    for step, piece in enumerate(pieces):
        if piece == '<|mt_start|>':
            start, pending = step, []
        elif start is not None and re.fullmatch(r'<\|mt_\d{4}\|>', piece):
            pending.append(int(piece[5:9]))
        elif piece == '<|mt_end|>' and start is not None:
            if len(pending) == 2 and 0 <= pending[0] < 256 and 256 <= pending[1] < 512:
                pairs.append([pending[0], pending[1] - 256])
                end_steps.append(step)
            start, pending = None, []
    if pairs != codes or len(generated.hidden_states) <= max(end_steps, default=-1):
        return [None] * len(codes)
    return [generated.hidden_states[step][-1][0, -1].float() for step in end_steps]


@torch.inference_mode()
def aligned_predictions(runtime, image_path, query):
    """Paired inference from ONE generation, before any human-label access.

    The released foundation is used without Faithful/interaction/region heads.
    This diagnoses the spatial interface, not our future proposed method.
    Native empty outputs remain empty in both arms: coverage is reported.
    """
    import numpy as np
    from .eval_fresh import parse_native_codes
    from .train_fresh import prefix_inputs
    runtime.set_image(image_path)
    inputs = prefix_inputs(runtime.processor, runtime.image, query).to(runtime.model.device)
    generated = runtime.model.generate(**inputs, max_new_tokens=128, do_sample=False,
        return_dict_in_generate=True, output_hidden_states=True)
    answer = runtime.processor.decode(generated.sequences[0, inputs.input_ids.shape[1]:], skip_special_tokens=False)
    codes, malformed = parse_native_codes(answer)
    shape = (runtime.image.height, runtime.image.width)
    masks = {name: np.zeros(shape, dtype=bool) for name in ('native', 'read_style')}
    audit = {'alignment_version': ALIGNMENT_VERSION, 'upstream_revision': UPSTREAM_REVISION,
             'answer': answer, 'native_codes': codes, 'malformed_codes': malformed,
             'native_empty': not bool(codes), 'points': [], 'weights_changed': False,
             'paper_score_reproduction': False, 'uses_ground_truth': False}
    if not codes:
        return masks, audit
    features, grid = merged_image_features(generated.hidden_states[0][-1], inputs, runtime.model.config)
    states = native_prediction_states(generated, inputs.input_ids.shape[1], runtime.processor, codes)
    for pair, semantic in zip(codes, states):
        value = torch.tensor([pair], device=features.device)
        prompt = runtime.tokenizer.deconcate_quant_embed(runtime.tokenizer.quantizer.embed_code(value)[:, None])
        prompt = prompt.reshape(1, runtime.tokenizer.num_mask_tokens, -1)
        native = runtime.tokenizer.model.inject_language_embd(runtime.sam_states, prompt, nf_nobj=(1, 1))
        if semantic is None:
            points, detail = None, {'skip_reason': 'unmatched_native_code_group'}
        else:
            points, detail = similarity_as_points(features, semantic, grid, shape)
        adapted = native if points is None else runtime.tokenizer.model.inject_language_embd(
            runtime.sam_states, prompt, nf_nobj=(1, 1), point_inputs=points)
        for name, logits in [('native', native), ('read_style', adapted)]:
            resized = F.interpolate(logits, size=shape, mode='bilinear')
            masks[name] |= (resized[0, 0] > .5).cpu().numpy()
        if points is not None:
            detail.update(coords_sam=points['point_coords'][0].cpu().tolist(),
                          labels=points['point_labels'][0].cpu().tolist())
        audit['points'].append(detail)
    return masks, audit
