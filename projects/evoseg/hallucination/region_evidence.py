"""Experimental mask-conditioned evidence; no novelty or causality claim."""
import torch
from torch import nn
import torch.nn.functional as F


def pool_regions(image_features, logits, grid):
    """Native mask > .5, mapped onto the verified raster-order visual grid."""
    height, width = grid
    if image_features.ndim != 2 or len(image_features) != height * width:
        raise ValueError('visual token count and merged image grid disagree')
    weights = F.interpolate((logits.float() > .5).float(), size=grid, mode='area').reshape(-1)
    features = image_features.float()
    global_feature = features.mean(0, keepdim=True)
    foreground = (features * weights[:, None]).sum(0, keepdim=True) / weights.sum().clamp_min(1e-6)
    background = (features * (1 - weights[:, None])).sum(0, keepdim=True) / (1 - weights).sum().clamp_min(1e-6)
    if weights.sum() == 0:
        foreground = global_feature
    if (1 - weights).sum() == 0:
        background = global_feature
    return foreground, background, global_feature, weights.mean().reshape(1, 1)


class RegionEvidenceAdapter(nn.Module):
    """Preserve full reasoning features while conditioning a bounded residual on a mask.

    A global control has identical parameters but receives the image mean in
    both region slots. Its region pooling is still computed, matching budget.
    """
    def __init__(self, language_dim, prompt_dim=256, projection_dim=64):
        super().__init__()
        self.normalization = nn.LayerNorm(language_dim)
        self.projection = nn.Linear(language_dim, projection_dim)
        self.prompt_projection = nn.Sequential(nn.LayerNorm(prompt_dim), nn.Linear(prompt_dim, projection_dim))
        self.fusion = nn.Sequential(nn.Linear(projection_dim * 5 + 1, 128), nn.GELU())
        self.residual = nn.Linear(128, prompt_dim)
        self.support = nn.Linear(128, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        nn.init.zeros_(self.support.weight)
        nn.init.constant_(self.support.bias, 2.)

    def forward(self, prompt, semantic, interaction, foreground, background, global_feature, area,
                mode='regional'):
        if mode not in ('regional', 'global'):
            raise ValueError('unknown region-evidence control')
        if mode == 'global':
            foreground = background = global_feature
            area = torch.ones_like(area)  # do not leak mask extent into the global control
        values = [F.gelu(self.projection(self.normalization(v.float())))
                  for v in (semantic, interaction, foreground, background)]
        fused = self.fusion(torch.cat([*values, self.prompt_projection(prompt.float()), area.float()], -1))
        # Zero-initialized; no arbitrary prompt rescaling at initialization.
        scale = prompt.float().square().mean(-1, keepdim=True).sqrt().clamp_min(1e-3)
        correction = .1 * scale * self.residual(fused).tanh()
        return {'grounding_prompt': prompt + correction,
                'support_logit': self.support(fused).squeeze(-1)}


@torch.inference_mode()
def predict_regions(runtime, image_path, query, head=None, mode='regional'):
    """Image/query-only inference. Candidate masks/features never use GT."""
    from PIL import Image
    import numpy as np
    from .train_fresh import prefix_inputs
    from .eval_fresh import parse_native_codes
    from .interaction import factorial_interaction
    runtime.set_image(image_path)
    inputs = prefix_inputs(runtime.processor, runtime.image, query).to(runtime.model.device)
    generated = runtime.model.generate(**inputs, max_new_tokens=128, do_sample=False,
        return_dict_in_generate=True, output_hidden_states=True)
    answer = runtime.processor.decode(generated.sequences[0, inputs.input_ids.shape[1]:], skip_special_tokens=False)
    codes, malformed = parse_native_codes(answer)
    mask = np.zeros((runtime.image.height, runtime.image.width), dtype=bool)
    information = {'answer': answer, 'native_codes': codes, 'malformed_codes': malformed,
        'head_used': True, 'empty_probability': None, 'region_support_probabilities': [],
        'language_frozen_in_region_training': True}
    if not codes:
        return mask, information, []
    hidden = generated.hidden_states[0][-1]
    semantic = hidden[:, -1].float()
    tokens = hidden[0, inputs.input_ids[0] == runtime.model.config.image_token_id].float()
    temporal, grid_h, grid_w = map(int, inputs.image_grid_thw[0].tolist())
    merge = runtime.model.config.vision_config.spatial_merge_size
    if temporal != 1 or grid_h % merge or grid_w % merge:
        raise RuntimeError('regional pilot requires a single uncropped image and exact merged grid')
    grid = (grid_h // merge, grid_w // merge)
    if len(tokens) != grid[0] * grid[1]:
        raise RuntimeError('region pooling image-token geometry mismatch')
    del generated, hidden
    neutral = Image.new('RGB', runtime.image.size, (127, 127, 127))
    key = str(image_path)
    if key not in runtime.controls:
        runtime.controls = {key: (runtime.prefix_feature(runtime.image, 'an object'),
                                  runtime.prefix_feature(neutral, 'an object'))}
    visual, neither = runtime.controls[key]
    query_only = runtime.prefix_feature(neutral, query)
    if runtime.variant == 'full_view':
        visual, query_only, neither = visual * 0, query_only * 0, neither * 0
    interaction = factorial_interaction(semantic, visual, query_only, neither)
    proposals = []
    for pair in codes:
        embedding = runtime.tokenizer.quantizer.embed_code(torch.tensor([pair], device='cuda'))[:, None]
        base = runtime.tokenizer.deconcate_quant_embed(embedding).reshape(1, -1)
        parent = runtime.interface(base, semantic, visual, query_only, neither)
        prompt = parent['grounding_prompt']
        logits = runtime.tokenizer.model.inject_language_embd(runtime.sam_states, prompt[:, None], nf_nobj=(1, 1))
        foreground, background, global_feature, area = pool_regions(tokens, logits, grid)
        proposal = {'prompt': prompt, 'semantic': semantic, 'interaction': interaction,
                    'foreground': foreground, 'background': background, 'global_feature': global_feature,
                    'area': area, 'parent_logits': logits.float(), 'parent_empty_logit': parent['empty_logit'].float()}
        proposals.append(proposal)
        parent_empty = float(parent['empty_logit'].sigmoid())
        information['empty_probability'] = parent_empty
        if parent_empty >= runtime.empty_threshold:
            continue
        if head is not None:
            result = head(**{k: proposal[k] for k in ('prompt', 'semantic', 'interaction', 'foreground',
                                                    'background', 'global_feature', 'area')}, mode=mode)
            support = float(result['support_logit'].sigmoid())
            information['region_support_probabilities'].append(support)
            if support < .5:
                continue
            logits = runtime.tokenizer.model.inject_language_embd(runtime.sam_states,
                result['grounding_prompt'][:, None], nf_nobj=(1, 1))
        resized = F.interpolate(logits, size=mask.shape, mode='bilinear')
        mask |= (resized[0, 0] > .5).cpu().numpy()
    return mask, information, proposals
