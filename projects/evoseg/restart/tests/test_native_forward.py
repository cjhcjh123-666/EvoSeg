from types import SimpleNamespace

import torch
from torch import nn

from projects.evoseg.restart import native_model


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.hidden = nn.Linear(4, 4)
        self.head = nn.Linear(4, 8)

    def get_base_model(self):
        return self

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head

    def model(self, inputs_embeds, **kwargs):
        return SimpleNamespace(last_hidden_state=self.hidden(inputs_embeds))


class TinySAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.sam_mask_decoder = nn.Module()
        self.sam_mask_decoder.conv_s0 = nn.Conv2d(1, 1, 1)
        self.sam_mask_decoder.conv_s1 = nn.Conv2d(1, 1, 1)
        self.hidden_dim = 1
        self.no_mem_embed = nn.Parameter(torch.zeros(1, 1, 1), requires_grad=False)
        self.use_high_res_features_in_sam = True

    def image_encoder(self, pixels):
        return {'backbone_fpn': [torch.ones(1, 1, 2, 2), torch.ones(1, 1, 1, 1)]}

    def _prepare_backbone_features(self, features):
        return None, [v.flatten(2).permute(2, 0, 1) for v in features['backbone_fpn']], None, [(2, 2), (1, 1)]

    def _use_multimask(self, **kwargs):
        return False

    def _forward_sam_heads(self, backbone_features, high_res_features, language_embd, **kwargs):
        logits = backbone_features + high_res_features[0].mean((2, 3), keepdim=True)
        logits = logits + language_embd[:, :, 0][:, :, None, None]
        return None, None, None, logits, None, None, None


class TinyFoundation(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = TinyLM()
        self.vision_model = nn.Identity()
        self.mlp1 = nn.Identity()
        self.text_hidden_fcs = nn.Linear(4, 1)
        self.grounding_encoder = nn.Module()
        self.grounding_encoder.sam2_model = TinySAM()
        self.grounding_encoder.preprocess_image = lambda pixels, **kwargs: pixels.float()
        self.img_context_token_id, self.seg_token_idx = 2, 3

    def extract_feature(self, pixels):
        return torch.ones(len(pixels), 256, 4)


def test_native_forward_counts_losses_and_decoder_high_res_gradients(monkeypatch):
    # Only tiny dummy tensors exercise plumbing, never real training labels.
    original_loss = native_model.native_mask_losses
    monkeypatch.setattr(native_model, 'native_mask_losses',
                        lambda logits, truth: original_loss(logits, truth, num_points=16))
    foundation = TinyFoundation()
    network = native_model.NativeFineTune(foundation).train()
    ids = torch.tensor([1] + [2] * 5120 + [3] * 50)
    labels = ids.clone()
    labels[:5121] = -100
    sample = {'input_ids': ids, 'labels': labels, 'pixel_values': torch.zeros(20, 3, 2, 2),
              'grounding_pixels': torch.zeros(10, 3, 2, 2), 'targets': torch.ones(10, 5, 4, 4)}
    loss, components = network(sample)
    assert set(components) == {'llm', 'mask_ce', 'mask_dice'}
    assert torch.isfinite(loss)
    loss.backward()
    assert foundation.text_hidden_fcs.weight.grad.abs().sum() > 0
    assert foundation.language_model.hidden.weight.grad.abs().sum() > 0
    assert foundation.grounding_encoder.sam2_model.sam_mask_decoder.conv_s0.weight.grad.abs().sum() > 0
