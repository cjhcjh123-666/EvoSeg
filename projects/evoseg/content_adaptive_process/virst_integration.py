"""Install CAPG immediately before the unmodified official VIRST SegPrompter."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from projects.evoseg.continuous_process_virst.virst_integration import QueryStateCapture

from .process_module import ContentAdaptiveProcessConditioner, ContentAdaptiveProcessOutput


@dataclass
class CAPGDiagnostics:
    original: ContentAdaptiveProcessOutput
    permuted: dict[str, ContentAdaptiveProcessOutput]


class ContentAdaptiveSegPrompter(nn.Module):
    def __init__(
        self,
        official: nn.Module,
        query_dim: int,
        max_states: int = 6,
        transition_mode: str = "adaptive",
    ) -> None:
        super().__init__()
        self.official = official
        for name in ("token_dim", "max_length", "grid_hw", "nhead"):
            setattr(self, name, getattr(official, name))
        self.conditioner = ContentAdaptiveProcessConditioner(
            query_dim=query_dim,
            vision_dim=256,
            process_dim=official.token_dim,
            max_states=max_states,
            transition_mode=transition_mode,
        )
        self.fusion_norm = nn.LayerNorm(official.token_dim)
        self._query_states: Tensor | None = None
        self._query_padding_mask: Tensor | None = None
        self._permutations: tuple[str, ...] = ()
        self.last_diagnostics: CAPGDiagnostics | None = None

    def set_query_context(self, states: Tensor, padding_mask: Tensor | None = None) -> None:
        self._query_states = states
        self._query_padding_mask = padding_mask

    def set_permutations(self, values: tuple[str, ...]) -> None:
        if set(values) - {"reverse", "block_swap"}:
            raise ValueError("only reverse and block_swap are supported")
        self._permutations = values

    @staticmethod
    def _permute_video(video: Tensor, kind: str) -> Tensor:
        if kind == "reverse":
            return video.flip(1)
        split = video.shape[1] // 2
        return torch.cat([video[:, split:], video[:, :split]], dim=1)

    def score_training_object_masks(self, object_masks: Tensor) -> Tensor:
        if self.last_diagnostics is None:
            raise RuntimeError("run model forward before training-only object scoring")
        return self.conditioner.score_training_object_masks(
            self.last_diagnostics.original, object_masks
        )

    def forward(
        self,
        seg_token: Tensor,
        image_token: Tensor,
        seg_mask: Tensor | None = None,
        return_attn: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if self._query_states is None:
            raise RuntimeError("query states were not captured")
        conversations, objects, _ = seg_token.shape
        image_batch, frames, channels, height, width = image_token.shape
        if conversations != image_batch:
            raise ValueError("segmentation and image batch sizes differ")
        process = self.conditioner(
            self._query_states, image_token, self._query_padding_mask
        )
        permuted = {
            name: self.conditioner(
                self._query_states,
                self._permute_video(image_token, name),
                self._query_padding_mask,
            )
            for name in self._permutations
        }
        self.last_diagnostics = CAPGDiagnostics(process, permuted)

        memory = self.official.spatial_down(
            image_token.reshape(conversations * frames, channels, height, width)
        )
        memory = memory.reshape(conversations, frames, self.token_dim, -1).transpose(2, 3)
        memory = memory.reshape(conversations, frames * self.grid_hw**2, self.token_dim)
        memory = self.official.memory_norm(memory).transpose(0, 1).contiguous()
        seg = self.official.seg_proj(seg_token).unsqueeze(2).expand(
            conversations, objects, frames, self.token_dim
        )
        seg = self.fusion_norm(seg + process.frame_states.unsqueeze(1))
        seg = seg.reshape(conversations, objects * frames, self.token_dim).transpose(0, 1)
        if seg_mask is None:
            padding = torch.zeros(
                conversations, objects * frames, dtype=torch.bool, device=seg.device
            )
        else:
            keep = seg_mask.to(torch.bool).unsqueeze(2).expand(conversations, objects, frames)
            padding = ~keep.reshape(conversations, objects * frames)
        attention_scores = []
        for layer in self.official.decoder:
            seg, attention = layer(
                tgt=seg,
                memory=memory,
                T_mem=frames,
                T_q=frames,
                tgt_mask=None,
                tgt_key_padding_mask=padding,
                memory_key_padding_mask=None,
                return_attn=True,
            )
            attention_scores.append(attention)
        decoded = seg.transpose(0, 1).reshape(conversations, objects, frames, self.token_dim)
        if not return_attn:
            return decoded
        attention = attention_scores[-1]
        frame_score = attention.view(
            conversations, self.nhead, objects * frames, frames, self.grid_hw**2
        ).mean(dim=(1, 2, 4))
        return decoded, frame_score


def install_content_adaptive_virst(
    model: nn.Module, max_states: int = 6, transition_mode: str = "adaptive"
) -> QueryStateCapture:
    wrapper = ContentAdaptiveSegPrompter(
        model.model.seg_prompter,
        query_dim=model.config.hidden_size,
        max_states=max_states,
        transition_mode=transition_mode,
    )
    model.model.seg_prompter = wrapper
    return QueryStateCapture(model, model.model, wrapper, model.seg_token_idx)
