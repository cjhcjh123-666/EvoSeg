"""Non-invasive ProcessVIRST integration for the official VIRST model.

The official checkout stays unchanged. This module wraps its initialized
``SegPrompter`` and captures frozen language hidden states immediately before
the official forward reaches that prompter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .process_module import ProcessConditioner, ProcessConditionerOutput


@dataclass
class ProcessDiagnostics:
    original: ProcessConditionerOutput
    permuted_score: Tensor | None
    permutation: str | None


class ProcessAwareSegPrompter(nn.Module):
    """Reuse official SegPrompter weights, adding process states pre-decoder."""

    def __init__(self, official: nn.Module, query_dim: int, slots: int = 4) -> None:
        super().__init__()
        self.official = official
        self.token_dim = official.token_dim
        self.max_length = official.max_length
        self.grid_hw = official.grid_hw
        self.nhead = official.nhead
        self.conditioner = ProcessConditioner(
            query_dim=query_dim,
            vision_dim=256,
            process_dim=official.token_dim,
            slots=slots,
            heads=official.nhead,
        )
        self.fusion_norm = nn.LayerNorm(official.token_dim)
        self._query_states: Tensor | None = None
        self._query_padding_mask: Tensor | None = None
        self._permutation: str | None = None
        self.last_diagnostics: ProcessDiagnostics | None = None

    def set_query_context(
        self,
        states: Tensor,
        padding_mask: Tensor | None = None,
        permutation: str | None = None,
    ) -> None:
        self._query_states = states
        self._query_padding_mask = padding_mask
        self._permutation = permutation

    @staticmethod
    def _permute_video(video: Tensor, kind: str) -> Tensor:
        frames = video.shape[1]
        if kind == "reverse":
            return video.flip(1)
        if kind == "block_swap":
            split = frames // 2
            return torch.cat([video[:, split:], video[:, :split]], dim=1)
        raise ValueError(f"unsupported temporal permutation: {kind}")

    def forward(
        self,
        seg_token: Tensor,
        image_token: Tensor,
        seg_mask: Tensor | None = None,
        return_attn: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if self._query_states is None:
            raise RuntimeError("query context was not captured before SegPrompter execution")
        num_conv, objects, _ = seg_token.shape
        n_img, frames, channels, height, width = image_token.shape
        if num_conv != n_img:
            raise ValueError("seg_token and image_token batch sizes differ")
        if frames > self.max_length:
            raise ValueError(f"frames={frames} exceeds official max_length={self.max_length}")

        process = self.conditioner(
            self._query_states,
            image_token,
            self._query_padding_mask,
        )
        permuted_score = None
        if self._permutation is not None:
            permuted = self.conditioner(
                self._query_states,
                self._permute_video(image_token, self._permutation),
                self._query_padding_mask,
            )
            permuted_score = permuted.alignment_score
        self.last_diagnostics = ProcessDiagnostics(process, permuted_score, self._permutation)

        x = image_token.reshape(num_conv * frames, channels, height, width)
        x = self.official.spatial_down(x)
        x = x.reshape(num_conv, frames, self.token_dim, -1).transpose(2, 3)
        x = x.reshape(num_conv, frames * (self.grid_hw**2), self.token_dim)
        memory = self.official.memory_norm(x).transpose(0, 1).contiguous()

        seg = self.official.seg_proj(seg_token)
        seg = seg.unsqueeze(2).expand(num_conv, objects, frames, self.token_dim)
        process_residual = process.frame_states.unsqueeze(1).expand(-1, objects, -1, -1)
        seg = self.fusion_norm(seg + process_residual)
        seg = seg.reshape(num_conv, objects * frames, self.token_dim).transpose(0, 1)

        attn_scores = []
        if seg_mask is None:
            target_padding = torch.zeros(
                (num_conv, objects * frames), dtype=torch.bool, device=seg_token.device
            )
        else:
            keep = seg_mask.to(torch.bool).unsqueeze(2).expand(num_conv, objects, frames)
            target_padding = ~keep.reshape(num_conv, objects * frames)

        for layer in self.official.decoder:
            seg, attn = layer(
                tgt=seg,
                memory=memory,
                T_mem=frames,
                T_q=frames,
                tgt_mask=None,
                tgt_key_padding_mask=target_padding,
                memory_key_padding_mask=None,
                return_attn=True,
            )
            attn_scores.append(attn)
        decoded = seg.transpose(0, 1).reshape(num_conv, objects, frames, self.token_dim)
        if not return_attn:
            return decoded
        attention = attn_scores[-1]
        frame_score = attention.view(
            num_conv, self.nhead, objects * frames, frames, self.grid_hw**2
        ).mean(dim=(1, 2, 4))
        return decoded, frame_score


class QueryStateCapture:
    """Capture query-token hidden states without editing official VIRST source."""

    def __init__(self, outer_model: nn.Module, inner_model: nn.Module, prompter: ProcessAwareSegPrompter, seg_token_idx: int):
        self.outer_model = outer_model
        self.inner_model = inner_model
        self.prompter = prompter
        self.seg_token_idx = seg_token_idx
        self.original_ids: Tensor | None = None
        self.original_attention: Tensor | None = None
        self.permutation: str | None = None
        self._pre = outer_model.register_forward_pre_hook(self._capture_inputs, with_kwargs=True)
        self._post = inner_model.register_forward_hook(self._capture_hidden, with_kwargs=True)

    def close(self) -> None:
        self._pre.remove()
        self._post.remove()

    def set_permutation(self, permutation: str | None) -> None:
        self.permutation = permutation

    def _capture_inputs(self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]):
        self.original_ids = kwargs.get("input_ids")
        self.original_attention = kwargs.get("attention_mask")
        return args, kwargs

    def _capture_hidden(self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any], output: Any):
        hidden = self._last_hidden_state(output)
        expanded_attention = kwargs.get("attention_mask")
        states = []
        for row in range(hidden.shape[0]):
            expanded_length = int(expanded_attention[row].sum()) if expanded_attention is not None else hidden.shape[1]
            if self.original_ids is None:
                start, stop = max(0, expanded_length - 128), expanded_length
            else:
                ids = self.original_ids[row]
                original_length = (
                    int(self.original_attention[row].sum())
                    if self.original_attention is not None
                    else ids.shape[0]
                )
                ids = ids[:original_length]
                image_positions = (ids < 0).nonzero(as_tuple=True)[0]
                seg_positions = (ids == self.seg_token_idx).nonzero(as_tuple=True)[0]
                original_start = int(image_positions[-1]) + 1 if image_positions.numel() else 0
                original_stop = int(seg_positions[0]) if seg_positions.numel() else original_length
                offset = expanded_length - original_length
                start = max(0, original_start + offset)
                stop = max(start + 1, min(expanded_length, original_stop + offset))
            # The VLM is frozen in ProcessVIRST SFT. Detaching here prevents the
            # order/segmentation losses from retaining its large activation graph.
            states.append(hidden[row, start:stop].detach())
        max_length = max(item.shape[0] for item in states)
        padded = hidden.new_zeros((len(states), max_length, hidden.shape[-1]))
        padding = torch.ones((len(states), max_length), dtype=torch.bool, device=hidden.device)
        for row, value in enumerate(states):
            padded[row, : value.shape[0]] = value
            padding[row, : value.shape[0]] = False
        self.prompter.set_query_context(padded, padding, self.permutation)

    @staticmethod
    def _last_hidden_state(output: Any) -> Tensor:
        """Handle Hugging Face and VideoChat's nested ``(output, labels)`` form."""
        if isinstance(output, Tensor):
            return output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if isinstance(output, (tuple, list)) and output:
            return QueryStateCapture._last_hidden_state(output[0])
        raise TypeError(f"cannot extract last_hidden_state from {type(output)!r}")


def install_process_virst(model: nn.Module, slots: int = 4) -> QueryStateCapture:
    """Install ProcessVIRST on an initialized official model and return its hook."""
    if isinstance(model.model.seg_prompter, ProcessAwareSegPrompter):
        raise RuntimeError("ProcessVIRST is already installed")
    official = model.model.seg_prompter
    wrapper = ProcessAwareSegPrompter(official, query_dim=model.config.hidden_size, slots=slots)
    model.model.seg_prompter = wrapper
    return QueryStateCapture(model, model.model, wrapper, model.seg_token_idx)
