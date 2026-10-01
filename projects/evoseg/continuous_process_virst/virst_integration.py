"""Install CPG immediately before the official VIRST SegPrompter decoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .process_module import ContinuousProcessConditioner, ContinuousProcessOutput


@dataclass
class ContinuousProcessDiagnostics:
    original: ContinuousProcessOutput
    permuted: dict[str, ContinuousProcessOutput]


class ContinuousProcessSegPrompter(nn.Module):
    def __init__(self, official: nn.Module, query_dim: int, max_states: int = 6) -> None:
        super().__init__()
        self.official = official
        self.token_dim = official.token_dim
        self.max_length = official.max_length
        self.grid_hw = official.grid_hw
        self.nhead = official.nhead
        self.conditioner = ContinuousProcessConditioner(
            query_dim=query_dim,
            vision_dim=256,
            process_dim=official.token_dim,
            max_states=max_states,
            heads=official.nhead,
        )
        self.fusion_norm = nn.LayerNorm(official.token_dim)
        self._query_states: Tensor | None = None
        self._query_padding_mask: Tensor | None = None
        self._permutations: tuple[str, ...] = ()
        self.last_diagnostics: ContinuousProcessDiagnostics | None = None

    def set_query_context(self, states: Tensor, padding_mask: Tensor | None = None) -> None:
        self._query_states = states
        self._query_padding_mask = padding_mask

    def set_permutations(self, values: tuple[str, ...]) -> None:
        unsupported = set(values) - {"reverse", "block_swap"}
        if unsupported:
            raise ValueError(f"unsupported temporal permutations: {sorted(unsupported)}")
        self._permutations = values

    @staticmethod
    def _permute_video(video: Tensor, kind: str) -> Tensor:
        if kind == "reverse":
            return video.flip(1)
        if kind == "block_swap":
            split = video.shape[1] // 2
            return torch.cat([video[:, split:], video[:, :split]], dim=1)
        raise ValueError(kind)

    def forward(
        self,
        seg_token: Tensor,
        image_token: Tensor,
        seg_mask: Tensor | None = None,
        return_attn: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if self._query_states is None:
            raise RuntimeError("query context was not captured before SegPrompter")
        num_conv, objects, _ = seg_token.shape
        n_img, frames, channels, height, width = image_token.shape
        if num_conv != n_img:
            raise ValueError("seg_token and image_token batch sizes differ")
        if frames > self.max_length:
            raise ValueError(f"frames={frames} exceeds official max_length={self.max_length}")

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
        self.last_diagnostics = ContinuousProcessDiagnostics(process, permuted)

        x = image_token.reshape(num_conv * frames, channels, height, width)
        x = self.official.spatial_down(x)
        x = x.reshape(num_conv, frames, self.token_dim, -1).transpose(2, 3)
        x = x.reshape(num_conv, frames * (self.grid_hw**2), self.token_dim)
        memory = self.official.memory_norm(x).transpose(0, 1).contiguous()

        seg = self.official.seg_proj(seg_token)
        seg = seg.unsqueeze(2).expand(num_conv, objects, frames, self.token_dim)
        residual = process.frame_states.unsqueeze(1).expand(-1, objects, -1, -1)
        seg = self.fusion_norm(seg + residual)
        seg = seg.reshape(num_conv, objects * frames, self.token_dim).transpose(0, 1)

        if seg_mask is None:
            target_padding = torch.zeros(
                (num_conv, objects * frames), dtype=torch.bool, device=seg_token.device
            )
        else:
            keep = seg_mask.to(torch.bool).unsqueeze(2).expand(num_conv, objects, frames)
            target_padding = ~keep.reshape(num_conv, objects * frames)
        attention_scores = []
        for layer in self.official.decoder:
            seg, attention = layer(
                tgt=seg,
                memory=memory,
                T_mem=frames,
                T_q=frames,
                tgt_mask=None,
                tgt_key_padding_mask=target_padding,
                memory_key_padding_mask=None,
                return_attn=True,
            )
            attention_scores.append(attention)
        decoded = seg.transpose(0, 1).reshape(num_conv, objects, frames, self.token_dim)
        if not return_attn:
            return decoded
        attention = attention_scores[-1]
        frame_score = attention.view(
            num_conv, self.nhead, objects * frames, frames, self.grid_hw**2
        ).mean(dim=(1, 2, 4))
        return decoded, frame_score


class QueryStateCapture:
    def __init__(
        self,
        outer_model: nn.Module,
        inner_model: nn.Module,
        prompter: ContinuousProcessSegPrompter,
        seg_token_idx: int,
    ) -> None:
        self.prompter = prompter
        self.seg_token_idx = seg_token_idx
        self.original_ids: Tensor | None = None
        self.original_attention: Tensor | None = None
        self._pre = outer_model.register_forward_pre_hook(
            self._capture_inputs, with_kwargs=True
        )
        self._post = inner_model.register_forward_hook(
            self._capture_hidden, with_kwargs=True
        )

    def close(self) -> None:
        self._pre.remove()
        self._post.remove()

    def _capture_inputs(
        self, module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
    ):
        # Inference context deliberately captures only model inputs. GT masks,
        # object IDs, and interval labels have no path into this hook.
        self.original_ids = kwargs.get("input_ids")
        self.original_attention = kwargs.get("attention_mask")
        return args, kwargs

    def _capture_hidden(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ):
        hidden = self._last_hidden_state(output)
        expanded_attention = kwargs.get("attention_mask")
        states = []
        for row in range(hidden.shape[0]):
            expanded_length = (
                int(expanded_attention[row].sum())
                if expanded_attention is not None
                else hidden.shape[1]
            )
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
            states.append(hidden[row, start:stop].detach())
        maximum = max(item.shape[0] for item in states)
        padded = hidden.new_zeros((len(states), maximum, hidden.shape[-1]))
        padding = torch.ones(
            (len(states), maximum), dtype=torch.bool, device=hidden.device
        )
        for row, value in enumerate(states):
            padded[row, : value.shape[0]] = value
            padding[row, : value.shape[0]] = False
        self.prompter.set_query_context(padded, padding)

    @staticmethod
    def _last_hidden_state(output: Any) -> Tensor:
        if isinstance(output, Tensor):
            return output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if isinstance(output, (tuple, list)) and output:
            return QueryStateCapture._last_hidden_state(output[0])
        raise TypeError(f"cannot extract hidden state from {type(output)!r}")


def install_continuous_process_virst(
    model: nn.Module, max_states: int = 6
) -> QueryStateCapture:
    if isinstance(model.model.seg_prompter, ContinuousProcessSegPrompter):
        raise RuntimeError("CPG-VIRST is already installed")
    wrapper = ContinuousProcessSegPrompter(
        model.model.seg_prompter,
        query_dim=model.config.hidden_size,
        max_states=max_states,
    )
    model.model.seg_prompter = wrapper
    return QueryStateCapture(model, model.model, wrapper, model.seg_token_idx)
