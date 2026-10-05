"""Inject FTG into the unmodified public VIRST evaluator.

Environment variables:

``VIRST_FTG_CHECKPOINT``
    Optional adapter checkpoint from ``train_virst_ftg.py``.
``VIRST_FTG_VARIANT``
    ``public_virst``, ``identity_only``, ``state_only``, ``factorized_no_gate``,
    ``ftg``, ``unconditioned_ftg``, or ``scalar_ftg``.
``VIRST_FTG_DIAGNOSTICS``
    Optional JSONL destination for gate/state diagnostics.
``VIRST_FTG_OFFICIAL_EVAL_PATH``
    Evaluator to execute after patching; defaults to ``eval.py``.
"""

from __future__ import annotations

import json
import os
import runpy
import sys
import types
from pathlib import Path

import torch
from torch import nn

from projects.evoseg.ftg.virst_interface import find_virst_core, install_virst_ftg


class SegmentationOnlyLanguageHead(nn.Module):
    """Keep the unused logits field tiny during mask-only evaluation."""

    def forward(self, hidden_states):
        return hidden_states[..., :1]


def enable_segmentation_only_language_path(model: nn.Module) -> None:
    """Remove language-loss/logit memory without changing segmentation states.

    VIRST consumes only ``output.hidden_states`` after its inherited language
    forward in mask evaluation.  The public implementation nevertheless builds
    full-vocabulary FP32 logits and a language loss, which costs more than a
    GiB and can OOM on shared GPUs.  Preserve the transformer forward and its
    label-dependent visual-token compression exactly, but suppress the returned
    LM-loss labels and replace the now-unused head with a one-channel view.
    Generation is intentionally outside this evaluation-only launcher.
    """
    core = find_virst_core(model)
    original_decoder_forward = core.model.forward

    def segmentation_decoder_forward(self, *args, **kwargs):
        # Labels participate in VIRST's internal visual-token compression, so
        # they must reach the decoder.  Drop only the labels returned for the
        # unused language-loss branch after hidden states have been computed.
        outputs, _compressed_labels = original_decoder_forward(*args, **kwargs)
        return outputs, None

    core.model.forward = types.MethodType(
        segmentation_decoder_forward, core.model
    )
    core.lm_head = SegmentationOnlyLanguageHead()


def enable_cpu_video_storage(model: nn.Module) -> None:
    """Keep decoded full-video frames on CPU in SAM2's inference state.

    Long-RVOS clips can exceed 500 frames.  The released VIRST wrapper forces
    every decoded frame to remain resident on the GPU, although the public SAM2
    tracker has an output-preserving CPU-storage mode intended for this case.
    """
    core = find_virst_core(model)
    tracker = core.model.seg_model
    original_init_state = tracker.init_state

    def cpu_video_init_state(self, *args, **kwargs):
        kwargs["offload_video_to_cpu"] = True
        return original_init_state(*args, **kwargs)

    tracker.init_state = types.MethodType(cpu_video_init_state, tracker)


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    payload = (json.dumps(value, ensure_ascii=False) + "\n").encode()
    try:
        if os.write(descriptor, payload) != len(payload):
            raise OSError(f"short append to {path}")
    finally:
        os.close(descriptor)


def main() -> None:
    official_eval = os.environ.get("VIRST_FTG_OFFICIAL_EVAL_PATH", "eval.py")
    # When this launcher is executed by absolute path, Python otherwise places
    # ``projects/evoseg/ftg`` before the public checkout and resolves our local
    # ``model.py`` instead of VIRST's ``model`` package.
    public_virst_eval = os.environ.get("VIRST_OFFICIAL_EVAL_PATH", official_eval)
    official_root = str(Path(public_virst_eval).resolve().parent)
    if official_root in sys.path:
        sys.path.remove(official_root)
    sys.path.insert(0, official_root)

    import model.builder as builder

    variant = os.environ.get("VIRST_FTG_VARIANT", "ftg")
    adapter_path = os.environ.get("VIRST_FTG_CHECKPOINT")
    diagnostic_value = os.environ.get("VIRST_FTG_DIAGNOSTICS")
    original_load = builder.load_checkpoint_virst

    def ftg_load(model, checkpoint):
        loaded = original_load(model, checkpoint)
        segmentation_only = os.environ.get(
            "VIRST_SEGMENTATION_ONLY", "1"
        ).lower() not in {"0", "false", "no"}
        if segmentation_only:
            enable_segmentation_only_language_path(loaded)
        cpu_video_storage = os.environ.get(
            "VIRST_OFFLOAD_VIDEO_TO_CPU", "1"
        ).lower() not in {"0", "false", "no"}
        if cpu_video_storage:
            enable_cpu_video_storage(loaded)
        adapter = (
            torch.load(adapter_path, map_location="cpu", weights_only=False)
            if adapter_path
            else None
        )
        configured_variant = adapter.get("variant") if adapter else variant
        if adapter is not None and variant != configured_variant:
            raise ValueError(
                f"requested variant {variant} differs from adapter {configured_variant}"
            )
        wrapper = install_virst_ftg(
            loaded,
            variant=configured_variant,
            max_gate_delta=(adapter.get("max_gate_delta", 0.5) if adapter else 0.5),
        )
        if adapter is not None:
            wrapper.composer.load_state_dict(adapter["composer"], strict=True)

        if diagnostic_value:
            diagnostic_path = Path(diagnostic_value)

            def record(module, inputs, output):
                diagnostics = module.last_diagnostics
                if diagnostics is None:
                    raise RuntimeError("missing VIRST FTG diagnostics")
                public_prompt = diagnostics.identity + diagnostics.state
                correction = diagnostics.prompt - public_prompt
                append_jsonl(
                    diagnostic_path,
                    {
                        "variant": module.variant,
                        "identity_norm": diagnostics.identity.detach().float()
                        .norm(dim=-1).cpu().tolist(),
                        "state_norm": diagnostics.state.detach().float()
                        .norm(dim=-1).cpu().tolist(),
                        "gate_mean": float(diagnostics.gate.detach().float().mean()),
                        "gate_std": float(diagnostics.gate.detach().float().std()),
                        "correction_norm": correction.detach().float()
                        .norm(dim=-1).cpu().tolist(),
                        "identity_mean_error": float(
                            (
                                diagnostics.prompt.detach().float().mean(dim=2)
                                - diagnostics.identity.detach().float().squeeze(2)
                            ).abs().max()
                        ),
                    },
                )

            wrapper.register_forward_hook(record)
        print(
            json.dumps(
                {
                    "virst_ftg_installed": True,
                    "variant": configured_variant,
                    "adapter": adapter_path,
                    "segmentation_only_language_path": segmentation_only,
                    "cpu_video_storage": cpu_video_storage,
                    "trainable_parameters": sum(
                        parameter.numel() for parameter in wrapper.composer.parameters()
                    ),
                }
            ),
            flush=True,
        )
        return loaded

    builder.load_checkpoint_virst = ftg_load
    runpy.run_path(official_eval, run_name="__main__")


if __name__ == "__main__":
    main()
