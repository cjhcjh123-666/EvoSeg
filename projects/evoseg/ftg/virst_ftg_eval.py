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
from pathlib import Path

import torch

from projects.evoseg.ftg.virst_interface import install_virst_ftg


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
