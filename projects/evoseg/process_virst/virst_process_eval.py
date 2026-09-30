"""Inject ProcessVIRST into the official evaluator without editing VIRST.

This is an integration/smoke runner. Training uses the same installation helper
but owns the optimizer and order loss explicitly.
"""

from __future__ import annotations

import json
import os
import runpy
from pathlib import Path

import torch

from projects.evoseg.process_virst.virst_integration import (
    ProcessAwareSegPrompter,
    install_process_virst,
)


def _append(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    payload = (json.dumps(value, ensure_ascii=False) + "\n").encode()
    try:
        if os.write(descriptor, payload) != len(payload):
            raise OSError(f"short append to {path}")
    finally:
        os.close(descriptor)


def _find_core(model: torch.nn.Module) -> torch.nn.Module:
    matches = []
    for module in model.modules():
        nested = getattr(module, "model", None)
        if (
            hasattr(module, "seg_token_idx")
            and nested is not None
            and hasattr(nested, "seg_prompter")
        ):
            matches.append(module)
    if len(matches) != 1:
        raise RuntimeError(f"expected one VIRST core, found {len(matches)}")
    return matches[0]


def main() -> None:
    diagnostic_path = Path(os.environ["PROCESS_VIRST_DIAGNOSTICS"])
    official_eval = os.environ.get(
        "PROCESS_VIRST_OFFICIAL_EVAL_PATH",
        os.environ.get("VIRST_OFFICIAL_EVAL_PATH", "eval.py"),
    )
    if os.environ.get("PROCESS_VIRST_GROUNDMORE_EXACT20") == "1":
        from projects.evoseg.process_virst.groundmore_virst_eval import (
            install_exact_20_frame_sampling,
        )

        install_exact_20_frame_sampling()

    import model.builder as builder

    original_load = builder.load_checkpoint_virst

    def process_load(model, checkpoint):
        loaded = original_load(model, checkpoint)
        core = _find_core(loaded)
        capture = install_process_virst(core)
        permutation_values = os.environ.get(
            "PROCESS_VIRST_PERMUTATIONS",
            os.environ.get("PROCESS_VIRST_PERMUTATION", ""),
        )
        permutations = tuple(value for value in permutation_values.split(",") if value)
        capture.set_permutations(permutations)
        prompter = core.model.seg_prompter
        assert isinstance(prompter, ProcessAwareSegPrompter)
        process_checkpoint = os.environ.get("PROCESS_VIRST_CHECKPOINT")
        if process_checkpoint:
            state = torch.load(process_checkpoint, map_location="cpu", weights_only=False)
            prompter.conditioner.load_state_dict(state["process_virst"], strict=True)
            prompter.fusion_norm.load_state_dict(state["fusion_norm"], strict=True)

        def record_diagnostics(module, inputs, output):
            diagnostics = module.last_diagnostics
            if diagnostics is None:
                raise RuntimeError("ProcessVIRST produced no diagnostics")
            original = diagnostics.original
            _append(
                diagnostic_path,
                {
                    "alignment_score": original.alignment_score.detach().float().cpu().tolist(),
                    "validity": original.validity.detach().float().cpu().tolist(),
                    "alignment": original.alignment.detach().float().cpu().tolist(),
                    "frame_state_norm": original.frame_states.detach().float().norm(dim=-1).cpu().tolist(),
                    "beta": float(original.beta.detach().float().cpu()),
                    "permutations": {
                        name: {
                            "alignment_score": value.alignment_score.detach().float().cpu().tolist(),
                            "alignment": value.alignment.detach().float().cpu().tolist(),
                            "frame_state_norm": value.frame_states.detach().float().norm(dim=-1).cpu().tolist(),
                            "frame_state_response_l2": (
                                original.frame_states.detach().float()
                                - value.frame_states.detach().float()
                            ).norm(dim=-1).cpu().tolist(),
                        }
                        for name, value in diagnostics.permuted.items()
                    },
                },
            )

        prompter.register_forward_hook(record_diagnostics)
        trainable = sum(parameter.numel() for parameter in prompter.parameters() if parameter.requires_grad)
        process_trainable = sum(parameter.numel() for parameter in prompter.conditioner.parameters())
        print(
            json.dumps(
                {
                    "process_virst_installed": True,
                    "process_module_params": process_trainable,
                    "wrapped_prompter_trainable_params_before_freeze_policy": trainable,
                    "permutations": permutations,
                    "process_checkpoint": process_checkpoint,
                }
            ),
            flush=True,
        )
        return loaded

    builder.load_checkpoint_virst = process_load
    runpy.run_path(official_eval, run_name="__main__")


if __name__ == "__main__":
    main()
