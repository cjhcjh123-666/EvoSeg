"""Anti-shortcut evaluation protocol for referring segmentation.

Two probes that expose whether the model truly grounds masks in language
instead of relying on visual shortcuts:

1. no-object: a query describes an object that is absent from the image
   (category-aware negatives).  A well-grounded model must output low
   objectness and an empty mask; a shortcut model hallucinates a mask.

2. query-swap: two different queries for the same image (same media, different
   targets).  A well-grounded model must produce clearly different masks; a
   shortcut model returns nearly the same mask for both.

The script reports objectness statistics, hallucination rate, mean swap-mask
IoU, and each query's IoU with its own ground truth.  All outputs are written
outside the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_ROOT_FOR_IMPORT))

import numpy as np
import torch
import yaml
from tqdm import tqdm

from llava.evo_seg.capability import SegmentationCapability
from llava.evo_seg.contracts import GroundingBatch
from llava.evo_seg.decoder import QueryConditionedSpatialDecoder
from llava.evo_seg.sam2_lisa_head import Sam2LisaHead
from llava.evo_seg.sam2_adapter import SAM2ImageFeatureProvider
from llava.evo_seg.sam2_video_adapter import build_sam2_video_image_predictor
from llava.evo_seg.training import (
    GroundingProjector,
    SegEmbeddingInjector,
    apply_lora,
    seg_training_active,
)
from llava.evo_seg.vila_adapter import VILASegmentationAdapter
from scripts.evo_seg.train_s4b import _image_sample, _mask_iou, _upsample_dense


def _load_config(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _forward_image(
    record: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    template: str,
    seg_id: int,
    device: torch.device,
    adapter: VILASegmentationAdapter,
    capability: SegmentationCapability,
    projector: Any,
    decoder: Any,
    provider: SAM2ImageFeatureProvider,
    hidden_layer: int,
    spatial_scale: int,
    head: str,
) -> Any:
    sample = _image_sample(record, model, tokenizer, template, seg_id, device)
    with torch.no_grad(), seg_training_active(True), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        query_states = adapter.extract_query_states_training(
            sample["input_ids"], sample["media"], sample["media_config"],
            sample["query_mask"], sample["attention_mask"], hidden_layer=hidden_layer,
        )
        seg_positions = (
            sample["query_mask"][:, : sample["seg_position"] + 1].sum(dim=1) - 1
        ).to(dtype=torch.long, device=device)
        if head == "lisa":
            seg_state = query_states.states.to(dtype=torch.float32)
            seg_state = torch.nan_to_num(seg_state, nan=0.0, posinf=0.0, neginf=0.0)
            seg_state = seg_state[torch.arange(seg_state.shape[0], device=device), seg_positions]
            dense = provider.encode_frames(sample["rgb"])
            with torch.autocast(device_type="cuda", enabled=False):
                result = decoder(seg_state, dense.features, dense.frame_mask, dense.high_res_features)
        else:
            projected = projector(query_states.states, query_states.mask, seg_positions)
            dense = _upsample_dense(provider.encode_frames(sample["rgb"]), spatial_scale)
            result = capability(
                GroundingBatch(
                    query_states=projected,
                    query_mask=torch.ones(1, projected.shape[1], dtype=torch.bool, device=device),
                    dense_features=dense.features,
                    frame_mask=dense.frame_mask,
                    sample_ids=(record["sample_id"],),
                ),
                {"enabled": True, "task": "image"},
            )
    return sample, result


def _eval_no_object(
    records: List[Dict[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    template: str,
    seg_id: int,
    device: torch.device,
    adapter: Any,
    capability: Any,
    projector: Any,
    decoder: Any,
    provider: Any,
    hidden_layer: int,
    spatial_scale: int,
    head: str,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for record in tqdm(records, desc="no-object"):
        _, result = _forward_image(
            record, model, tokenizer, template, seg_id, device,
            adapter, capability, projector, decoder, provider,
            hidden_layer, spatial_scale, head,
        )
        objectness = float(torch.sigmoid(result.object_logits[0, 0]).item())
        mask = result.mask_logits[0, 0, 0] > 0
        area_ratio = float(mask.float().mean().item())
        results.append(
            {
                "sample_id": record["sample_id"],
                "query": record["query"],
                "objectness": objectness,
                "mask_area_ratio": area_ratio,
                "hallucinates": bool(area_ratio > 0.0),
            }
        )
    return results


def _eval_swap(
    records: List[Dict[str, Any]],
    pairs: List[Dict[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    template: str,
    seg_id: int,
    device: torch.device,
    adapter: Any,
    capability: Any,
    projector: Any,
    decoder: Any,
    provider: Any,
    hidden_layer: int,
    spatial_scale: int,
    head: str,
) -> List[Dict[str, Any]]:
    by_id = {record["sample_id"]: record for record in records if record["control_kind"] == "positive"}
    results: List[Dict[str, Any]] = []
    for pair in tqdm(pairs, desc="query-swap"):
        left = by_id.get(pair["left_sample_id"])
        right = by_id.get(pair["right_sample_id"])
        if left is None or right is None:
            continue
        sample_left, result_left = _forward_image(
            left, model, tokenizer, template, seg_id, device,
            adapter, capability, projector, decoder, provider,
            hidden_layer, spatial_scale, head,
        )
        sample_right, result_right = _forward_image(
            right, model, tokenizer, template, seg_id, device,
            adapter, capability, projector, decoder, provider,
            hidden_layer, spatial_scale, head,
        )
        pred_left = result_left.mask_logits[0, 0, 0].detach()
        pred_right = result_right.mask_logits[0, 0, 0].detach()
        target_left = sample_left["target_mask"][0, 0, 0].detach()
        target_right = sample_right["target_mask"][0, 0, 0].detach()
        swap_iou = _mask_iou(pred_left, target_right) + _mask_iou(pred_right, target_left)
        # Mask overlap between the two queries on the same image.
        pred_left_bool = pred_left > 0
        pred_right_bool = pred_right > 0
        inter = (pred_left_bool & pred_right_bool).sum().float().item()
        union = (pred_left_bool | pred_right_bool).sum().float().item()
        overlap = inter / union if union > 0 else 0.0
        results.append(
            {
                "pair": (pair["left_sample_id"], pair["right_sample_id"]),
                "query_left": left["query"],
                "query_right": right["query"],
                "iou_left_gt": _mask_iou(pred_left, target_left),
                "iou_right_gt": _mask_iou(pred_right, target_right),
                "swap_iou_gt": swap_iou / 2.0,
                "mask_overlap": overlap,
            }
        )
    return results


def _merge(args: Any) -> None:
    no_object = []
    for shard in sorted(args.output.glob("no_object.*.json")):
        no_object.extend(json.loads(shard.read_text(encoding="utf-8")))
    swap = []
    for shard in sorted(args.output.glob("swap.*.json")):
        swap.extend(json.loads(shard.read_text(encoding="utf-8")))

    def mean(key: str, rows: List[Dict[str, Any]]) -> Optional[float]:
        values = [row[key] for row in rows if key in row and row[key] is not None]
        return float(np.mean(values)) if values else None

    summary = {
        "no_object": {
            "samples": len(no_object),
            "mean_objectness": mean("objectness", no_object),
            "mean_mask_area_ratio": mean("mask_area_ratio", no_object),
            "hallucination_rate": mean("hallucinates", no_object),
        },
        "swap": {
            "pairs": len(swap),
            "mean_iou_left_gt": mean("iou_left_gt", swap),
            "mean_iou_right_gt": mean("iou_right_gt", swap),
            "mean_swap_iou_gt": mean("swap_iou_gt", swap),
            "mean_mask_overlap": mean("mask_overlap", swap),
        },
    }
    (args.output / "no_object.json").write_text(json.dumps(no_object, indent=2), encoding="utf-8")
    (args.output / "swap.json").write_text(json.dumps(swap, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vila-model", type=Path, required=True)
    parser.add_argument("--sam2-source-root", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--no-object-manifest", type=Path)
    parser.add_argument("--val-manifest-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        _merge(args)
        return 0

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("anti-shortcut evaluation requires CUDA")
    torch.cuda.set_device(device)
    config = _load_config(args.config.expanduser().resolve())
    training = dict(config.get("training", {}))
    decoder_config = dict(config.get("decoder", {}))
    sam2_config = dict(config.get("sam2", {}))
    prompt_config = dict(config.get("prompt", {}))
    template = str(
        prompt_config.get(
            "instruction_template",
            "Segment the object described by this referring expression:\n{query}",
        )
    )
    hidden_layer = int(training.get("hidden_layer", -1))
    spatial_scale = int(decoder_config.get("spatial_scale", 1))

    import llava

    model = llava.load(str(args.vila_model), device=str(device), device_map={"": str(device)})
    model.eval()
    model.requires_grad_(False)
    # Keep VILA in its native bf16 precision (QuantLinearTE casts to bf16);
    # the fp16 residual stream can overflow to Inf and corrupt predictions.
    model.to(torch.bfloat16)
    tokenizer = model.tokenizer
    seg_id = tokenizer.convert_tokens_to_ids("[SEG]")
    if seg_id == tokenizer.unk_token_id:
        tokenizer.add_special_tokens({"additional_special_tokens": ["[SEG]"]})
        seg_id = tokenizer.convert_tokens_to_ids("[SEG]")
        model.get_llm().resize_token_embeddings(len(tokenizer))
    embed_module = getattr(getattr(model.get_llm(), "model", model.get_llm()), "embed_tokens")
    seg_injector = SegEmbeddingInjector(embed_module, seg_id, int(model.llm.config.hidden_size))
    seg_injector.seg_embedding = torch.nn.Parameter(seg_injector.seg_embedding.detach().to(device))

    provider = SAM2ImageFeatureProvider(
        {
            "source_root": str(args.sam2_source_root),
            "model_config": sam2_config["model_config"],
            "checkpoint_path": str(args.sam2_checkpoint),
            "device": str(device),
            "apply_postprocessing": bool(sam2_config.get("apply_postprocessing", True)),
        },
        predictor_factory=build_sam2_video_image_predictor,
    )
    provider.initialize()

    head = str(config.get("head", "query_decoder"))
    if head == "lisa":
        decoder = Sam2LisaHead(
            hidden_size=int(model.llm.config.hidden_size),
            sam2_model=provider._predictor.model,
            prompt_dim=int(decoder_config.get("feature_dim", 256)),
        ).to(device=device)
        projector = torch.nn.Identity()
    else:
        decoder = QueryConditionedSpatialDecoder(
            query_dim=int(model.llm.config.hidden_size),
            feature_dim=int(decoder_config.get("feature_dim", 256)),
            model_dim=int(decoder_config.get("model_dim", 128)),
            num_heads=int(decoder_config.get("num_heads", 8)),
            num_layers=int(decoder_config.get("num_layers", 2)),
            num_object_queries=int(decoder_config.get("num_object_queries", 1)),
            dropout=float(decoder_config.get("dropout", 0.0)),
        ).to(device=device)
        projector = GroundingProjector(int(model.llm.config.hidden_size)).to(device=device)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    decoder.load_state_dict(checkpoint["decoder"])
    if head != "lisa":
        projector.load_state_dict(checkpoint["projector"])
    seg_injector.seg_embedding.data.copy_(checkpoint["seg_embedding"].to(device))
    lora = dict(config.get("lora", {}))
    llm = model.get_llm()
    llm_layers = getattr(getattr(llm, "model", llm), "layers")
    lora_layers = list(llm_layers[-int(lora.get("layers", 8)) :])
    lora_adapters, lora_handles = apply_lora(
        lora_layers,
        list(lora.get("projections", ["self_attn.q_proj", "self_attn.v_proj"])),
        int(lora.get("rank", 8)),
        float(lora.get("alpha", 16)),
    )
    for name, adapters in lora_adapters.items():
        for index, adapter in enumerate(adapters):
            adapter.load_state_dict(checkpoint["lora"][name][index])

    capability = SegmentationCapability(decoder)
    adapter = VILASegmentationAdapter(model, provider, capability)

    kwargs = dict(
        model=model,
        tokenizer=tokenizer,
        template=template,
        seg_id=seg_id,
        device=device,
        adapter=adapter,
        capability=capability,
        projector=projector,
        decoder=decoder,
        provider=provider,
        hidden_layer=hidden_layer,
        spatial_scale=spatial_scale,
        head=head,
    )

    if args.no_object_manifest:
        records = _load_jsonl(args.no_object_manifest.expanduser().resolve())
        if args.max_samples:
            records = records[: args.max_samples]
        records = records[args.shard_index :: args.shard_count]
        results = _eval_no_object(records, **kwargs)
        (args.output / f"no_object.{args.shard_index}.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
    if args.val_manifest_dir:
        val_dir = args.val_manifest_dir.expanduser().resolve()
        records = _load_jsonl(val_dir / "val.jsonl")
        pairs = _load_jsonl(val_dir / "val.pairs.jsonl")
        if args.max_samples:
            pairs = pairs[: args.max_samples]
        pairs = pairs[args.shard_index :: args.shard_count]
        results = _eval_swap(records, pairs, **kwargs)
        (args.output / f"swap.{args.shard_index}.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
