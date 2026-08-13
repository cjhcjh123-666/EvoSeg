"""Formal S4b evaluation: RefCOCO mask IoU and Ref-YT-VOS J&F.

Loads a training checkpoint produced by ``train_s4b.py`` and evaluates on the
validation manifests.  Video metrics propagate the predicted anchor mask with a
frozen shared SAM2 video model (same predicted-anchor path as S3) and compute
region similarity (J) and boundary F-score (F) against ground-truth masks.
All outputs are written outside the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_ROOT_FOR_IMPORT))

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from tqdm import tqdm

from llava.evo_seg.capability import SegmentationCapability
from llava.evo_seg.contracts import GroundingBatch
from llava.evo_seg.decoder import QueryConditionedSpatialDecoder
from llava.evo_seg.sam2_adapter import RGBFrameBatch, SAM2ImageFeatureProvider
from llava.evo_seg.sam2_lisa_head import Sam2LisaHead
from llava.evo_seg.sam2_video_adapter import (
    SAM2VideoMaskPropagator,
    build_sam2_video_image_predictor,
)
from llava.evo_seg.training import (
    GroundingProjector,
    SegEmbeddingInjector,
    apply_lora,
    seg_training_active,
)
from llava.evo_seg.vila_adapter import VILASegmentationAdapter
from scripts.evo_seg.train_s4b import (
    _align_targets_for_refine,
    _image_sample,
    _mask_iou,
    _query_span_mask,
    _rgb_batch,
    _sam2_refine_trainable,
    _target_tensor,
    _upsample_dense,
    _video_sample,
)


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config


def _load_records(manifest_dir: Path) -> List[Dict[str, Any]]:
    records = []
    path = manifest_dir / "val.jsonl"
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _sobel_boundary(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(np.float32)
    gradient_x = np.abs(mask - np.roll(mask, 1, axis=1))
    gradient_y = np.abs(mask - np.roll(mask, 1, axis=0))
    return ((gradient_x + gradient_y) > 0).astype(bool)


def _boundary_f_score(prediction: np.ndarray, ground_truth: np.ndarray, eps: float = 1e-6) -> float:
    prediction_boundary = _sobel_boundary(prediction)
    ground_truth_boundary = _sobel_boundary(ground_truth)
    overlap = float((prediction_boundary & ground_truth_boundary).sum())
    precision = overlap / float(prediction_boundary.sum() + eps)
    recall = overlap / float(ground_truth_boundary.sum() + eps)
    return 2.0 * precision * recall / (precision + recall + eps)


def _dilation_disk(radius: float) -> np.ndarray:
    """Boolean disk structuring element used by the DAVIS boundary F metric."""
    radius = max(int(round(radius)), 1)
    size = 2 * radius + 1
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    return (x * x + y * y) <= radius * radius


def _boundary_f_score_official(
    prediction: np.ndarray, ground_truth: np.ndarray, bound_th: float = 0.008, eps: float = 1e-6
) -> float:
    """DAVIS / Ref-YT-VOS boundary F: precision/recall over dilated boundaries.

    Matches the official protocol used by DAVIS 2017 and Ref-YT-VOS:
    ``bound_th=0.008`` gives a dilation radius of ``ceil(0.008 * sqrt(H*W))``
    and the boundary is the 1-pixel morphological boundary of the mask.
    """
    from scipy import ndimage

    def _boundary(mask: np.ndarray) -> np.ndarray:
        dilated = ndimage.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool))
        return dilated & ~mask

    pred_boundary = _boundary(prediction)
    gt_boundary = _boundary(ground_truth)
    radius = bound_th * np.linalg.norm(np.asarray(prediction.shape, dtype=np.float64))
    disk = _dilation_disk(radius)
    pred_dilated = ndimage.binary_dilation(pred_boundary, structure=disk)
    gt_dilated = ndimage.binary_dilation(gt_boundary, structure=disk)
    n_pred = float(pred_boundary.sum())
    n_gt = float(gt_boundary.sum())
    if n_pred == 0.0 and n_gt == 0.0:
        return 1.0
    if n_pred == 0.0 or n_gt == 0.0:
        return 0.0
    precision = float((pred_boundary & gt_dilated).sum()) / n_pred
    recall = float((gt_boundary & pred_dilated).sum()) / n_gt
    return 2.0 * precision * recall / (precision + recall + eps)


def _load_gt_mask(path: Path, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    array = np.asarray(Image.open(path).convert("L"))
    mask = (array > 0).astype(bool)
    if size is not None and mask.shape != size:
        image = Image.fromarray(mask.astype(np.uint8) * 255).resize((size[1], size[0]), Image.NEAREST)
        mask = np.asarray(image) > 0
    return mask


def _mask_iou_counts(prediction: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, float]:
    """IoU plus raw intersection/union pixel counts (for cumulative cIoU)."""
    if target.numel() == 0 or target.dtype != torch.bool:
        return 0.0, 0.0, 0.0
    predicted = (prediction > 0).to(device=target.device, dtype=torch.bool)
    resized = (
        F.interpolate(
            target.to(dtype=torch.float32)[None, None],
            size=prediction.shape[-2:],
            mode="nearest",
        )
        .squeeze(0)
        .squeeze(0)
        > 0
    )
    intersection = (predicted & resized).sum().float().item()
    union = (predicted | resized).sum().float().item()
    iou = intersection / union if union > 0 else 0.0
    return iou, intersection, union


def _eval_image_record(
    record: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    template: str,
    seg_id: int,
    device: torch.device,
    adapter: VILASegmentationAdapter,
    capability: SegmentationCapability,
    projector: Any,
    provider: SAM2ImageFeatureProvider,
    decoder: Any,
    hidden_layer: int,
    spatial_scale: int,
    head: str,
) -> Dict[str, Any]:
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
            refined = result.mask_logits.detach().to(dtype=torch.float32)
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
            refined = _sam2_refine_trainable(provider, sample["rgb"], result.mask_logits, device)
    target = sample["target_mask"][0, 0, 0].detach()
    coarse_iou, coarse_inter, coarse_union = _mask_iou_counts(result.mask_logits[0, 0, 0].detach(), target)
    refined_iou, refined_inter, refined_union = _mask_iou_counts(refined[0, 0, 0].detach(), target)
    return {
        "sample_id": record["sample_id"],
        "media_id": record["media_id"],
        "query": record["query"],
        "coarse_iou": coarse_iou,
        "refined_iou": refined_iou,
        "coarse_inter": coarse_inter,
        "coarse_union": coarse_union,
        "refined_inter": refined_inter,
        "refined_union": refined_union,
    }


def _eval_video_record(
    record: Dict[str, Any],
    model: Any,
    tokenizer: Any,
    template: str,
    seg_id: int,
    device: torch.device,
    adapter: VILASegmentationAdapter,
    capability: SegmentationCapability,
    projector: Any,
    provider: SAM2ImageFeatureProvider,
    propagator: SAM2VideoMaskPropagator,
    decoder: Any,
    hidden_layer: int,
    head: str,
) -> Dict[str, Any]:
    sample = _video_sample(record, model, tokenizer, template, seg_id, device)
    with torch.no_grad(), seg_training_active(True), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        query_states = adapter.extract_query_states_training(
            sample["input_ids"], sample["media"], sample["media_config"],
            sample["query_mask"], sample["attention_mask"], hidden_layer=hidden_layer,
        )
        seg_positions = (
            sample["query_mask"][:, : sample["seg_position"] + 1].sum(dim=1) - 1
        ).to(dtype=torch.long, device=device)
        dense = provider.encode_frames(sample["rgb"])
        if head == "lisa":
            seg_state = query_states.states.to(dtype=torch.float32)
            seg_state = torch.nan_to_num(seg_state, nan=0.0, posinf=0.0, neginf=0.0)
            seg_state = seg_state[torch.arange(seg_state.shape[0], device=device), seg_positions]
            with torch.autocast(device_type="cuda", enabled=False):
                result = decoder(seg_state, dense.features, dense.frame_mask, dense.high_res_features)
            refined_anchor = result.mask_logits[:, :, :1][0, :, 0].to(dtype=torch.float32)  # [N,H,W]
        else:
            projected = projector(query_states.states, query_states.mask, seg_positions)
            result = capability(
                GroundingBatch(
                    query_states=projected,
                    query_mask=torch.ones(1, projected.shape[1], dtype=torch.bool, device=device),
                    dense_features=dense.features,
                    frame_mask=dense.frame_mask,
                    sample_ids=(record["sample_id"],),
                ),
                {"enabled": True, "task": "video"},
            )
            anchor_rgb = RGBFrameBatch(
                frames=sample["rgb"].frames[:, :1],
                frame_mask=torch.ones(1, 1, dtype=torch.bool),
                sample_ids=(record["sample_id"],),
            )
            refined_anchor = _sam2_refine_trainable(
                provider, anchor_rgb, result.mask_logits[:, :, :1], device
            )[0, :, 0]  # [N,H,W] at frame resolution

    media_dir = Path(record["media_path"])
    frame_paths = sorted(media_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"video has no frames: {media_dir}")
    arrays = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as handle:
            arrays.append(np.asarray(handle.convert("RGB"), dtype=np.uint8).copy())
    full_frames = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous().unsqueeze(0)
    full_batch = RGBFrameBatch(
        frames=full_frames,
        frame_mask=torch.ones(1, len(arrays), dtype=torch.bool),
        sample_ids=(record["sample_id"],),
    )
    anchor_index = int(record["frame_indices"][0])
    anchor_compact = frame_paths.index(media_dir / f"{anchor_index:05d}.jpg")
    anchor_logits = refined_anchor.detach()
    propagated = propagator.propagate(
        full_batch,
        torch.tensor([anchor_compact], dtype=torch.long),
        anchor_logits.unsqueeze(0),
    )
    predicted = (propagated.mask_logits[0, 0].detach().cpu().numpy() > 0)  # [T,H,W]

    # Ground-truth masks: per-object directory for the official valid split.
    target_id = record["target_id"]
    annotations_dir = media_dir.parent.parent / "Annotations" / media_dir.name / str(target_id)
    gt_frames = sorted(annotations_dir.glob("*.png")) if annotations_dir.is_dir() else []
    j_scores, f_scores = [], []
    frame_count = len(arrays)
    for gt_path in gt_frames:
        frame_index = int(gt_path.stem)
        if frame_index >= frame_count:
            continue
        ground_truth = _load_gt_mask(gt_path)
        prediction = predicted[frame_index]
        if prediction.shape != ground_truth.shape:
            prediction = np.asarray(
                Image.fromarray((prediction * 255).astype(np.uint8)).resize(
                    (ground_truth.shape[1], ground_truth.shape[0]), Image.NEAREST
                )
            ) > 0
        intersection = float((prediction & ground_truth).sum())
        union = float((prediction | ground_truth).sum())
        j_scores.append(intersection / union if union > 0 else 0.0)
        f_scores.append(_boundary_f_score_official(prediction, ground_truth))
    mean_j = float(np.mean(j_scores)) if j_scores else 0.0
    mean_f = float(np.mean(f_scores)) if f_scores else 0.0
    anchor_iou = (
        _mask_iou(
            anchor_logits[0].detach(),
            _target_tensor(str(record["mask_paths"][0]), device)[0][0, 0, 0].detach(),
        )
        if record["mask_paths"][0]
        else 0.0
    )
    return {
        "sample_id": record["sample_id"],
        "media_id": record["media_id"],
        "query": record["query"],
        "frames": frame_count,
        "gt_frames": len(j_scores),
        "anchor_iou": anchor_iou,
        "j": mean_j,
        "f": mean_f,
        "jf": (mean_j + mean_f) / 2.0,
    }


def _dataset_of(sample_id: str) -> str:
    """Map a sample id to its benchmark split: refcoco / refcocoplus / refcocog."""
    if sample_id.startswith("refcocoplus."):
        return "refcocoplus"
    if sample_id.startswith("refcocog."):
        return "refcocog"
    return "refcoco"


def _paper_image_metrics(rows: List[Dict[str, Any]], iou_key: str, inter_key: str, union_key: str) -> Dict[str, Any]:
    """Paper-aligned image metrics for one group.

    - mIoU    : mean of per-expression IoU
    - cIoU    : cumulative IoU = sum(intersection) / sum(union)  (RefCOCO overall-IoU convention)
    - Acc@0.5 / Acc@0.75 : fraction of expressions with IoU above the threshold
    """
    ious = [row[iou_key] for row in rows if iou_key in row and row[iou_key] is not None]
    inters = [row[inter_key] for row in rows if inter_key in row and row[inter_key] is not None]
    unions = [row[union_key] for row in rows if union_key in row and row[union_key] is not None]
    if not ious:
        return {"samples": 0, "mIoU": None, "cIoU": None, "Acc@0.5": None, "Acc@0.75": None}
    total_union = float(sum(unions)) if unions else 0.0
    ciou = float(sum(inters)) / total_union if (inters and total_union > 0) else None
    return {
        "samples": len(rows),
        "mIoU": float(np.mean(ious)),
        "cIoU": ciou,
        "Acc@0.5": float(np.mean([iou >= 0.5 for iou in ious])),
        "Acc@0.75": float(np.mean([iou >= 0.75 for iou in ious])),
    }


def _paper_image_summary(image_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Group image results by RefCOCO / RefCOCO+ / RefCOCOg and report paper metrics."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in image_results:
        if "coarse_iou" not in row:
            continue
        groups.setdefault(_dataset_of(row.get("sample_id", "")), []).append(row)
    summary: Dict[str, Any] = {}
    for dataset in ("refcoco", "refcocoplus", "refcocog"):
        rows = groups.get(dataset, [])
        summary[dataset] = {
            "coarse": _paper_image_metrics(rows, "coarse_iou", "coarse_inter", "coarse_union"),
            "refined": _paper_image_metrics(rows, "refined_iou", "refined_inter", "refined_union"),
        }
    all_rows = [row for row in image_results if "coarse_iou" in row]
    summary["all"] = {
        "coarse": _paper_image_metrics(all_rows, "coarse_iou", "coarse_inter", "coarse_union"),
        "refined": _paper_image_metrics(all_rows, "refined_iou", "refined_inter", "refined_union"),
    }
    return summary


def _paper_video_summary(video_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Official Ref-YT-VOS aggregate: mean over videos of per-video (J+F)/2."""
    valid = [row for row in video_results if "j" in row and row["j"] is not None]
    if not valid:
        return {"samples": 0, "mean_j": None, "mean_f": None, "mean_jf": None, "mean_anchor_iou": None}
    return {
        "samples": len(valid),
        "mean_j": float(np.mean([row["j"] for row in valid])),
        "mean_f": float(np.mean([row["f"] for row in valid])),
        "mean_jf": float(np.mean([row["jf"] for row in valid])),
        "mean_anchor_iou": float(np.mean([row["anchor_iou"] for row in valid]))
        if all("anchor_iou" in row and row["anchor_iou"] is not None for row in valid)
        else None,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vila-model", type=Path)
    parser.add_argument("--sam2-source-root", type=Path)
    parser.add_argument("--sam2-checkpoint", type=Path)
    parser.add_argument("--manifest-dir", type=Path)
    parser.add_argument("--video-manifest-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--skip-video", action="store_true", help="only evaluate image records")
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        image_shards = sorted(args.output.glob("image_results.*.json"))
        video_shards = sorted(args.output.glob("video_results.*.json"))
        image_results: List[Dict[str, Any]] = []
        video_results: List[Dict[str, Any]] = []
        for shard in image_shards:
            image_results.extend(json.loads(shard.read_text(encoding="utf-8")))
        for shard in video_shards:
            video_results.extend(json.loads(shard.read_text(encoding="utf-8")))

        def mean_metric(results: List[Dict[str, Any]], key: str) -> Optional[float]:
            values = [item[key] for item in results if key in item and item[key] is not None]
            return float(np.mean(values)) if values else None

        summary = {
            "image": {
                "samples": len(image_results),
                "mean_coarse_iou": mean_metric(image_results, "coarse_iou"),
                "mean_refined_iou": mean_metric(image_results, "refined_iou"),
                "paper": _paper_image_summary(image_results),
            },
            "video": {
                "samples": len(video_results),
                "mean_anchor_iou": mean_metric(video_results, "anchor_iou"),
                "mean_j": mean_metric(video_results, "j"),
                "mean_f": mean_metric(video_results, "f"),
                "mean_jf": mean_metric(video_results, "jf"),
                "paper": _paper_video_summary(video_results),
            },
        }
        (args.output / "image_results.json").write_text(json.dumps(image_results, indent=2), encoding="utf-8")
        (args.output / "video_results.json").write_text(json.dumps(video_results, indent=2), encoding="utf-8")
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0

    required = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "vila_model": args.vila_model,
        "sam2_source_root": args.sam2_source_root,
        "sam2_checkpoint": args.sam2_checkpoint,
        "manifest_dir": args.manifest_dir,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"missing required arguments: {', '.join(missing)}")

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("evaluation requires CUDA")
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
    propagator = SAM2VideoMaskPropagator(provider)
    propagator.initialize()

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
    if checkpoint.get("sam2_mask_decoder"):
        mask_decoder = provider._predictor.model.sam_mask_decoder
        current = mask_decoder.state_dict()
        compatible = {key: value for key, value in checkpoint["sam2_mask_decoder"].items() if key in current}
        mask_decoder.load_state_dict(compatible, strict=False)

    capability = SegmentationCapability(decoder)
    adapter = VILASegmentationAdapter(model, provider, capability)

    image_results: List[Dict[str, Any]] = []
    video_results: List[Dict[str, Any]] = []
    image_records = [
        record
        for record in _load_records(args.manifest_dir)
        if record["control_kind"] == "positive" and record["media_type"] == "image"
    ]
    if args.max_samples:
        image_records = image_records[: args.max_samples]
    image_records = image_records[args.shard_index :: args.shard_count]
    for record in tqdm(image_records, desc=f"image eval shard {args.shard_index}", ncols=100):
        try:
            result = _eval_image_record(
                record, model, tokenizer, template, seg_id, device,
                adapter, capability, projector, provider, decoder,
                hidden_layer, spatial_scale, head,
            )
        except Exception as error:  # noqa: BLE001
            result = {"sample_id": record["sample_id"], "error": str(error)}
        image_results.append(result)

    video_manifest_dir = args.video_manifest_dir or args.manifest_dir
    if not args.skip_video:
        video_records = [
            record
            for record in _load_records(video_manifest_dir)
            if record["control_kind"] == "positive" and record["media_type"] == "video"
        ]
        if args.max_samples:
            video_records = video_records[: args.max_samples]
        video_records = video_records[args.shard_index :: args.shard_count]
        for record in tqdm(video_records, desc=f"video eval shard {args.shard_index}", ncols=100):
            try:
                result = _eval_video_record(
                    record, model, tokenizer, template, seg_id, device,
                    adapter, capability, projector, provider, propagator, decoder,
                    hidden_layer, head,
                )
            except Exception as error:  # noqa: BLE001
                result = {"sample_id": record["sample_id"], "error": str(error)}
            video_results.append(result)

    def mean_metric(results: List[Dict[str, Any]], key: str) -> Optional[float]:
        values = [item[key] for item in results if key in item and item[key] is not None]
        return float(np.mean(values)) if values else None

    summary = {
        "image": {
            "samples": len(image_results),
            "mean_coarse_iou": mean_metric(image_results, "coarse_iou"),
            "mean_refined_iou": mean_metric(image_results, "refined_iou"),
            "paper": _paper_image_summary(image_results),
        },
        "video": {
            "samples": len(video_results),
            "mean_anchor_iou": mean_metric(video_results, "anchor_iou"),
            "mean_j": mean_metric(video_results, "j"),
            "mean_f": mean_metric(video_results, "f"),
            "mean_jf": mean_metric(video_results, "jf"),
            "paper": _paper_video_summary(video_results),
        },
    }
    (args.output / f"image_results.{args.shard_index}.json").write_text(json.dumps(image_results, indent=2), encoding="utf-8")
    if not args.skip_video:
        (args.output / f"video_results.{args.shard_index}.json").write_text(json.dumps(video_results, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
