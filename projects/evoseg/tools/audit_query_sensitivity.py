"""Audit whether the S4b query representation actually carries referring info.

The core hypothesis behind the weak segmentation is that the language->mask
information flow collapses: the projected query states (or the decoder masks)
may be almost identical for different queries on the same image, which would
make the model an image-prior shortcut (it "guesses" a mask without listening
to the query).  This tool measures:

  * pairwise cosine similarity of projected query states for same-image
    different-query pairs vs. different-image pairs,
  * decoder mask overlap (IoU) between swapped queries on the same image,
  * per-sample query-sensitivity delta of coarse IoU under a swapped query.

All numbers are diagnostics only; no weights are trained or modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import yaml

_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_ROOT_FOR_IMPORT))

from llava.evo_seg.capability import SegmentationCapability
from llava.evo_seg.contracts import GroundingBatch
from llava.evo_seg.decoder import QueryConditionedSpatialDecoder
from llava.evo_seg.sam2_adapter import SAM2ImageFeatureProvider
from llava.evo_seg.sam2_lisa_head import Sam2LisaHead
from llava.evo_seg.sam2_video_adapter import build_sam2_video_image_predictor
from llava.evo_seg.training import (
    GroundingProjector,
    SegEmbeddingInjector,
    apply_lora,
    seg_training_active,
)
from llava.evo_seg.vila_adapter import VILASegmentationAdapter
from scripts.evo_seg.train_s4b import _image_sample, _mask_iou, _upsample_dense


def _load_config(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_records(path: Path) -> List[Dict]:
    records = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vila-model", type=Path, required=True)
    parser.add_argument("--sam2-source-root", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--pairs-file", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-pairs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hidden-layers",
        default="-1,-16,-24",
        help="comma-separated hidden-layer indices to scan for query-state discrimination",
    )
    args = parser.parse_args(argv)

    config = _load_config(args.config.expanduser().resolve())
    training = dict(config.get("training", {}))
    decoder_config = dict(config.get("decoder", {}))
    sam2_config = dict(config.get("sam2", {}))
    prompt_config = dict(config.get("prompt", {}))
    hidden_layer = int(training.get("hidden_layer", -1))
    spatial_scale = int(decoder_config.get("spatial_scale", 1))
    template = str(
        prompt_config.get(
            "instruction_template",
            "Segment the object described by this referring expression:\n{query}",
        )
    )
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    import llava

    model = llava.load(str(args.vila_model), device=str(device), device_map={"": str(device)})
    model.eval()
    model.requires_grad_(False)
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
    head = str(config.get("head", "query_decoder"))
    if head == "lisa":
        decoder = Sam2LisaHead(
            hidden_size=int(model.llm.config.hidden_size),
            sam2_model=provider._predictor.model,
            prompt_dim=int(decoder_config.get("feature_dim", 256)),
        ).to(device=device, dtype=torch.float32)
        projector = torch.nn.Identity()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    decoder.load_state_dict(checkpoint["decoder"])
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

    records_by_id = {
        r["sample_id"]: r
        for r in _load_records(args.manifest_dir.expanduser().resolve() / "val.jsonl")
    }
    pairs_file = args.pairs_file.expanduser().resolve() if args.pairs_file else None
    pairs = _load_records(pairs_file) if pairs_file else []
    rng = np.random.RandomState(args.seed)
    rng.shuffle(pairs)
    pairs = pairs[: args.max_pairs]
    if not pairs:
        raise SystemExit("no swap pairs found; pass --pairs-file with same-image pairs")

    def forward_record(record: Dict, hidden_layer_idx: int):
        sample = _image_sample(record, model, tokenizer, template, seg_id, device)
        with torch.no_grad(), seg_training_active(True), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            query_states = adapter.extract_query_states_training(
                sample["input_ids"],
                sample["media"],
                sample["media_config"],
                sample["query_mask"],
                sample["attention_mask"],
                hidden_layer=hidden_layer_idx,
            )
            seg_positions = (
                sample["query_mask"][:, : sample["seg_position"] + 1].sum(dim=1) - 1
            ).to(dtype=torch.long, device=device)
            dense = _upsample_dense(provider.encode_frames(sample["rgb"]), spatial_scale)
            if head == "lisa":
                seg_state = query_states.states.to(dtype=torch.float32)[
                    torch.arange(query_states.states.shape[0], device=device), seg_positions
                ]
                result = decoder(seg_state, dense.features, dense.frame_mask, dense.high_res_features)
                projected = seg_state.detach().float().cpu().unsqueeze(1)
            else:
                projected = projector(
                    query_states.states.to(dtype=torch.float32), query_states.mask, seg_positions
                )
                result = capability(
                    GroundingBatch(
                        query_states=projected,
                        query_mask=torch.ones(1, projected.shape[1], dtype=torch.bool, device=device),
                        dense_features=dense.features.to(dtype=torch.float32),
                        frame_mask=dense.frame_mask,
                        sample_ids=(record["sample_id"],),
                    ),
                    {"enabled": True, "task": "image"},
                )
        return (
            projected.detach().float().cpu(),
            query_states.states.detach().float().cpu(),
            result.mask_logits.detach().float().cpu(),
            sample["target_mask"][0, 0, 0].detach().cpu(),
        )

    hidden_layers = [int(x) for x in args.hidden_layers.split(",") if x.strip()]
    raw_cos_by_layer = {h: [] for h in hidden_layers}
    projected_cos = []
    same_mask_overlap, swapped_mask_overlap = [], []
    pair_records = []
    for pair in pairs:
        left_id = pair.get("left") or pair.get("sample_id_a") or pair.get("left_sample_id")
        right_id = pair.get("right") or pair.get("sample_id_b") or pair.get("right_sample_id")
        left = records_by_id.get(left_id)
        right = records_by_id.get(right_id)
        if left is None or right is None or left.get("media_id") != right.get("media_id"):
            continue
        try:
            qa, raw_a, mask_a, gt_a = forward_record(left, hidden_layers[0])
            qb, raw_b, mask_b, gt_b = forward_record(right, hidden_layers[0])
            layer_raw = {}
            for h in hidden_layers:
                _, raw_ha, _, _ = forward_record(left, h)
                _, raw_hb, _, _ = forward_record(right, h)
                ra = raw_ha.mean(dim=1).squeeze(0)
                rb = raw_hb.mean(dim=1).squeeze(0)
                layer_raw[h] = float(F.cosine_similarity(ra[None], rb[None]).item())
        except Exception as error:  # noqa: BLE001
            print(f"pair {left_id}/{right_id} failed: {error}", flush=True)
            continue
        pair_records.append((left, right))
        # query-state cosine similarity
        va = qa.mean(dim=1).squeeze(0)
        vb = qb.mean(dim=1).squeeze(0)
        projected_cos.append(float(F.cosine_similarity(va[None], vb[None]).item()))
        for h, value in layer_raw.items():
            raw_cos_by_layer[h].append(value)
        # same-image mask overlap (a's mask vs b's mask)
        sa = (mask_a[0, 0, 0].sigmoid() > 0.5)
        sb = (mask_b[0, 0, 0].sigmoid() > 0.5)
        inter = (sa & sb).sum().float()
        union = (sa | sb).sum().float()
        same_mask_overlap.append(float((inter / union.clamp_min(1)).item()))
        # iou of each query vs its own gt and vs the other's gt
        a_own = _mask_iou(mask_a[0, 0, 0], gt_a)
        a_swap = _mask_iou(mask_a[0, 0, 0], gt_b)
        b_own = _mask_iou(mask_b[0, 0, 0], gt_b)
        b_swap = _mask_iou(mask_b[0, 0, 0], gt_a)
        swapped_mask_overlap.append((a_own - a_swap, b_own - b_swap))

    def summary(values):
        arr = np.asarray(values, dtype=float)
        return {
            "n": int(len(arr)),
            "mean": round(float(arr.mean()), 4) if arr.size else None,
            "std": round(float(arr.std()), 4) if arr.size else None,
            "min": round(float(arr.min()), 4) if arr.size else None,
            "max": round(float(arr.max()), 4) if arr.size else None,
        }

    report = {
        "query_state_cosine": {
            "projected_after_projector": summary(projected_cos),
            "raw_vila_states_by_layer": {
                str(h): summary(values) for h, values in raw_cos_by_layer.items()
            },
            "interpretation": (
                "near 1.0 => query representation collapses (image-prior); "
                "low => queries are discriminative"
            ),
        },
        "mask_overlap_same_image": summary(same_mask_overlap),
        "swap_delta_own_minus_other": {
            "n": len(swapped_mask_overlap),
            "mean": round(
                float(np.mean([a for a, _ in swapped_mask_overlap] + [b for _, b in swapped_mask_overlap])), 4
            )
            if swapped_mask_overlap
            else None,
            "interpretation": (
                "positive => own query scores its own target higher than the swapped query; "
                "near 0 or negative => query is ignored"
            ),
        },
        "pair_count": len(pair_records),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
