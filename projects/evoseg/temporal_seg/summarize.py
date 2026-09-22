"""Create paired statistics, figures, status, and the Chinese morning report."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils


SUCCESS = {"success", "success_no_seg"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_records(path):
    by_key = {}
    if not path.exists():
        return []
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            by_key[record["key"]] = record
    return list(by_key.values())


def mean_or_none(values):
    return float(np.mean(values)) if values else None


def fmt(value, scale=100):
    return "尚无完整样本" if value is None else f"{value * scale:.2f}"


def bootstrap_by_video(rows, value_key, n_boot=2000, seed=42):
    by_video = defaultdict(list)
    for row in rows:
        by_video[row["video_id"]].append(float(row[value_key]))
    videos = sorted(by_video)
    if not videos:
        return None, None, None
    point = float(np.mean([v for values in by_video.values() for v in values]))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        sampled = rng.choice(videos, size=len(videos), replace=True)
        values = [value for video in sampled for value in by_video[video]]
        boots.append(float(np.mean(values)))
    low, high = np.quantile(boots, [0.025, 0.975])
    return point, float(low), float(high)


def object_type_means(records):
    grouped = defaultdict(list)
    for record in records:
        if record.get("status") not in SUCCESS:
            continue
        key = (
            record["video_id"],
            record["object_id"],
            record["description_type"],
            int(record["frame_budget"]),
        )
        grouped[key].append(record)
    output = {}
    for key, values in grouped.items():
        output[key] = {
            "J": mean_or_none([v["J"] for v in values]),
            "F": mean_or_none([v["F"] for v in values]),
            "J_and_F": mean_or_none([v["J_and_F"] for v in values]),
            "length_words": mean_or_none([v["description_length_words"] for v in values]),
            "n_expressions": len(values),
            "visual_tokens": mean_or_none([v["token_info"]["visual_tokens"] for v in values]),
            "peak_memory_bytes": max(v["peak_memory_bytes"] for v in values),
            "latency_seconds": mean_or_none(
                [
                    v["latency_seconds_synchronized"]
                    for v in values
                    if v.get("thermal_state") == "warm"
                ]
            ),
        }
    return output


def paired_rows(means):
    a_rows = []
    objects = sorted({(v, o) for v, o, _t, b in means if b == 16})
    for video, obj in objects:
        static = means.get((video, obj, "static", 16))
        dynamic = means.get((video, obj, "dynamic", 16))
        if not static or not dynamic:
            continue
        a_rows.append(
            {
                "row_type": "A_object_pair",
                "video_id": video,
                "object_id": obj,
                "description_type": "dynamic_minus_static",
                "frame_budget_from": 16,
                "frame_budget_to": 16,
                "value_from": static["J_and_F"],
                "value_to": dynamic["J_and_F"],
                "paired_delta": dynamic["J_and_F"] - static["J_and_F"],
                "n_expressions_from": static["n_expressions"],
                "n_expressions_to": dynamic["n_expressions"],
                "mean_length_words_from": static["length_words"],
                "mean_length_words_to": dynamic["length_words"],
            }
        )
    b_rows = []
    object_types = sorted({(v, o, t) for v, o, t, _b in means})
    for video, obj, typ in object_types:
        low = means.get((video, obj, typ, 8))
        high = means.get((video, obj, typ, 32))
        if not low or not high:
            continue
        b_rows.append(
            {
                "row_type": "B_object_pair",
                "video_id": video,
                "object_id": obj,
                "description_type": typ,
                "frame_budget_from": 8,
                "frame_budget_to": 32,
                "value_from": low["J_and_F"],
                "value_to": high["J_and_F"],
                "paired_delta": high["J_and_F"] - low["J_and_F"],
                "n_expressions_from": low["n_expressions"],
                "n_expressions_to": high["n_expressions"],
                "mean_length_words_from": low["length_words"],
                "mean_length_words_to": high["length_words"],
            }
        )
    return a_rows, b_rows


def write_csv(path, rows):
    fields = [
        "row_type",
        "video_id",
        "object_id",
        "description_type",
        "frame_budget_from",
        "frame_budget_to",
        "value_from",
        "value_to",
        "paired_delta",
        "n_expressions_from",
        "n_expressions_to",
        "mean_length_words_from",
        "mean_length_words_to",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "n_pairs",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(run_dir, means, a_rows, b_rows):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"matplotlib unavailable: {exc}"]
    errors = []
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(exist_ok=True)

    if a_rows:
        fig, ax = plt.subplots(figsize=(5, 5))
        for row in a_rows:
            ax.plot([0, 1], [row["value_from"], row["value_to"]], "o-", alpha=0.35)
        ax.set_xticks([0, 1], ["Static", "Dynamic"])
        ax.set_ylabel("J&F")
        ax.set_title("Same-object paired descriptions (N=16)")
        fig.tight_layout()
        fig.savefig(figure_dir / "same_object_static_dynamic.png", dpi=180)
        plt.close(fig)

    curves = defaultdict(lambda: defaultdict(list))
    for (_video, _obj, typ, budget), values in means.items():
        curves[typ][budget].append(values["J_and_F"])
    if curves:
        fig, ax = plt.subplots(figsize=(6, 4))
        for typ in sorted(curves):
            budgets = sorted(curves[typ])
            ax.plot(
                budgets,
                [np.mean(curves[typ][b]) for b in budgets],
                "o-",
                label=typ,
            )
        ax.set_xlabel("VLM visible frames")
        ax.set_ylabel("object-weighted J&F")
        ax.legend()
        ax.set_xticks([8, 16, 32])
        fig.tight_layout()
        fig.savefig(figure_dir / "frame_budget_curve.png", dpi=180)
        plt.close(fig)
    return errors


def make_case_figure(run_dir, records, manifest):
    candidates = defaultdict(dict)
    for record in records:
        if record.get("status") in SUCCESS and record["frame_budget"] == 16:
            candidates[(record["video_id"], record["object_id"])][
                record["description_type"]
            ] = record
    pair = next(
        ((key, value) for key, value in candidates.items() if {"static", "dynamic"} <= set(value)),
        None,
    )
    if pair is None:
        return None
    (video_id, object_id), values = pair
    item = next(
        item
        for item in manifest["objects"]
        if item["video_id"] == video_id and item["object_id"] == object_id
    )
    index = len(item["evaluation_frame_names"]) // 2
    frame_name = item["evaluation_frame_names"][index]
    image_path = Path(manifest["dataset"]["image_root"]) / video_id / f"{frame_name}.jpg"
    image = Image.open(image_path).convert("RGB")
    panels = [("image", image)]
    gt_path = Path(item["evaluation_mask_paths"][index])
    gt = np.asarray(Image.open(gt_path).convert("L")) > 0 if gt_path.exists() else np.zeros((image.height, image.width), bool)
    for label, mask in [("GT", gt)]:
        panel = image.copy()
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay.putalpha(Image.fromarray(mask.astype(np.uint8) * 110))
        color = Image.new("RGBA", image.size, (0, 255, 0, 0))
        color.putalpha(overlay.getchannel("A"))
        panel = Image.alpha_composite(panel.convert("RGBA"), color).convert("RGB")
        panels.append((label, panel))
    for typ in ("static", "dynamic"):
        mask_records = json.loads((run_dir / values[typ]["rle_masks_path"]).read_text())
        mask = mask_utils.decode(mask_records[index]["rle"]).astype(bool)
        panel = image.copy()
        alpha = Image.fromarray(mask.astype(np.uint8) * 110)
        color = Image.new("RGBA", image.size, (255, 0, 0, 0))
        color.putalpha(alpha)
        panel = Image.alpha_composite(panel.convert("RGBA"), color).convert("RGB")
        panels.append((typ, panel))
    canvas = Image.new("RGB", (image.width * len(panels), image.height + 36), "white")
    draw = ImageDraw.Draw(canvas)
    for panel_index, (label, panel) in enumerate(panels):
        canvas.paste(panel, (panel_index * image.width, 36))
        draw.text((panel_index * image.width + 8, 8), label, fill="black")
    output = run_dir / "figures" / "real_mask_case.png"
    canvas.save(output)
    return str(output)


def length_strata(means, a_rows):
    """Length diagnostics that preserve the object-level statistical unit."""
    object_rows = [
        {
            "video_id": video,
            "object_id": obj,
            "type": typ,
            "length_words": value["length_words"],
            "J_and_F": value["J_and_F"],
        }
        for (video, obj, typ, budget), value in means.items()
        if budget == 16
    ]
    if len(object_rows) < 6:
        return {"definition": "insufficient object-level N=16 samples", "object_type": [], "paired_a": []}

    q1, q2 = np.quantile(
        [row["length_words"] for row in object_rows], [1 / 3, 2 / 3]
    )
    predicates = (
        ("short", lambda x: x <= q1),
        ("medium", lambda x: q1 < x <= q2),
        ("long", lambda x: x > q2),
    )
    object_output = []
    for label, predicate in predicates:
        for typ in ("static", "dynamic", "hybrid"):
            rows = [
                row
                for row in object_rows
                if row["type"] == typ and predicate(row["length_words"])
            ]
            object_output.append(
                {
                    "stratum": label,
                    "type": typ,
                    "n_objects": len(rows),
                    "J_and_F": mean_or_none([row["J_and_F"] for row in rows]),
                    "interpret_cautiously": len(rows) < 10,
                }
            )

    paired_output = []
    if len(a_rows) >= 3:
        pair_lengths = [
            (row["mean_length_words_from"] + row["mean_length_words_to"]) / 2
            for row in a_rows
        ]
        pair_q1, pair_q2 = np.quantile(pair_lengths, [1 / 3, 2 / 3])
        pair_predicates = (
            ("short", lambda x: x <= pair_q1),
            ("medium", lambda x: pair_q1 < x <= pair_q2),
            ("long", lambda x: x > pair_q2),
        )
        for label, predicate in pair_predicates:
            rows = [
                row
                for row, length in zip(a_rows, pair_lengths)
                if predicate(length)
            ]
            point, low, high = bootstrap_by_video(rows, "paired_delta")
            paired_output.append(
                {
                    "stratum": label,
                    "n_object_pairs": len(rows),
                    "dynamic_minus_static": point,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "interpret_cautiously": len(rows) < 10,
                }
            )
    return {
        "definition": (
            "global tertiles of object/type mean expression length at N=16; "
            "paired A strata use tertiles of the pair's mean Static/Dynamic length"
        ),
        "object_type": object_output,
        "paired_a": paired_output,
    }


def summarize(run_dir: Path):
    records = read_records(run_dir / "predictions.jsonl")
    means = object_type_means(records)
    a_rows, b_rows = paired_rows(means)
    a_point, a_low, a_high = bootstrap_by_video(a_rows, "paired_delta")
    b_stats = {}
    for typ in ("static", "dynamic", "hybrid"):
        rows = [r for r in b_rows if r["description_type"] == typ]
        b_stats[typ] = bootstrap_by_video(rows, "paired_delta")
    summary_rows = [
        {
            "row_type": "A_bootstrap_summary",
            "description_type": "dynamic_minus_static",
            "frame_budget_from": 16,
            "frame_budget_to": 16,
            "paired_delta": a_point,
            "bootstrap_ci_low": a_low,
            "bootstrap_ci_high": a_high,
            "n_pairs": len(a_rows),
        }
    ]
    for typ, (point, low, high) in b_stats.items():
        summary_rows.append(
            {
                "row_type": "B_bootstrap_summary",
                "description_type": typ,
                "frame_budget_from": 8,
                "frame_budget_to": 32,
                "paired_delta": point,
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "n_pairs": len([r for r in b_rows if r["description_type"] == typ]),
            }
        )
    write_csv(run_dir / "paired_results.csv", a_rows + b_rows + summary_rows)
    errors = make_plots(run_dir, means, a_rows, b_rows)

    config_path = run_dir / "run_config.json"
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists() and config_path.exists():
        configured = json.loads(config_path.read_text())["protocol"]["manifest"]
        manifest_path = Path(configured)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest:
        try:
            make_case_figure(run_dir, records, manifest)
        except Exception as exc:
            errors.append(f"case figure: {exc}")

    successful = [r for r in records if r.get("status") in SUCCESS]
    failed = [r for r in records if str(r.get("status", "")).startswith("failed")]
    videos = {r["video_id"] for r in successful}
    objects = {(r["video_id"], r["object_id"]) for r in successful}
    token_by_budget = defaultdict(list)
    memory_by_budget = defaultdict(list)
    latency_by_budget = defaultdict(list)
    for record in successful:
        budget = int(record["frame_budget"])
        token_by_budget[budget].append(record["token_info"]["visual_tokens"])
        memory_by_budget[budget].append(record["peak_memory_bytes"] / 2**30)
        if record.get("thermal_state") == "warm":
            latency_by_budget[budget].append(record["latency_seconds_synchronized"])

    def cost_line(budget):
        tokens = token_by_budget[budget]
        memory = memory_by_budget[budget]
        return (
            f"N={budget}: visual tokens 均值={mean_or_none(tokens)}, "
            f"范围=[{min(tokens) if tokens else None}, {max(tokens) if tokens else None}], "
            f"实测最大峰值显存={max(memory) if memory else None} GiB, "
            f"热运行同步延迟={mean_or_none(latency_by_budget[budget])} s"
        )

    dynamic_b = b_stats["dynamic"]
    accuracy_lines = ["| 描述类型 | N=8 J&F | N=16 J&F | N=32 J&F | N=32−N=8 (95% CI) |", "|---|---:|---:|---:|---:|"]
    for typ in ("static", "dynamic", "hybrid"):
        by_budget = {}
        for budget in (8, 16, 32):
            by_budget[budget] = mean_or_none(
                [
                    value["J_and_F"]
                    for (_video, _obj, row_type, row_budget), value in means.items()
                    if row_type == typ and row_budget == budget
                ]
            )
        point, low, high = b_stats[typ]
        accuracy_lines.append(
            f"| {typ} | {fmt(by_budget[8])} | {fmt(by_budget[16])} | "
            f"{fmt(by_budget[32])} | {fmt(point)} [{fmt(low)}, {fmt(high)}] |"
        )
    accuracy_table = "\n".join(accuracy_lines)
    quality_path = run_dir / "logs" / "smoke_quality.json"
    quality = json.loads(quality_path.read_text()) if quality_path.exists() else {}
    interface_verified = bool(quality.get("passed")) and not failed
    enough_pairs = len(a_rows) >= 24 and len(
        [r for r in b_rows if r["description_type"] == "dynamic"]
    ) >= 24
    a_excludes_zero = (
        a_low is not None
        and a_high is not None
        and (a_low > 0 or a_high < 0)
    )
    dynamic_crosses_zero = (
        dynamic_b[1] is None
        or dynamic_b[2] is None
        or dynamic_b[1] <= 0 <= dynamic_b[2]
    )
    if not interface_verified:
        priority = "分割接口/实现问题"
        priority_reason = "smoke 接口质量门尚未通过，不能先解释模型机制"
    elif not enough_pairs:
        priority = "数据覆盖问题"
        priority_reason = "完整对象内配对量仍不足以支持稳定判断"
    else:
        priority = "时序表示"
        priority_reason = (
            "接口与数据检查已通过，但增加动态描述可见帧的区间仍跨零；"
            "下一步应先测 `[SEG]` 表示是否随跨帧证据变化"
        )

    supported = []
    if interface_verified:
        supported.append("VLM 帧预算与固定分割帧已经分离，smoke 质量门通过")
    if a_excludes_zero:
        direction = "高于" if a_point > 0 else "低于"
        supported.append(
            f"当前模型下 Dynamic 的对象内 J&F 显著{direction} Static"
        )
    if not dynamic_crosses_zero:
        direction = "提高" if dynamic_b[0] > 0 else "降低"
        supported.append(f"N=32 相对 N=8 对动态描述的 J&F 稳定{direction}")
    unsupported = []
    if dynamic_crosses_zero:
        unsupported.append("更多可见帧会改善动态指代（当前 95% 区间跨零）")
    unsupported.extend(
        [
            "差异由时序理解能力单独导致",
            "不同帧数条件具有固定的总计算预算",
        ]
    )
    strata = length_strata(means, a_rows)
    report = f"""# EvoSeg 跨帧过程诊断：首轮报告

## ① 实际跑了什么、覆盖多少视频和对象

使用原始 Sa2VA-Qwen3-VL-4B，在同一 manifest 上分离 `vlm_frames` 与固定的分割/评价帧。当前有 {len(successful)} 条成功结果，覆盖 {len(videos)} 个视频、{len(objects)} 个对象；失败 {len(failed)} 条。本结果是诊断子集，不是完整官方基准。

## ② Static/Dynamic 配对差异

N=16 时，先在对象内分别平均同类型多表达，再计算 Dynamic−Static。完整配对对象数为 {len(a_rows)}；J&F 差值为 {fmt(a_point)} 个百分点，按源视频 2000 次配对 bootstrap 的 95% 区间为 [{fmt(a_low)}, {fmt(a_high)}]。该差异只表示当前模型下的描述条件差异，不等价于“模型完全不理解时序”。

## ③ 增加帧数是否改善、代价是多少

动态描述的对象内 N=32−N=8 J&F 差值为 {fmt(dynamic_b[0])} 个百分点，95% 区间为 [{fmt(dynamic_b[1])}, {fmt(dynamic_b[2])}]，完整配对对象数为 {len([r for r in b_rows if r['description_type'] == 'dynamic'])}。

{accuracy_table}

- {cost_line(8)}
- {cost_line(16)}
- {cost_line(32)}

总 token 随帧数增加；这里没有固定计算预算。冷启动模型加载与首条推理已在 `run_config.json`/`predictions.jsonl` 单独标记。并行抢占环境下延迟不作正式效率结论。

## ④ 哪些解释得到支持，哪些尚不能判断

只有完整对象内配对进入上述主统计。得到支持：{'；'.join(supported) if supported else '当前尚无机制性解释得到支持'}。尚不能判断：{'；'.join(unsupported)}。这些结果只说明当前模型下的描述条件差异，不能直接写成“模型完全不理解时序”。失败与重试保留在逐表达记录中。

长度分层使用对象作为统计单位，避免多表达对象获得额外权重；`interpret_cautiously=true` 的稀疏分层不作强结论：

```json
{json.dumps(strata, ensure_ascii=False)}
```

## ⑤ 下一步优先项

下一步优先验证 **{priority}**：{priority_reason}。当前绘图/汇总警告：{errors or '无'}。
"""
    (run_dir / "MORNING_REPORT.md").write_text(report)

    next_steps = f"""# Next experiments

本文件由当前实测结果生成，不预设模型会因更多帧而改善。

## 当前决定

优先验证 **{priority}**。原因：{priority_reason}。

- 当前完整 A 配对：{len(a_rows)}；Dynamic−Static={fmt(a_point)} 个百分点，95% CI=[{fmt(a_low)}, {fmt(a_high)}]。
- 当前完整动态 B 配对：{len([r for r in b_rows if r['description_type'] == 'dynamic'])}；N=32−N=8={fmt(dynamic_b[0])} 个百分点，95% CI=[{fmt(dynamic_b[1])}, {fmt(dynamic_b[2])}]。
- smoke 接口质量门：{'通过' if quality.get('passed') else '未通过或尚未完成'}；失败结果：{len(failed)}。

## 最小下一步

1. 在不改变 SAM2 输入与传播的前提下，保存同对象 N=8/16/32 的 `[SEG]` 隐向量，测余弦距离及其与对象内 J&F 变化的关系。
2. 只沿用官方 Static/Dynamic/Hybrid 类型和原始表达长度分层复核表示变化；不新增查询、标签或人工事件分类，不据结果挑样本。
3. 若 `[SEG]` 表示几乎不随新增帧变化，再设计最小的时序表示消融；本轮不启动四组大型训练。
"""
    (run_dir / "NEXT_EXPERIMENTS.md").write_text(next_steps)

    status_path = run_dir / "STATUS.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status.update(
        {
            "updated_at": utc_now(),
            "successful_results": len(successful),
            "failed_results": len(failed),
            "covered_videos": len(videos),
            "covered_objects": len(objects),
            "complete_a_pairs": len(a_rows),
            "complete_b_dynamic_pairs": len(
                [r for r in b_rows if r["description_type"] == "dynamic"]
            ),
            "summary_warnings": errors,
        }
    )
    tmp = status_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(status_path)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(Path(args.run_dir).resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
