#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep end-to-end detector/classifier thresholds on local val.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--classifier-weights", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--det-conf-values", type=str, default="0.08,0.12,0.16,0.20,0.24,0.28")
    parser.add_argument("--det-iou-values", type=str, default="0.45,0.50,0.55,0.60")
    parser.add_argument("--box-expand-values", type=str, default="0.00,0.01,0.02,0.04,0.06")
    parser.add_argument("--cls-min-prob-values", type=str, default="0.00,0.10,0.15,0.20,0.25")
    parser.add_argument("--cls-min-margin-values", type=str, default="0.00,0.02,0.05,0.08")
    parser.add_argument("--det-imgsz", type=int, default=1280)
    parser.add_argument("--cls-batch-size", type=int, default=128)
    parser.add_argument("--slice-size", type=int, default=0)
    parser.add_argument("--slice-overlap", type=int, default=0)
    parser.add_argument("--slice-merge-iou", type=float, default=0.50)
    parser.add_argument("--det-tta-mode", type=str, default="none")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output-csv", type=Path, default=Path("reports/sweep_end_to_end_results.csv"))
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def build_command(args: argparse.Namespace, config: dict[str, float]) -> list[str]:
    script_path = Path(__file__).with_name("evaluate_end_to_end.py")
    return [
        sys.executable,
        str(script_path),
        "--images-dir",
        str(args.images_dir),
        "--detector-weights",
        str(args.detector_weights),
        "--classifier-weights",
        str(args.classifier_weights),
        "--source-root",
        str(args.source_root),
        "--device",
        args.device,
        "--det-conf",
        str(config["det_conf"]),
        "--det-iou",
        str(config["det_iou"]),
        "--det-imgsz",
        str(args.det_imgsz),
        "--cls-batch-size",
        str(args.cls_batch_size),
        "--cls-min-prob",
        str(config["cls_min_prob"]),
        "--cls-min-margin",
        str(config["cls_min_margin"]),
        "--box-expand-ratio",
        str(config["box_expand_ratio"]),
        "--slice-size",
        str(args.slice_size),
        "--slice-overlap",
        str(args.slice_overlap),
        "--slice-merge-iou",
        str(args.slice_merge_iou),
        "--det-tta-mode",
        args.det_tta_mode,
        "--iou-threshold",
        str(args.iou_threshold),
    ]


def parse_result(stdout: str) -> dict:
    start = stdout.rfind("{")
    if start < 0:
        raise ValueError("Could not find JSON result in evaluator output.")
    return json.loads(stdout[start:])


def main() -> None:
    args = parse_args()
    det_conf_values = parse_float_list(args.det_conf_values)
    det_iou_values = parse_float_list(args.det_iou_values)
    box_expand_values = parse_float_list(args.box_expand_values)
    cls_min_prob_values = parse_float_list(args.cls_min_prob_values)
    cls_min_margin_values = parse_float_list(args.cls_min_margin_values)

    configs = list(
        itertools.product(
            det_conf_values,
            det_iou_values,
            box_expand_values,
            cls_min_prob_values,
            cls_min_margin_values,
        )
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp",
        "det_conf",
        "det_iou",
        "box_expand_ratio",
        "cls_min_prob",
        "cls_min_margin",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "elapsed_sec",
        "iou_threshold",
        "num_images",
        "total_gt_chars",
        "total_pred_chars",
    ]
    results: list[dict] = []
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

    print(f"Running {len(configs)} configs", flush=True)
    for index, values in enumerate(configs, start=1):
        config = {
            "det_conf": values[0],
            "det_iou": values[1],
            "box_expand_ratio": values[2],
            "cls_min_prob": values[3],
            "cls_min_margin": values[4],
        }
        print(f"[{index}/{len(configs)}] {config}", flush=True)
        completed = subprocess.run(
            build_command(args, config),
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
        result = parse_result(completed.stdout)
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **config,
            **result,
        }
        results.append(row)
        with args.output_csv.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writerow(row)
        print(
            f"  -> f1={row['f1']:.6f} precision={row['precision']:.6f} recall={row['recall']:.6f} "
            f"tp={row['tp']} fp={row['fp']} fn={row['fn']}",
            flush=True,
        )

    results.sort(key=lambda item: item["f1"], reverse=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Saved results to {args.output_csv}", flush=True)
    print("Top configs:", flush=True)
    for row in results[: args.top_k]:
        print(
            json.dumps(
                {
                    "f1": row["f1"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "det_conf": row["det_conf"],
                    "det_iou": row["det_iou"],
                    "box_expand_ratio": row["box_expand_ratio"],
                    "cls_min_prob": row["cls_min_prob"],
                    "cls_min_margin": row["cls_min_margin"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
