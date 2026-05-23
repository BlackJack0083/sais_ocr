#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDNAMES = [
    "date",
    "scope",
    "run_name",
    "detector",
    "classifier",
    "det_conf",
    "det_iou",
    "score",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "iou_threshold",
    "num_images",
    "total_gt_chars",
    "total_pred_chars",
    "changes",
    "notes",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append one experiment row to reports/experiment_tracking.csv.")
    parser.add_argument("--csv", type=Path, default=Path("reports/experiment_tracking.csv"))
    parser.add_argument("--date", required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--det-conf", default="")
    parser.add_argument("--det-iou", default="")
    parser.add_argument("--changes", required=True)
    parser.add_argument("--notes", required=True)
    parser.add_argument("--result-json", type=Path, default=None)
    parser.add_argument("--score", default="")
    parser.add_argument("--tp", default="")
    parser.add_argument("--fp", default="")
    parser.add_argument("--fn", default="")
    parser.add_argument("--precision", default="")
    parser.add_argument("--recall", default="")
    parser.add_argument("--f1", default="")
    parser.add_argument("--iou-threshold", default="")
    parser.add_argument("--num-images", default="")
    parser.add_argument("--total-gt-chars", default="")
    parser.add_argument("--total-pred-chars", default="")
    return parser.parse_args()


def load_result(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def append_row(csv_path: Path, row: dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    result = load_result(args.result_json)
    row = {
        "date": args.date,
        "scope": args.scope,
        "run_name": args.run_name,
        "detector": args.detector,
        "classifier": args.classifier,
        "det_conf": args.det_conf,
        "det_iou": args.det_iou,
        "score": result.get("f1", args.score),
        "tp": result.get("tp", args.tp),
        "fp": result.get("fp", args.fp),
        "fn": result.get("fn", args.fn),
        "precision": result.get("precision", args.precision),
        "recall": result.get("recall", args.recall),
        "f1": result.get("f1", args.f1),
        "iou_threshold": result.get("iou_threshold", args.iou_threshold),
        "num_images": result.get("num_images", args.num_images),
        "total_gt_chars": result.get("total_gt_chars", args.total_gt_chars),
        "total_pred_chars": result.get("total_pred_chars", args.total_pred_chars),
        "changes": args.changes,
        "notes": args.notes,
    }
    append_row(args.csv, row)


if __name__ == "__main__":
    main()
