#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


TRACKING_FIELDNAMES = [
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
    parser = argparse.ArgumentParser(description="Append top sweep rows into experiment_tracking.csv.")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--tracking-csv", type=Path, default=Path("reports/experiment_tracking.csv"))
    parser.add_argument("--date", required=True)
    parser.add_argument("--scope", default="local_val")
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--detector", required=True)
    parser.add_argument("--classifier", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--changes", required=True)
    parser.add_argument("--notes-prefix", required=True)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=TRACKING_FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input_csv)[: args.top_k]
    out_rows: list[dict[str, str]] = []
    for rank, row in enumerate(rows, start=1):
        run_name = f"{args.run_prefix}_rank{rank:02d}"
        out_rows.append(
            {
                "date": args.date,
                "scope": args.scope,
                "run_name": run_name,
                "detector": args.detector,
                "classifier": args.classifier,
                "det_conf": row.get("det_conf", ""),
                "det_iou": row.get("det_iou", ""),
                "score": row.get("f1", ""),
                "tp": row.get("tp", ""),
                "fp": row.get("fp", ""),
                "fn": row.get("fn", ""),
                "precision": row.get("precision", ""),
                "recall": row.get("recall", ""),
                "f1": row.get("f1", ""),
                "iou_threshold": row.get("iou_threshold", ""),
                "num_images": row.get("num_images", ""),
                "total_gt_chars": row.get("total_gt_chars", ""),
                "total_pred_chars": row.get("total_pred_chars", ""),
                "changes": args.changes,
                "notes": (
                    f"{args.notes_prefix}; rank={rank}; "
                    f"box_expand_ratio={row.get('box_expand_ratio', '')}; "
                    f"cls_min_prob={row.get('cls_min_prob', '')}; "
                    f"cls_min_margin={row.get('cls_min_margin', '')}; "
                    f"elapsed_sec={row.get('elapsed_sec', '')}"
                ),
            }
        )
    append_rows(args.tracking_csv, out_rows)


if __name__ == "__main__":
    main()
