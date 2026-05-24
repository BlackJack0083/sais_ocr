#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare HUST-OBC subset whose mapped characters overlap with competition labels."
    )
    parser.add_argument("--hust-json-root", type=Path, required=True, help="Existing processed HUST-OBC json root.")
    parser.add_argument("--hust-id-to-char", type=Path, required=True, help="Path to HUST ID_to_chinese.json.")
    parser.add_argument("--competition-label-map", type=Path, required=True, help="Competition label_map.json.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-samples-per-class", type=int, default=2)
    return parser.parse_args()


def resolve_single_char(raw: str) -> str | None:
    value = raw.strip()
    if len(value) == 1:
        return value
    for sep in ("；", ";", "、", "/", ","):
        if sep in value:
            parts = [part.strip() for part in value.split(sep) if part.strip()]
            singles = [part for part in parts if len(part) == 1]
            if len(singles) == 1:
                return singles[0]
    return None


def load_records(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def split_records(records: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    labels = [record["label"] for record in records]
    counts = Counter(labels)

    train_fixed = [record for record in records if counts[record["label"]] == 1]
    remaining = [record for record in records if counts[record["label"]] > 1]

    if not remaining:
        return train_fixed, []

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    indices = list(range(len(remaining)))
    train_idx, val_idx = next(splitter.split(indices, [record["label"] for record in remaining]))

    train_records = train_fixed + [remaining[idx] for idx in train_idx]
    val_records = [remaining[idx] for idx in val_idx]

    random.Random(seed).shuffle(train_records)
    random.Random(seed).shuffle(val_records)
    return train_records, val_records


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    hust_train = load_records(args.hust_json_root / "train.json")
    hust_val = load_records(args.hust_json_root / "val.json")
    hust_all = hust_train + hust_val

    hust_id_to_char = json.loads(args.hust_id_to_char.read_text(encoding="utf-8"))
    competition_label_map = json.loads(args.competition_label_map.read_text(encoding="utf-8"))
    competition_chars = set(competition_label_map.keys())

    mapped_records: list[dict] = []
    skipped_multi_map = 0
    skipped_not_overlap = 0

    for record in hust_all:
        class_id = record["class_name"]
        mapped_char = resolve_single_char(str(hust_id_to_char.get(class_id, "")))
        if mapped_char is None:
            skipped_multi_map += 1
            continue
        if mapped_char not in competition_chars:
            skipped_not_overlap += 1
            continue
        mapped_records.append(
            {
                "path": record["path"],
                "label": competition_label_map[mapped_char],
                "text": mapped_char,
                "source": "HUST-OBC",
                "hust_class_id": class_id,
            }
        )

    class_counts = Counter(item["text"] for item in mapped_records)
    kept_chars = {char for char, count in class_counts.items() if count >= args.min_samples_per_class}
    filtered_records = [item for item in mapped_records if item["text"] in kept_chars]

    train_records, val_records = split_records(filtered_records, val_ratio=args.val_ratio, seed=args.seed)

    summary = {
        "mapped_records_total": len(mapped_records),
        "filtered_records_total": len(filtered_records),
        "overlap_classes_total": len(set(item["text"] for item in mapped_records)),
        "kept_classes_total": len(kept_chars),
        "train_samples": len(train_records),
        "val_samples": len(val_records),
        "skipped_multi_or_non_single_mapping": skipped_multi_map,
        "skipped_not_overlap": skipped_not_overlap,
        "top_overlap_chars": Counter(item["text"] for item in filtered_records).most_common(30),
    }

    (args.output / "train.json").write_text(json.dumps(train_records, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "val.json").write_text(json.dumps(val_records, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # This label map stays aligned to competition labels to simplify mixed training.
    (args.output / "label_map.json").write_text(
        json.dumps(competition_label_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
