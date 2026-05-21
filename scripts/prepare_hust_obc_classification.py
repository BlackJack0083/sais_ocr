#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from sklearn.model_selection import StratifiedShuffleSplit


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare HUST-OBC classification splits.")
    parser.add_argument("--source", type=Path, required=True, help="Path to HUST-OBC deciphered root.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for JSON manifests.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def build_records(source: Path) -> tuple[list[dict], dict[str, int]]:
    class_dirs = sorted([p for p in source.iterdir() if p.is_dir()], key=lambda p: p.name)
    label_map = {class_dir.name: idx for idx, class_dir in enumerate(class_dirs)}
    records: list[dict] = []
    for class_dir in class_dirs:
        label = label_map[class_dir.name]
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES:
                records.append(
                    {
                        "path": str(image_path.resolve()),
                        "label": label,
                        "class_name": class_dir.name,
                    }
                )
    return records, label_map


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

    records, label_map = build_records(args.source)
    train_records, val_records = split_records(records, args.val_ratio, args.seed)

    (args.output / "label_map.json").write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output / "train.json").write_text(
        json.dumps(train_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output / "val.json").write_text(
        json.dumps(val_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Total classes: {len(label_map)}")
    print(f"Total samples: {len(records)}")
    print(f"Train samples: {len(train_records)}")
    print(f"Val samples: {len(val_records)}")
    print(f"Saved manifests to: {args.output}")


if __name__ == "__main__":
    main()
