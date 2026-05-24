#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild a classifier split from competition crops with per-class stratified validation coverage."
    )
    parser.add_argument("--base-dataset", type=Path, required=True, help="Existing competition classifier dataset root.")
    parser.add_argument("--output", type=Path, required=True, help="Output dataset root.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--min-val-per-class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-mode",
        type=str,
        default="record",
        choices=["record", "image_group"],
        help="Split by individual crop records or by source image groups to avoid same-image leakage.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def source_group_key(record: dict) -> str:
    source = record.get("source_image_path") or record.get("path") or ""
    return str(source)


def split_by_image_group(
    all_records: list[dict],
    val_ratio: float,
    seed: int,
    min_val_per_class: int,
) -> tuple[list[dict], list[dict], dict[str, dict[str, int]]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in all_records:
        groups[source_group_key(record)].append(record)

    class_totals = Counter(record["text"] for record in all_records)
    remaining_train = class_totals.copy()
    class_summary = {text: {"total": int(total), "train": int(total), "val": 0} for text, total in class_totals.items()}

    group_items = list(groups.items())
    rng.shuffle(group_items)

    group_meta: list[dict] = []
    for key, records in group_items:
        class_counts = Counter(record["text"] for record in records)
        group_meta.append(
            {
                "key": key,
                "records": records,
                "class_counts": class_counts,
                "size": len(records),
                "classes": set(class_counts.keys()),
            }
        )

    target_val_samples = int(round(len(all_records) * val_ratio))
    val_groups: list[dict] = []
    covered_classes: set[str] = set()

    def can_move(meta: dict) -> bool:
        for text, count in meta["class_counts"].items():
            if remaining_train[text] - count < 1:
                return False
        return True

    def move_group_to_val(meta: dict) -> None:
        val_groups.append(meta)
        for text, count in meta["class_counts"].items():
            remaining_train[text] -= count
            covered_classes.add(text)
            class_summary[text]["train"] -= int(count)
            class_summary[text]["val"] += int(count)

    # Phase 1: greedily cover as many classes as possible with whole-image groups.
    while True:
        best_meta = None
        best_score = None
        for meta in group_meta:
            if meta in val_groups or not can_move(meta):
                continue
            uncovered = sum(1 for text in meta["classes"] if text not in covered_classes)
            if uncovered <= 0:
                continue
            rarity = sum(1.0 / class_totals[text] for text in meta["classes"] if text not in covered_classes)
            score = (uncovered, rarity, -meta["size"])
            if best_score is None or score > best_score:
                best_score = score
                best_meta = meta
        if best_meta is None:
            break
        move_group_to_val(best_meta)
        if sum(group["size"] for group in val_groups) >= target_val_samples and len(covered_classes) >= min_val_per_class:
            break

    # Phase 2: fill val toward target size with the smallest safe groups.
    for meta in sorted(group_meta, key=lambda item: (item["size"], len(item["classes"]))):
        if meta in val_groups or not can_move(meta):
            continue
        if sum(group["size"] for group in val_groups) >= target_val_samples:
            break
        move_group_to_val(meta)

    val_group_keys = {meta["key"] for meta in val_groups}
    new_val = [record for key, records in groups.items() if key in val_group_keys for record in records]
    new_train = [record for key, records in groups.items() if key not in val_group_keys for record in records]

    rng.shuffle(new_train)
    rng.shuffle(new_val)
    return new_train, new_val, class_summary


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)

    train_records = load_records(args.base_dataset / "train.json")
    val_records = load_records(args.base_dataset / "val.json")
    label_map = json.loads((args.base_dataset / "label_map.json").read_text(encoding="utf-8"))

    all_records = train_records + val_records
    new_train: list[dict] = []
    new_val: list[dict] = []
    class_summary: dict[str, dict[str, int]] = {}

    if args.split_mode == "record":
        by_class: dict[str, list[dict]] = defaultdict(list)
        for record in all_records:
            by_class[record["text"]].append(record)

        for text, records in by_class.items():
            shuffled = records[:]
            rng.shuffle(shuffled)
            total = len(shuffled)
            desired_val = max(args.min_val_per_class, int(round(total * args.val_ratio)))
            if total == 1:
                desired_val = 0
            elif desired_val >= total:
                desired_val = total - 1

            val_part = shuffled[:desired_val]
            train_part = shuffled[desired_val:]
            new_val.extend(val_part)
            new_train.extend(train_part)
            class_summary[text] = {
                "total": total,
                "train": len(train_part),
                "val": len(val_part),
            }
    else:
        new_train, new_val, class_summary = split_by_image_group(
            all_records=all_records,
            val_ratio=args.val_ratio,
            seed=args.seed,
            min_val_per_class=args.min_val_per_class,
        )

    (args.output / "train.json").write_text(json.dumps(new_train, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "val.json").write_text(json.dumps(new_val, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "label_map.json").write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {
        "base_dataset": str(args.base_dataset.resolve()),
        "num_classes": len(label_map),
        "train_samples": len(new_train),
        "val_samples": len(new_val),
        "min_val_per_class": args.min_val_per_class,
        "val_ratio": args.val_ratio,
        "split_mode": args.split_mode,
        "classes_with_val_samples": sum(1 for item in class_summary.values() if item["val"] > 0),
        "classes_without_val_samples": sum(1 for item in class_summary.values() if item["val"] == 0),
        "classes_with_single_total_sample": sum(1 for item in class_summary.values() if item["total"] == 1),
        "shared_source_images": len({source_group_key(item) for item in new_train} & {source_group_key(item) for item in new_val}),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "class_summary.json").write_text(json.dumps(class_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
