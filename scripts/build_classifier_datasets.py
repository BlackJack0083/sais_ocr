#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cleaned classifier datasets and split variants.")
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--hust-root", type=Path, required=True)
    parser.add_argument("--chronicles-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--hust-target-per-class", type=int, default=50)
    parser.add_argument("--hust-max-per-class", type=int, default=50)
    parser.add_argument("--chronicles-target-per-class", type=int, default=50)
    parser.add_argument("--chronicles-max-per-class", type=int, default=50)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_dataset(root: Path, train_records: list[dict], val_records: list[dict], label_map: dict[str, int], summary: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "train.json").write_text(json.dumps(train_records, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "val.json").write_text(json.dumps(val_records, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "label_map.json").write_text(json.dumps(label_map, ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def source_group_key(record: dict) -> str:
    return str(record.get("source_image_path") or record.get("source_image") or record.get("path") or "")


def split_group(records: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[source_group_key(record)].append(record)

    class_totals = Counter(record["text"] for record in records)
    remaining_train = class_totals.copy()
    group_items = list(groups.items())
    rng.shuffle(group_items)

    metas: list[dict] = []
    for key, group_records in group_items:
        class_counts = Counter(record["text"] for record in group_records)
        metas.append(
            {
                "key": key,
                "records": group_records,
                "class_counts": class_counts,
                "classes": set(class_counts),
                "size": len(group_records),
            }
        )

    target_val_samples = int(round(len(records) * val_ratio))
    val_groups: list[dict] = []
    covered_classes: set[str] = set()

    def can_move(meta: dict) -> bool:
        for text, count in meta["class_counts"].items():
            if remaining_train[text] - count < 1:
                return False
        return True

    def move(meta: dict) -> None:
        val_groups.append(meta)
        for text, count in meta["class_counts"].items():
            remaining_train[text] -= count
            covered_classes.add(text)

    while True:
        best_meta = None
        best_score = None
        for meta in metas:
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
        move(best_meta)
        if sum(item["size"] for item in val_groups) >= target_val_samples:
            break

    for meta in sorted(metas, key=lambda item: (item["size"], len(item["classes"]))):
        if meta in val_groups or not can_move(meta):
            continue
        if sum(item["size"] for item in val_groups) >= target_val_samples:
            break
        move(meta)

    val_group_keys = {meta["key"] for meta in val_groups}
    val_records = [record for key, group_records in groups.items() if key in val_group_keys for record in group_records]
    train_records = [record for key, group_records in groups.items() if key not in val_group_keys for record in group_records]
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def split_random(records: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_class[record["text"]].append(record)

    train_records: list[dict] = []
    val_records: list[dict] = []
    for class_records in by_class.values():
        shuffled = class_records[:]
        rng.shuffle(shuffled)
        total = len(shuffled)
        desired_val = int(round(total * val_ratio))
        if total == 1:
            desired_val = 0
        elif desired_val <= 0:
            desired_val = 1
        elif desired_val >= total:
            desired_val = total - 1
        val_records.extend(shuffled[:desired_val])
        train_records.extend(shuffled[desired_val:])
    rng.shuffle(train_records)
    rng.shuffle(val_records)
    return train_records, val_records


def supplement_to_target(
    base_train: list[dict],
    supplement_records: list[dict],
    target_per_class: int,
    max_per_class: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    base_counts = Counter(item["text"] for item in base_train)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for record in supplement_records:
        by_class[record["text"]].append(record)

    selected: list[dict] = []
    for text, class_records in sorted(by_class.items()):
        deficit = max(0, target_per_class - base_counts.get(text, 0))
        if deficit <= 0:
            continue
        shuffled = class_records[:]
        rng.shuffle(shuffled)
        take = min(deficit, max_per_class, len(shuffled))
        selected.extend(shuffled[:take])
    rng.shuffle(selected)
    return selected


def build_mix(
    competition_train: list[dict],
    competition_val: list[dict],
    label_map: dict[str, int],
    output_root: Path,
    name: str,
    hust_train: list[dict] | None,
    chronicles_train: list[dict] | None,
    meta: dict,
    seed: int,
) -> None:
    train_records = competition_train[:]
    if hust_train:
        train_records.extend(hust_train)
    if chronicles_train:
        train_records.extend(chronicles_train)
    random.Random(seed).shuffle(train_records)
    summary = {
        "name": name,
        "competition_train_samples": len(competition_train),
        "competition_val_samples": len(competition_val),
        "hust_train_used": len(hust_train or []),
        "chronicles_train_used": len(chronicles_train or []),
        "final_train_samples": len(train_records),
        "final_val_samples": len(competition_val),
        "num_classes": len(label_map),
        **meta,
    }
    save_dataset(output_root / name, train_records, competition_val, label_map, summary)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    competition_label_map = json.loads((args.competition_root / "label_map.json").read_text(encoding="utf-8"))
    competition_records = load_records(args.competition_root / "train.json") + load_records(args.competition_root / "val.json")
    hust_train_all = load_records(args.hust_root / "train.json")
    chronicles_train_all = load_records(args.chronicles_root / "train.json")

    group_train, group_val = split_group(competition_records, val_ratio=args.val_ratio, seed=args.seed)
    random_train, random_val = split_random(competition_records, val_ratio=args.val_ratio, seed=args.seed)
    full_train = competition_records[:]
    random.Random(args.seed).shuffle(full_train)

    splits_root = args.output_root / "splits"
    save_dataset(
        splits_root / "competition_group_val",
        group_train,
        group_val,
        competition_label_map,
        {
            "split_mode": "image_group",
            "train_samples": len(group_train),
            "val_samples": len(group_val),
            "num_classes": len(competition_label_map),
        },
    )
    save_dataset(
        splits_root / "competition_random_val",
        random_train,
        random_val,
        competition_label_map,
        {
            "split_mode": "random_record",
            "train_samples": len(random_train),
            "val_samples": len(random_val),
            "num_classes": len(competition_label_map),
        },
    )
    save_dataset(
        splits_root / "competition_fulltrain",
        full_train,
        [],
        competition_label_map,
        {
            "split_mode": "fulltrain",
            "train_samples": len(full_train),
            "val_samples": 0,
            "num_classes": len(competition_label_map),
        },
    )

    hust_group = supplement_to_target(
        group_train,
        hust_train_all,
        target_per_class=args.hust_target_per_class,
        max_per_class=args.hust_max_per_class,
        seed=args.seed,
    )
    hust_random = supplement_to_target(
        random_train,
        hust_train_all,
        target_per_class=args.hust_target_per_class,
        max_per_class=args.hust_max_per_class,
        seed=args.seed,
    )
    hust_full = supplement_to_target(
        full_train,
        hust_train_all,
        target_per_class=args.hust_target_per_class,
        max_per_class=args.hust_max_per_class,
        seed=args.seed,
    )

    chron_group = supplement_to_target(
        group_train + hust_group,
        chronicles_train_all,
        target_per_class=args.chronicles_target_per_class,
        max_per_class=args.chronicles_max_per_class,
        seed=args.seed,
    )
    chron_random = supplement_to_target(
        random_train + hust_random,
        chronicles_train_all,
        target_per_class=args.chronicles_target_per_class,
        max_per_class=args.chronicles_max_per_class,
        seed=args.seed,
    )
    chron_full = supplement_to_target(
        full_train + hust_full,
        chronicles_train_all,
        target_per_class=args.chronicles_target_per_class,
        max_per_class=args.chronicles_max_per_class,
        seed=args.seed,
    )

    mixes_root = args.output_root / "mixes"
    build_mix(
        group_train,
        group_val,
        competition_label_map,
        mixes_root,
        "group_only",
        None,
        None,
        {"base_split": "competition_group_val"},
        args.seed,
    )
    build_mix(
        group_train,
        group_val,
        competition_label_map,
        mixes_root,
        "group_hust",
        hust_group,
        None,
        {
            "base_split": "competition_group_val",
            "hust_target_per_class": args.hust_target_per_class,
            "hust_max_per_class": args.hust_max_per_class,
        },
        args.seed,
    )
    build_mix(
        group_train,
        group_val,
        competition_label_map,
        mixes_root,
        "group_hust_chron",
        hust_group,
        chron_group,
        {
            "base_split": "competition_group_val",
            "hust_target_per_class": args.hust_target_per_class,
            "hust_max_per_class": args.hust_max_per_class,
            "chronicles_target_per_class": args.chronicles_target_per_class,
            "chronicles_max_per_class": args.chronicles_max_per_class,
        },
        args.seed,
    )
    build_mix(
        random_train,
        random_val,
        competition_label_map,
        mixes_root,
        "random_only",
        None,
        None,
        {"base_split": "competition_random_val"},
        args.seed,
    )
    build_mix(
        random_train,
        random_val,
        competition_label_map,
        mixes_root,
        "random_hust",
        hust_random,
        None,
        {
            "base_split": "competition_random_val",
            "hust_target_per_class": args.hust_target_per_class,
            "hust_max_per_class": args.hust_max_per_class,
        },
        args.seed,
    )
    build_mix(
        random_train,
        random_val,
        competition_label_map,
        mixes_root,
        "random_hust_chron",
        hust_random,
        chron_random,
        {
            "base_split": "competition_random_val",
            "hust_target_per_class": args.hust_target_per_class,
            "hust_max_per_class": args.hust_max_per_class,
            "chronicles_target_per_class": args.chronicles_target_per_class,
            "chronicles_max_per_class": args.chronicles_max_per_class,
        },
        args.seed,
    )
    build_mix(
        full_train,
        [],
        competition_label_map,
        mixes_root,
        "fulltrain_only",
        None,
        None,
        {"base_split": "competition_fulltrain"},
        args.seed,
    )
    build_mix(
        full_train,
        [],
        competition_label_map,
        mixes_root,
        "fulltrain_hust",
        hust_full,
        None,
        {
            "base_split": "competition_fulltrain",
            "hust_target_per_class": args.hust_target_per_class,
            "hust_max_per_class": args.hust_max_per_class,
        },
        args.seed,
    )
    build_mix(
        full_train,
        [],
        competition_label_map,
        mixes_root,
        "fulltrain_hust_chron",
        hust_full,
        chron_full,
        {
            "base_split": "competition_fulltrain",
            "hust_target_per_class": args.hust_target_per_class,
            "hust_max_per_class": args.hust_max_per_class,
            "chronicles_target_per_class": args.chronicles_target_per_class,
            "chronicles_max_per_class": args.chronicles_max_per_class,
        },
        args.seed,
    )

    manifest = {
        "competition_total_samples": len(competition_records),
        "competition_num_classes": len(competition_label_map),
        "hust_train_samples_total": len(hust_train_all),
        "chronicles_train_samples_total": len(chronicles_train_all),
        "output_root": str(args.output_root.resolve()),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
