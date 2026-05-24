#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mix competition classification data with HUST-OBC overlap data for auxiliary training."
    )
    parser.add_argument("--competition-root", type=Path, required=True)
    parser.add_argument("--hust-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hust-train-ratio", type=float, default=0.35)
    parser.add_argument("--include-hust-val", action="store_true")
    parser.add_argument(
        "--competition-val-root",
        type=Path,
        default=None,
        help="Optional alternate competition dataset root used only for val.json while train still comes from --competition-root.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def sample_records(records: list[dict], ratio: float, seed: int) -> list[dict]:
    if ratio >= 1.0:
        return records[:]
    rng = random.Random(seed)
    sampled = [item for item in records if rng.random() < ratio]
    if not sampled:
        sampled = records[: min(len(records), 1)]
    return sampled


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    competition_val_root = args.competition_val_root or args.competition_root

    comp_train = load_records(args.competition_root / "train.json")
    comp_val = load_records(competition_val_root / "val.json")
    hust_train = load_records(args.hust_root / "train.json")
    hust_val = load_records(args.hust_root / "val.json")

    comp_label_map = json.loads((args.competition_root / "label_map.json").read_text(encoding="utf-8"))
    hust_label_map = json.loads((args.hust_root / "label_map.json").read_text(encoding="utf-8"))

    if comp_label_map != hust_label_map:
        raise ValueError("Competition and HUST overlap label maps are not aligned.")

    mixed_hust_train = sample_records(hust_train, ratio=args.hust_train_ratio, seed=args.seed)
    if args.include_hust_val:
        mixed_hust_train.extend(sample_records(hust_val, ratio=args.hust_train_ratio, seed=args.seed + 1))

    train_records = comp_train + mixed_hust_train
    val_records = comp_val
    random.Random(args.seed).shuffle(train_records)

    (args.output / "train.json").write_text(json.dumps(train_records, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "val.json").write_text(json.dumps(val_records, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "label_map.json").write_text(json.dumps(comp_label_map, ensure_ascii=False, indent=2), encoding="utf-8")

    train_counter = Counter(item["text"] for item in train_records)
    summary = {
        "competition_train_samples": len(comp_train),
        "competition_val_samples": len(comp_val),
        "competition_val_root": str(competition_val_root.resolve()),
        "hust_train_samples": len(hust_train),
        "hust_val_samples": len(hust_val),
        "mixed_hust_train_used": len(mixed_hust_train),
        "final_train_samples": len(train_records),
        "final_val_samples": len(val_records),
        "num_classes": len(comp_label_map),
        "hust_train_ratio": args.hust_train_ratio,
        "include_hust_val": args.include_hust_val,
        "top_train_chars": train_counter.most_common(30),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
