#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


@dataclass
class Record:
    path: str
    label: int
    text: str
    source_image: str
    bbox: list[int]
    source_image_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a character classification dataset from competition PNG+XML pairs."
    )
    parser.add_argument("--source", type=Path, required=True, help="Raw competition dataset root.")
    parser.add_argument(
        "--split-root",
        type=Path,
        required=True,
        help="Detection dataset root that already contains images/train and images/val splits.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--min-samples", type=int, default=1, help="Drop classes with fewer than this many train samples.")
    parser.add_argument(
        "--materialize-crops",
        action="store_true",
        help="Save cropped PNG files to disk. By default only manifests are generated and crops are read on the fly.",
    )
    return parser.parse_args()


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def find_image_for_xml(xml_path: Path) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = xml_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def parse_position(raw: str) -> tuple[int, int, int, int] | None:
    raw = raw.strip()
    if not raw:
        return None

    if ";" in raw:
        points = []
        for item in raw.split(";"):
            item = item.strip()
            if not item:
                continue
            parts = [p.strip() for p in item.split(",") if p.strip()]
            if len(parts) != 2:
                return None
            x, y = map(float, parts)
            points.append((x, y))
        if not points:
            return None
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 4:
        return None
    x1, y1, x2, y2 = map(float, parts)
    return int(min(x1, x2)), int(min(y1, y2)), int(max(x1, x2)), int(max(y1, y2))


def load_split_stems(split_dir: Path) -> set[str]:
    return {path.stem for path in split_dir.iterdir() if path.is_file()}


def materialize_crops_by_image(manifests: dict[str, list[Record]]) -> None:
    from PIL import Image

    for split in ("train", "val"):
        records = manifests[split]
        grouped: dict[str, list[Record]] = defaultdict(list)
        for record in records:
            grouped[record.source_image_path].append(record)

        total_images = len(grouped)
        total_crops = len(records)
        written = 0
        for image_index, (source_image_path, image_records) in enumerate(grouped.items(), start=1):
            with Image.open(source_image_path) as image:
                rgb = image.convert("RGB")
                for record in image_records:
                    x1, y1, x2, y2 = record.bbox
                    crop = rgb.crop((x1, y1, x2, y2))
                    Path(record.path).parent.mkdir(parents=True, exist_ok=True)
                    crop.save(record.path)
                    written += 1

            if image_index == 1 or image_index % 200 == 0:
                print(
                    f"Materialized {split}: images={image_index}/{total_images} crops={written}/{total_crops}",
                    flush=True,
                )


def main() -> None:
    args = parse_args()

    train_stems = load_split_stems(args.split_root / "images" / "train")
    val_stems = load_split_stems(args.split_root / "images" / "val")
    split_by_stem = {stem: "train" for stem in train_stems}
    split_by_stem.update({stem: "val" for stem in val_stems})

    ensure_clean_dir(args.output)
    crop_root = args.output / "crops"
    if args.materialize_crops:
        (crop_root / "train").mkdir(parents=True, exist_ok=True)
        (crop_root / "val").mkdir(parents=True, exist_ok=True)

    pending_records: dict[str, list[tuple[str, Path, tuple[int, int, int, int], str]]] = {"train": [], "val": []}
    train_counter: Counter[str] = Counter()
    skipped_dirty = 0
    skipped_missing_split = 0

    xml_paths = sorted(args.source.rglob("*.xml"))
    for index, xml_path in enumerate(xml_paths, start=1):
        split = split_by_stem.get(xml_path.stem)
        if split is None:
            skipped_missing_split += 1
            continue

        image_path = find_image_for_xml(xml_path)
        if image_path is None:
            continue

        try:
            root = ET.fromstring(xml_path.read_text(encoding="utf-16"))
        except Exception:
            continue

        for char_index, elem in enumerate(root.iter("char")):
            text = (elem.text or "").strip()
            position = elem.attrib.get("position", "")
            if len(text) != 1:
                skipped_dirty += 1
                continue
            bbox = parse_position(position)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            if x2 <= x1 or y2 <= y1:
                continue
            pending_records[split].append((text, image_path, bbox, f"{xml_path.stem}_{char_index:04d}.png"))
            if split == "train":
                train_counter[text] += 1

        if index % 1000 == 0:
            print(f"Parsed {index}/{len(xml_paths)} XML files")

    keep_chars = {char for char, count in train_counter.items() if count >= args.min_samples}
    label_map = {char: idx for idx, char in enumerate(sorted(keep_chars))}

    print(f"Raw train chars: {len(train_counter)}")
    print(f"Kept chars: {len(label_map)}")

    manifests: dict[str, list[Record]] = {"train": [], "val": []}
    split_class_counter: dict[str, Counter[str]] = {"train": Counter(), "val": Counter()}

    for split, items in pending_records.items():
        for text, image_path, bbox, crop_name in items:
            if text not in label_map:
                continue
            x1, y1, x2, y2 = bbox
            crop_path = crop_root / split / crop_name if args.materialize_crops else image_path

            record = Record(
                path=str(crop_path.resolve()),
                label=label_map[text],
                text=text,
                source_image=image_path.name,
                bbox=[x1, y1, x2, y2],
                source_image_path=str(image_path.resolve()),
            )
            manifests[split].append(record)
            split_class_counter[split][text] += 1

    if args.materialize_crops:
        materialize_crops_by_image(manifests)

    for split in ("train", "val"):
        payload = [record.__dict__ for record in manifests[split]]
        (args.output / f"{split}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    (args.output / "label_map.json").write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "train_samples": len(manifests["train"]),
        "val_samples": len(manifests["val"]),
        "num_classes": len(label_map),
        "skipped_dirty_labels": skipped_dirty,
        "skipped_missing_split": skipped_missing_split,
        "top_train_chars": split_class_counter["train"].most_common(20),
        "top_val_chars": split_class_counter["val"].most_common(20),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
