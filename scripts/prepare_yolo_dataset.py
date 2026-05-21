#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff")


@dataclass
class Sample:
    image_path: Path
    label_lines: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert XML character annotations into a YOLO detection dataset."
    )
    parser.add_argument("--source", type=Path, required=True, help="Raw extracted dataset root.")
    parser.add_argument("--output", type=Path, required=True, help="Output YOLO dataset root.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlink.")
    return parser.parse_args()


def parse_position(raw: str) -> tuple[float, float, float, float] | None:
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
        return min(xs), min(ys), max(xs), max(ys)

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) == 4:
        x1, y1, x2, y2 = map(float, parts)
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    return None


def clamp_bbox(
    bbox: tuple[float, float, float, float], width: float, height: float
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = bbox
    x1 = min(max(x1, 0.0), width)
    y1 = min(max(y1, 0.0), height)
    x2 = min(max(x2, 0.0), width)
    y2 = min(max(y2, 0.0), height)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def bbox_to_yolo(bbox: tuple[float, float, float, float], width: float, height: float) -> str:
    x1, y1, x2, y2 = bbox
    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height
    return f"0 {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"


def find_image_for_xml(xml_path: Path) -> Path | None:
    stem = xml_path.stem
    for suffix in IMAGE_SUFFIXES:
        candidate = xml_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    for suffix in IMAGE_SUFFIXES:
        candidate = xml_path.parent / f"{stem}{suffix.upper()}"
        if candidate.exists():
            return candidate
    return None


def parse_xml(xml_path: Path) -> Sample | None:
    image_path = find_image_for_xml(xml_path)
    if image_path is None:
        return None

    text = xml_path.read_text(encoding="utf-16")
    root = ET.fromstring(text)

    width = float(root.attrib["width"])
    height = float(root.attrib["height"])
    label_lines: list[str] = []

    for char_elem in root.iter("char"):
        text_value = (char_elem.text or "").strip()
        # Keep only labels that decode to a single Unicode code point.
        if len(text_value) != 1:
            continue
        position = char_elem.attrib.get("position", "")
        bbox = parse_position(position)
        if bbox is None:
            continue
        bbox = clamp_bbox(bbox, width, height)
        if bbox is None:
            continue
        label_lines.append(bbox_to_yolo(bbox, width, height))

    if not label_lines:
        return None

    return Sample(image_path=image_path, label_lines=label_lines)


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, copy_images: bool) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy_images:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def write_dataset_yaml(output_root: Path) -> None:
    content = "\n".join(
        [
            f"path: {output_root.resolve()}",
            "train: images/train",
            "val: images/val",
            "",
            "names:",
            "  0: youzi",
            "",
        ]
    )
    (output_root / "dataset.yaml").write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    source_root = args.source
    output_root = args.output

    xml_paths = sorted(source_root.rglob("*.xml"))
    samples: list[Sample] = []
    skipped_missing_image = 0
    skipped_empty = 0

    for index, xml_path in enumerate(xml_paths, start=1):
        sample = parse_xml(xml_path)
        if sample is None:
            if find_image_for_xml(xml_path) is None:
                skipped_missing_image += 1
            else:
                skipped_empty += 1
            continue
        samples.append(sample)
        if index % 1000 == 0:
            print(f"Parsed {index}/{len(xml_paths)} XML files")

    random.Random(args.seed).shuffle(samples)
    val_count = int(len(samples) * args.val_ratio)
    val_samples = samples[:val_count]
    train_samples = samples[val_count:]

    for split in ("train", "val"):
        ensure_clean_dir(output_root / "images" / split)
        ensure_clean_dir(output_root / "labels" / split)

    def materialize(split: str, split_samples: list[Sample]) -> None:
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        for sample in split_samples:
            image_target = image_dir / sample.image_path.name
            label_target = label_dir / f"{sample.image_path.stem}.txt"
            link_or_copy(sample.image_path, image_target, args.copy_images)
            label_target.write_text("\n".join(sample.label_lines) + "\n", encoding="utf-8")

    materialize("train", train_samples)
    materialize("val", val_samples)
    write_dataset_yaml(output_root)

    print(f"Valid samples: {len(samples)}")
    print(f"Train samples: {len(train_samples)}")
    print(f"Val samples: {len(val_samples)}")
    print(f"Skipped XML without paired image: {skipped_missing_image}")
    print(f"Skipped XML without valid boxes: {skipped_empty}")
    print(f"Dataset YAML: {output_root / 'dataset.yaml'}")


if __name__ == "__main__":
    main()
