#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Materialize Oracle Bone crops from Chronicles-OCR whose labels overlap with competition classes."
    )
    parser.add_argument("--chronicles-root", type=Path, required=True, help="Chronicles-OCR root directory.")
    parser.add_argument("--jsonl", type=Path, default=None, help="Optional explicit path to Chronicles_OCR.jsonl.")
    parser.add_argument("--competition-label-map", type=Path, required=True, help="Competition label map JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Output directory.")
    parser.add_argument("--font-type", type=str, default="甲骨文")
    parser.add_argument("--min-samples-per-class", type=int, default=2)
    return parser.parse_args()


def resolve_single_char(raw: str) -> str | None:
    value = raw.strip()
    if len(value) == 1:
        return value
    return None


def clip_bbox(bbox: list[int], width: int, height: int) -> tuple[int, int, int, int] | None:
    if len(bbox) != 4:
        return None
    x, y, w, h = [int(v) for v in bbox]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + max(0, w))
    y2 = min(height, y + max(0, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def resolve_image_path(chronicles_root: Path, image_rel: str) -> Path | None:
    candidates = [
        chronicles_root / image_rel,
        chronicles_root / image_rel.replace("images/", "", 1),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    crop_root = args.output / "crops"
    crop_root.mkdir(parents=True, exist_ok=True)

    jsonl_path = args.jsonl or (args.chronicles_root / "data" / "Chronicles_OCR.jsonl")
    competition_label_map = json.loads(args.competition_label_map.read_text(encoding="utf-8"))
    competition_chars = set(competition_label_map.keys())

    raw_records: list[dict] = []
    skipped_non_target_font = 0
    skipped_dirty_label = 0
    skipped_not_overlap = 0
    skipped_bad_bbox = 0
    char_counter: Counter[str] = Counter()

    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            sample = json.loads(line)
            if sample.get("font_type") != args.font_type:
                skipped_non_target_font += 1
                continue
            image_rel = sample.get("image_path", "")
            image_path = resolve_image_path(args.chronicles_root, image_rel)
            if image_path is None:
                continue

            for spot_index, spot in enumerate(sample.get("spotting", [])):
                text = resolve_single_char(str(spot.get("modern_char", "")))
                if text is None:
                    skipped_dirty_label += 1
                    continue
                if text not in competition_chars:
                    skipped_not_overlap += 1
                    continue
                raw_records.append(
                    {
                        "image_path": image_path,
                        "image_rel": image_rel,
                        "bbox": spot.get("bbox", []),
                        "text": text,
                        "label": competition_label_map[text],
                        "sample_index": line_index,
                        "spot_index": spot_index,
                    }
                )
                char_counter[text] += 1

    keep_chars = {char for char, count in char_counter.items() if count >= args.min_samples_per_class}
    filtered_records = [item for item in raw_records if item["text"] in keep_chars]

    materialized: list[dict] = []
    current_image_path: Path | None = None
    current_image = None
    try:
        for item in filtered_records:
            image_path = item["image_path"]
            if current_image_path != image_path:
                if current_image is not None:
                    current_image.close()
                current_image = Image.open(image_path).convert("RGB")
                current_image_path = image_path
            assert current_image is not None
            bbox = clip_bbox(item["bbox"], *current_image.size)
            if bbox is None:
                skipped_bad_bbox += 1
                continue
            x1, y1, x2, y2 = bbox
            crop = current_image.crop((x1, y1, x2, y2))
            crop_name = f"{image_path.stem}_{item['spot_index']:04d}.png"
            crop_path = crop_root / crop_name
            crop.save(crop_path)
            materialized.append(
                {
                    "path": str(crop_path.resolve()),
                    "label": item["label"],
                    "text": item["text"],
                    "source": "Chronicles-OCR",
                    "source_image": image_path.name,
                    "source_image_path": str(image_path.resolve()),
                    "bbox": [x1, y1, x2, y2],
                }
            )
    finally:
        if current_image is not None:
            current_image.close()

    (args.output / "train.json").write_text(
        json.dumps(materialized, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (args.output / "val.json").write_text("[]\n", encoding="utf-8")
    (args.output / "label_map.json").write_text(
        json.dumps(competition_label_map, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "font_type": args.font_type,
        "raw_records_total": len(raw_records),
        "kept_classes_total": len(keep_chars),
        "materialized_samples": len(materialized),
        "skipped_non_target_font": skipped_non_target_font,
        "skipped_dirty_label": skipped_dirty_label,
        "skipped_not_overlap": skipped_not_overlap,
        "skipped_bad_bbox": skipped_bad_bbox,
        "top_chars": Counter(item["text"] for item in materialized).most_common(30),
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
