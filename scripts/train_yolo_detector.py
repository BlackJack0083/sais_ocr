#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_batch(raw: str) -> int | float:
    value = float(raw)
    if value.is_integer():
        return int(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO detector for ancient-character detection.")
    parser.add_argument("--model", type=Path, default=Path("yolov8m.pt"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path("runs"))
    parser.add_argument("--name", type=str, default="yolo_ancient_chars_yolov8m_aug1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument(
        "--batch",
        type=parse_batch,
        default=8,
        help="Batch size. Integer for fixed batch, -1 for Ultralytics auto batch (~60% VRAM), or 0.xx for target VRAM fraction.",
    )
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--close-mosaic", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model))
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        seed=args.seed,
        deterministic=True,
        patience=100,
        optimizer="auto",
        amp=True,
        save=True,
        val=True,
        plots=True,
        cache=False,
        single_cls=False,
        rect=False,
        cos_lr=False,
        resume=args.resume,
        close_mosaic=args.close_mosaic,
        degrees=7.0,
        translate=0.05,
        scale=0.2,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        copy_paste=0.0,
        hsv_h=0.01,
        hsv_s=0.15,
        hsv_v=0.15,
        erasing=0.0,
        auto_augment=None,
    )


if __name__ == "__main__":
    main()
