#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 <cuda_visible_devices> <det_conf_values> <output_csv> <done_file>" >&2
  exit 1
fi

cuda_visible_devices="$1"
det_conf_values="$2"
output_csv="$3"
done_file="$4"

CUDA_VISIBLE_DEVICES="$cuda_visible_devices" uv run python scripts/sweep_end_to_end.py \
  --images-dir data/processed/yolo_ancient_chars/images/val \
  --detector-weights runs/yolo_ancient_chars_yolov8m/weights/best.pt \
  --classifier-weights runs/competition_unicode_cls_v2_crops_e12_resume/best.pt \
  --source-root data/raw/train \
  --device cuda:0 \
  --det-conf-values "$det_conf_values" \
  --det-iou-values 0.40,0.45,0.50 \
  --box-expand-values 0.00,0.01 \
  --cls-min-prob-values 0.15,0.20,0.25,0.30 \
  --cls-min-margin-values 0.00,0.02,0.05 \
  --det-imgsz 1280 \
  --cls-batch-size 128 \
  --output-csv "$output_csv"

touch "$done_file"
