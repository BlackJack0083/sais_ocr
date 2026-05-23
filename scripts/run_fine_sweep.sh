#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 <cuda_visible_devices> <det_conf> <det_iou> <box_expand> <cls_prob_values> <cls_margin_values> <output_csv> <done_file>" >&2
  exit 1
fi

cuda_visible_devices="$1"
det_conf="$2"
det_iou="$3"
box_expand="$4"
cls_prob_values="$5"
cls_margin_values="$6"
output_csv="$7"
done_file="$8"

CUDA_VISIBLE_DEVICES="$cuda_visible_devices" uv run python scripts/sweep_end_to_end.py \
  --images-dir data/processed/yolo_ancient_chars/images/val \
  --detector-weights runs/yolo_ancient_chars_yolov8m/weights/best.pt \
  --classifier-weights runs/competition_unicode_cls_v2_crops_e12_resume/best.pt \
  --source-root data/raw/train \
  --device cuda:0 \
  --det-conf-values "$det_conf" \
  --det-iou-values "$det_iou" \
  --box-expand-values "$box_expand" \
  --cls-min-prob-values "$cls_prob_values" \
  --cls-min-margin-values "$cls_margin_values" \
  --det-imgsz 1280 \
  --cls-batch-size 128 \
  --output-csv "$output_csv"

touch "$done_file"
