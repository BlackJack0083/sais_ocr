#!/usr/bin/env bash
set -euo pipefail

while [[ ! -f reports/sweep_20260522_coarse_a.done || ! -f reports/sweep_20260522_coarse_b.done ]]; do
  sleep 60
done

uv run python scripts/merge_sweep_csvs.py \
  --inputs reports/sweep_20260522_coarse_a.csv reports/sweep_20260522_coarse_b.csv \
  --output reports/sweep_20260522_coarse_merged.csv

uv run python scripts/promote_sweep_results.py \
  --input-csv reports/sweep_20260522_coarse_merged.csv \
  --date 2026-05-22 \
  --run-prefix infer_coarse_20260522 \
  --detector "YOLOv8m detector original best (runs/yolo_ancient_chars_yolov8m/weights/best.pt)" \
  --classifier "Competition-domain Unicode classifier resumed to epoch 12 (runs/competition_unicode_cls_v2_crops_e12_resume/best.pt)" \
  --top-k 10 \
  --changes "Inference coarse sweep on current best weights" \
  --notes-prefix "Coarse sweep over det_conf/det_iou/box_expand_ratio/cls_min_prob/cls_min_margin"
