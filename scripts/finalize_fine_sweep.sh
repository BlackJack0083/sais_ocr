#!/usr/bin/env bash
set -euo pipefail

while [[ ! -f reports/sweep_20260522_fine.done ]]; do
  sleep 60
done

uv run python scripts/promote_sweep_results.py \
  --input-csv reports/sweep_20260522_fine.csv \
  --date 2026-05-22 \
  --run-prefix infer_fine_20260522 \
  --detector "YOLOv8m detector original best (runs/yolo_ancient_chars_yolov8m/weights/best.pt)" \
  --classifier "Competition-domain Unicode classifier resumed to epoch 12 (runs/competition_unicode_cls_v2_crops_e12_resume/best.pt)" \
  --top-k 10 \
  --changes "Inference fine sweep around the best coarse configuration" \
  --notes-prefix "Fine sweep over cls_min_prob/cls_min_margin around the best coarse point"
