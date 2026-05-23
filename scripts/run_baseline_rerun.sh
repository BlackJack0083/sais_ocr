#!/usr/bin/env bash
set -euo pipefail

CUDA_VISIBLE_DEVICES="${1:-1}" uv run python scripts/evaluate_end_to_end.py \
  --images-dir data/processed/yolo_ancient_chars/images/val \
  --detector-weights runs/yolo_ancient_chars_yolov8m/weights/best.pt \
  --classifier-weights runs/competition_unicode_cls_v2_crops_e12_resume/best.pt \
  --source-root data/raw/train \
  --device cuda:0 \
  --det-conf 0.12 \
  --det-iou 0.45 \
  --det-imgsz 1280 \
  --cls-batch-size 128 \
  --cls-min-prob 0.20 \
  --cls-min-margin 0.00 \
  --box-expand-ratio 0.00 \
  --iou-threshold 0.5 > logs/infer_20260522_baseline_rerun.log 2>&1

cat logs/infer_20260522_baseline_rerun.log

uv run python scripts/extract_last_json.py \
  --input logs/infer_20260522_baseline_rerun.log \
  --output reports/baseline_rerun_20260522.json

uv run python scripts/append_experiment_record.py \
  --date 2026-05-22 \
  --scope local_val \
  --run-name baseline_rerun_20260522 \
  --detector "YOLOv8m detector original best (runs/yolo_ancient_chars_yolov8m/weights/best.pt)" \
  --classifier "Competition-domain Unicode classifier resumed to epoch 12 (runs/competition_unicode_cls_v2_crops_e12_resume/best.pt)" \
  --det-conf 0.12 \
  --det-iou 0.45 \
  --changes "Baseline rerun before new inference experiments" \
  --notes "Rerun with current mainline parameters before coarse sweep; BOX_EXPAND_RATIO=0.00; CLASSIFY_MIN_PROB=0.20; CLASSIFY_MIN_MARGIN=0.00" \
  --result-json reports/baseline_rerun_20260522.json
