#!/usr/bin/env bash
set -euo pipefail

bash scripts/eval_and_record.sh \
  0 \
  logs/eval_slice1536_20260523.log \
  reports/eval_slice1536_20260523.json \
  2026-05-23 \
  infer_slice1536_20260523 \
  local_val \
  "YOLOv8m detector original best (runs/yolo_ancient_chars_yolov8m/weights/best.pt)" \
  "Competition-domain Unicode classifier resumed to epoch 12 (runs/competition_unicode_cls_v2_crops_e12_resume/best.pt)" \
  0.08 \
  0.40 \
  "Sliced inference candidate on current best tuned thresholds" \
  "slice_size=1536; slice_overlap=256; slice_merge_iou=0.50; cls_min_prob=0.33; cls_min_margin=0.07; det_tta_mode=none" \
  --slice-size 1536 \
  --slice-overlap 256 \
  --slice-merge-iou 0.50 \
  --det-tta-mode none \
  --iou-threshold 0.5
