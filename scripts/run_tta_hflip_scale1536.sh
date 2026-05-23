#!/usr/bin/env bash
set -euo pipefail

bash scripts/eval_and_record.sh \
  0 \
  logs/eval_tta_hflip_scale1536_20260523.log \
  reports/eval_tta_hflip_scale1536_20260523.json \
  2026-05-23 \
  infer_tta_hflip_scale1536_20260523 \
  local_val \
  "YOLOv8m detector original best (runs/yolo_ancient_chars_yolov8m/weights/best.pt)" \
  "Competition-domain Unicode classifier resumed to epoch 12 (runs/competition_unicode_cls_v2_crops_e12_resume/best.pt)" \
  0.08 \
  0.40 \
  "Detector-only TTA candidate on current best tuned thresholds" \
  "det_tta_mode=hflip+scale1536; slice_size=0; cls_min_prob=0.33; cls_min_margin=0.07" \
  --slice-size 0 \
  --slice-overlap 0 \
  --slice-merge-iou 0.50 \
  --det-tta-mode hflip+scale1536 \
  --iou-threshold 0.5
