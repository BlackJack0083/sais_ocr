#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 <cuda_visible_devices> <tag> <slice_size> <slice_overlap> <log_file> <result_json> <notes>" >&2
  exit 1
fi

cuda_visible_devices="$1"
tag="$2"
slice_size="$3"
slice_overlap="$4"
log_file="$5"
result_json="$6"
notes="$7"

bash scripts/eval_and_record.sh \
  "$cuda_visible_devices" \
  "$log_file" \
  "$result_json" \
  2026-05-23 \
  "$tag" \
  local_val \
  "YOLOv8m detector original best (runs/yolo_ancient_chars_yolov8m/weights/best.pt)" \
  "Competition-domain Unicode classifier resumed to epoch 12 (runs/competition_unicode_cls_v2_crops_e12_resume/best.pt)" \
  0.08 \
  0.40 \
  "Sliced inference candidate on current best tuned thresholds" \
  "$notes" \
  --slice-size "$slice_size" \
  --slice-overlap "$slice_overlap" \
  --slice-merge-iou 0.50 \
  --det-tta-mode none \
  --iou-threshold 0.5
