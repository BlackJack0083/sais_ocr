#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 13 ]]; then
  echo "usage: $0 <cuda_visible_devices> <log_file> <result_json> <date> <run_name> <scope> <detector_desc> <classifier_desc> <det_conf> <det_iou> <changes> <notes> <extra_eval_args...>" >&2
  exit 1
fi

cuda_visible_devices="$1"
log_file="$2"
result_json="$3"
date_str="$4"
run_name="$5"
scope="$6"
detector_desc="$7"
classifier_desc="$8"
det_conf="$9"
det_iou="${10}"
changes="${11}"
notes="${12}"
shift 12

CUDA_VISIBLE_DEVICES="$cuda_visible_devices" uv run python scripts/evaluate_end_to_end.py \
  --images-dir data/processed/yolo_ancient_chars/images/val \
  --detector-weights runs/yolo_ancient_chars_yolov8m/weights/best.pt \
  --classifier-weights runs/competition_unicode_cls_v2_crops_e12_resume/best.pt \
  --source-root data/raw/train \
  --device cuda:0 \
  --det-conf "$det_conf" \
  --det-iou "$det_iou" \
  --det-imgsz 1280 \
  --cls-batch-size 128 \
  --cls-min-prob 0.33 \
  --cls-min-margin 0.07 \
  --box-expand-ratio 0.00 \
  "$@" > "$log_file" 2>&1

cat "$log_file"

uv run python scripts/extract_last_json.py --input "$log_file" --output "$result_json"

uv run python scripts/append_experiment_record.py \
  --date "$date_str" \
  --scope "$scope" \
  --run-name "$run_name" \
  --detector "$detector_desc" \
  --classifier "$classifier_desc" \
  --det-conf "$det_conf" \
  --det-iou "$det_iou" \
  --changes "$changes" \
  --notes "$notes" \
  --result-json "$result_json"
