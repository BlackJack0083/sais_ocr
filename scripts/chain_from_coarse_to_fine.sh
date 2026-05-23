#!/usr/bin/env bash
set -euo pipefail

while [[ ! -f reports/sweep_20260522_coarse_merged.csv ]]; do
  sleep 60
done

uv run python scripts/launch_fine_sweep_from_coarse.py \
  --coarse-csv reports/sweep_20260522_coarse_merged.csv \
  --session-name infer_20260522_fine \
  --log-file logs/infer_20260522_fine.log \
  --output-csv reports/sweep_20260522_fine.csv \
  --done-file reports/sweep_20260522_fine.done \
  --cuda-visible-devices 0 \
  --workdir /mnt/data/hejiakai/sais_ocr

bash scripts/run_tmux_job.sh infer_20260522_fine_finalize /mnt/data/hejiakai/sais_ocr logs/infer_20260522_fine_finalize.log "bash scripts/finalize_fine_sweep.sh"
