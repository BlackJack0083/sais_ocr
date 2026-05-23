#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


def q(value: float) -> str:
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def frange(start: float, stop: float, step: float) -> list[str]:
    values: list[str] = []
    current = Decimal(str(start))
    end = Decimal(str(stop))
    inc = Decimal(str(step))
    while current <= end + Decimal("0.000001"):
        values.append(q(float(current)))
        current += inc
    return sorted(set(values), key=float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and launch a fine sweep from the best coarse sweep row.")
    parser.add_argument("--coarse-csv", type=Path, required=True)
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--done-file", type=Path, required=True)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--workdir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.coarse_csv.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {args.coarse_csv}")

    best = rows[0]
    det_conf = best["det_conf"]
    det_iou = best["det_iou"]
    box_expand = best["box_expand_ratio"]
    best_prob = float(best["cls_min_prob"])
    best_margin = float(best["cls_min_margin"])

    prob_values = ",".join(frange(max(0.0, best_prob - 0.03), min(1.0, best_prob + 0.03), 0.01))
    margin_values = ",".join(frange(max(0.0, best_margin - 0.02), min(1.0, best_margin + 0.02), 0.01))

    command = (
        f"bash scripts/run_fine_sweep.sh {args.cuda_visible_devices} "
        f"{det_conf} {det_iou} {box_expand} {prob_values} {margin_values} "
        f"{args.output_csv} {args.done_file}"
    )

    script = args.workdir / "scripts" / "run_tmux_job.sh"
    import subprocess

    subprocess.run(
        ["bash", str(script), args.session_name, str(args.workdir), str(args.log_file), command],
        check=True,
        cwd=args.workdir,
    )


if __name__ == "__main__":
    main()
