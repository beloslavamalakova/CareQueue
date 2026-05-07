#!/usr/bin/env python3
"""Small BCQ hyperparameter sweep runner."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path


def parse_list(value: str, cast):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="Processed/transitions.parquet")
    p.add_argument("--out_root", default="Sweeps")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--hidden", default="128")
    p.add_argument("--lr_q", default="1e-4")
    p.add_argument("--lr_bc", default="1e-4")
    p.add_argument("--thresholds", default="0.1,0.3,0.5")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    hidden_vals = parse_list(args.hidden, int)
    lr_q_vals = parse_list(args.lr_q, float)
    lr_bc_vals = parse_list(args.lr_bc, float)
    threshold_vals = parse_list(args.thresholds, float)
    seed_vals = parse_list(args.seeds, int)

    runs = []
    for hidden, lr_q, lr_bc, threshold, seed in itertools.product(hidden_vals, lr_q_vals, lr_bc_vals, threshold_vals, seed_vals):
        name = f"h{hidden}_lrq{lr_q:g}_lrbc{lr_bc:g}_thr{threshold:g}_seed{seed}"
        save_dir = out_root / name
        cmd = [
            sys.executable, "bcq_training.py",
            "--data", args.data,
            "--save_dir", str(save_dir),
            "--epochs", str(args.epochs),
            "--batch_size", str(args.batch_size),
            "--hidden", str(hidden),
            "--lr_q", str(lr_q),
            "--lr_bc", str(lr_bc),
            "--bcq_threshold", str(threshold),
            "--seed", str(seed),
        ]
        if args.device:
            cmd += ["--device", args.device]

        print("\n=== Running", name, "===")
        save_dir.mkdir(parents=True, exist_ok=True)
        with open(save_dir / "command.json", "w") as f:
            json.dump({"name": name, "cmd": cmd}, f, indent=2)
        subprocess.run(cmd, check=True)
        runs.append(name)

    with open(out_root / "runs.json", "w") as f:
        json.dump(runs, f, indent=2)
    print(f"Finished {len(runs)} runs. Results in {out_root}")


if __name__ == "__main__":
    main()
