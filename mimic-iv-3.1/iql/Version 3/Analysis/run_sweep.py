#!/usr/bin/env python3
"""
Run IQL training for multiple seeds and hyperparameter settings, then evaluate metrics.

Creates:
experiments/<tag>/seed_<k>/{best.pt,final.pt,schema_and_norm.json,...}
experiments/<tag>/metrics_per_seed.csv
experiments/<tag>/metrics_summary.csv
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from typing import Dict, List

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, required=True, help="Transitions parquet path")
    p.add_argument("--out_root", type=str, default="experiments", help="Where experiments are written")
    p.add_argument("--device", type=str, default=None, help="cpu/cuda (optional)")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")

    # You can adjust these defaults
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=1024)

    # Eval knobs
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--unsupported_thresh", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--fqe_steps", type=int, default=10000)

    return p.parse_args()


def run(cmd: List[str]) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def grid_to_tag(params: Dict) -> str:
    parts = []
    for k, v in params.items():
        parts.append(f"{k}{v}")
    return "_".join(parts)


def main():
    print(">>> run_sweep.py started")
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    os.makedirs(args.out_root, exist_ok=True)

    # Hyperparameter grid - i.e. the parameter combinations we explore
    grid = {
        "expectile_tau": [0.6, 0.7, 0.8],
        "awr_beta": [1.0, 3.0, 5.0],
        "hidden": [128, 256],
    }

    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    all_results = []

    for combo in itertools.product(*values):
        hp = dict(zip(keys, combo))
        tag = grid_to_tag(hp)
        exp_dir = os.path.join(args.out_root, tag)
        os.makedirs(exp_dir, exist_ok=True)

        per_seed_rows = []

        # Train + Eval for each seed
        for seed in seeds:
            run_dir = os.path.join(exp_dir, f"seed_{seed}")
            os.makedirs(run_dir, exist_ok=True)

            train_cmd = [
                sys.executable, "iql_training.py",
                "--data", args.data,
                "--save_dir", run_dir,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--seed", str(seed),
                "--expectile_tau", str(hp["expectile_tau"]),
                "--awr_beta", str(hp["awr_beta"]),
                "--hidden", str(hp["hidden"]),
                "--val_frac", str(args.val_frac),
            ]
            if args.device:
                train_cmd += ["--device", args.device]
            run(train_cmd)

            eval_cmd = [
                sys.executable, os.path.join("Analysis", "eval_iql_metrics.py"),
                "--data", args.data,
                "--run_dir", run_dir,
                "--ckpt", "best.pt",
                "--seed", str(seed),
                "--val_frac", str(args.val_frac),
                "--unsupported_thresh", str(args.unsupported_thresh),
                "--gamma", str(args.gamma),
                "--fqe_steps", str(args.fqe_steps),
                "--cwpdis_max_episodes", "2000",
                "--cwpdis_max_horizon", "50",
                "--beh_epochs", "10",
                ]
            if args.device:
                eval_cmd += ["--device", args.device]

            # capture JSON output from eval
            out = subprocess.check_output(eval_cmd).decode("utf-8").strip()
            last_brace = out.rfind("{")
            metrics = json.loads(out[last_brace:])
            metrics.update(hp)
            per_seed_rows.append(metrics)

        df = pd.DataFrame(per_seed_rows)
        df.to_csv(os.path.join(exp_dir, "metrics_per_seed.csv"), index=False)

        # Summary (mean/std across seeds)
        metric_cols = ["fqe", "kl_pi_vs_clinician", "unsupported_pct", "cwpdis"]
        summary = {}
        for c in metric_cols:
            summary[c + "_mean"] = float(df[c].mean())
            summary[c + "_std"] = float(df[c].std(ddof=1)) if len(df) > 1 else 0.0
        summary.update(hp)
        summary["tag"] = tag

        pd.DataFrame([summary]).to_csv(os.path.join(exp_dir, "metrics_summary.csv"), index=False)
        all_results.append(summary)

        print("\n=== SUMMARY:", tag, "===")
        print(pd.DataFrame([summary]).to_string(index=False))

    # Setting overall leaderboard
    all_df = pd.DataFrame(all_results)
    # Sort by FQE mean (descending) primarily
    all_df = all_df.sort_values("fqe_mean", ascending=False)
    all_df.to_csv(os.path.join(args.out_root, "leaderboard.csv"), index=False)

    print("\nSaved leaderboard to:", os.path.join(args.out_root, "leaderboard.csv"))
    print(all_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()