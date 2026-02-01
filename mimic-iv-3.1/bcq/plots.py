#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dir",
        default="../cache/bcq_run_tau02",
        help="BCQ run directory containing metrics.csv",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively instead of only saving",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv not found at: {metrics_path}")

    df = pd.read_csv(metrics_path)

    required = {"epoch", "val_bc", "val_q"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in metrics.csv: {missing}")

    # ---------- Plot 1: Validation BC loss ----------
    plt.figure()
    plt.plot(df["epoch"], df["val_bc"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title("Validation Behavior Cloning loss (BCQ)")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_bc = run_dir / "val_bc_loss.png"
    plt.savefig(out_bc, dpi=200)
    if args.show:
        plt.show()
    plt.close()

    # ---------- Plot 2: Validation Q Bellman loss ----------
    plt.figure()
    plt.plot(df["epoch"], df["val_q"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title("Validation Q Bellman loss (BCQ)")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_q = run_dir / "val_q_loss.png"
    plt.savefig(out_q, dpi=200)
    if args.show:
        plt.show()
    plt.close()

    print("Saved validation-only plots:")
    print(" ", out_bc)
    print(" ", out_q)


if __name__ == "__main__":
    main()
