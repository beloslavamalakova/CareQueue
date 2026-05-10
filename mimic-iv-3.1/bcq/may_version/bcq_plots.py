#!/usr/bin/env python3
"""Plot BCQ training metrics from Output/metrics.csv."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def minmax_01(series: pd.Series) -> pd.Series:
    smin = series.min()
    smax = series.max()
    if smax == smin:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - smin) / (smax - smin)


def plot_metric(df: pd.DataFrame, col: str, title: str, ylabel: str, outpath: Path) -> None:
    plt.figure()
    plt.plot(df["epoch"], df[col], marker="o", linestyle="-")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="Output/metrics.csv")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    metrics_path = Path(args.metrics)
    outdir = Path(args.outdir) if args.outdir else metrics_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_path).sort_values("epoch")
    df["val_q_norm"] = minmax_01(df["val_q_loss"])
    df["val_bc_norm"] = minmax_01(df["val_bc_loss"])
    df.to_csv(outdir / "metrics_with_norm.csv", index=False)

    plot_metric(df, "val_q_norm", "Normalized Validation Q Bellman Loss", "normalized loss", outdir / "bcq_q_norm.png")
    plot_metric(df, "val_bc_norm", "Normalized Validation Behavior Cloning Loss", "normalized loss", outdir / "bcq_bc_norm.png")
    plot_metric(df, "val_bc_acc", "Validation Behavior Cloning Accuracy", "accuracy", outdir / "bcq_bc_accuracy.png")

    print("Saved:")
    print(" -", outdir / "metrics_with_norm.csv")
    print(" -", outdir / "bcq_q_norm.png")
    print(" -", outdir / "bcq_bc_norm.png")
    print(" -", outdir / "bcq_bc_accuracy.png")


if __name__ == "__main__":
    main()
