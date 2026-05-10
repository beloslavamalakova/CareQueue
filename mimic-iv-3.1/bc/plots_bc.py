#!/usr/bin/env python3
from pathlib import Path
import argparse
import json

import pandas as pd
import matplotlib.pyplot as plt


def minmax_01(series: pd.Series) -> pd.Series:
    smin = series.min()
    smax = series.max()
    if smax == smin:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - smin) / (smax - smin)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        required=True,
        help="Path to BC save directory containing history.json",
    )
    ap.add_argument(
        "--outdir",
        default=None,
        help="Where to save plots. Default: same as run-dir",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    history_path = run_dir / "history.json"

    if not history_path.exists():
        raise FileNotFoundError(f"Could not find: {history_path}")

    outdir = Path(args.outdir) if args.outdir else run_dir
    outdir.mkdir(parents=True, exist_ok=True)

    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    if not history:
        raise SystemExit("history.json is empty.")

    df = pd.DataFrame(history).sort_values("epoch")

    # Save CSV version too
    df.to_csv(outdir / "bc_metrics_from_history.csv", index=False)

    # Normalized metrics, similar to your IQL plot
    df["train_loss_norm"] = minmax_01(df["train_loss"])
    df["val_loss_norm"] = minmax_01(df["val_loss"])

    # For accuracy/F1, usually no need to min-max normalize,
    # because they are already between 0 and 1.
    def plot_single(col, title, ylabel, fname, ylim=None):
        plt.figure()
        plt.plot(df["epoch"], df[col], marker="o", linestyle="-")
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.grid(True, alpha=0.3)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=150)
        plt.close()

    # 1. Normalized validation BC loss
    plot_single(
        "val_loss_norm",
        "Normalized Validation BC Loss",
        "normalized loss",
        "bc_val_loss_norm.png",
        ylim=(0, 1),
    )

    # 2. Normalized train vs validation BC loss
    plt.figure()
    plt.plot(df["epoch"], df["train_loss_norm"], marker="o", label="train loss")
    plt.plot(df["epoch"], df["val_loss_norm"], marker="o", label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("normalized loss")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.title("Normalized BC Train vs Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "bc_train_val_loss_norm.png", dpi=150)
    plt.close()

    # 3. Validation accuracy
    plot_single(
        "val_accuracy",
        "Validation Accuracy",
        "accuracy",
        "bc_val_accuracy.png",
        ylim=(0, 1),
    )

    # 4. Validation macro F1
    plot_single(
        "val_macro_f1",
        "Validation Macro F1",
        "macro F1",
        "bc_val_macro_f1.png",
        ylim=(0, 1),
    )

    # 5. Accuracy and macro F1 together
    plt.figure()
    plt.plot(df["epoch"], df["val_accuracy"], marker="o", label="val accuracy")
    plt.plot(df["epoch"], df["val_macro_f1"], marker="o", label="val macro F1")
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.title("BC Validation Accuracy and Macro F1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "bc_val_scores.png", dpi=150)
    plt.close()

    print("Saved:")
    print(" -", outdir / "bc_metrics_from_history.csv")
    print(" -", outdir / "bc_val_loss_norm.png")
    print(" -", outdir / "bc_train_val_loss_norm.png")
    print(" -", outdir / "bc_val_accuracy.png")
    print(" -", outdir / "bc_val_macro_f1.png")
    print(" -", outdir / "bc_val_scores.png")


if __name__ == "__main__":
    main()