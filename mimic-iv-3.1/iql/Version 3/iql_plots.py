import re
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

pat = re.compile(
    r"\[Epoch\s+(\d+)/(\d+)\]\s+train\s+q=([0-9.]+)\s+v=([0-9.]+)\s+pi=([0-9.]+)\s+\|\s+val\s+q=([0-9.]+)\s+v=([0-9.]+)\s+pi=([0-9.]+)"
)

def minmax_01(series):
    smin = series.min()
    smax = series.max()
    if smax == smin:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - smin) / (smax - smin)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=r"C:\Users\20231942\Desktop\Central Folder\TUe\Year 3\Honors\Code\CareQueue\mimic-iv-3.1\iql\Version 3\perfected_train.log", help="Path to training log (.out or .log)")
    ap.add_argument("--outdir", default=None, help="Where to save pngs (default: same folder as log)")
    args = ap.parse_args()

    log_path = Path(args.log)
    outdir = Path(args.outdir) if args.outdir else log_path.parent
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pat.search(line)
        if m:
            rows.append({
                "epoch": int(m.group(1)),
                "train_q": float(m.group(3)),
                "train_v": float(m.group(4)),
                "train_pi": float(m.group(5)),
                "val_q": float(m.group(6)),
                "val_v": float(m.group(7)),
                "val_pi": float(m.group(8)),
            })

    if not rows:
        raise SystemExit("No epoch lines matched. Check the log format / path.")

    df = pd.DataFrame(rows).sort_values("epoch")

    # Normalize validation losses to [0, 1]
    df["val_q_norm"] = minmax_01(df["val_q"])
    df["val_pi_norm"] = minmax_01(df["val_pi"])

    df.to_csv(outdir / "metrics_from_log.csv", index=False)

    def plot_single(col, title, fname):
        plt.figure()
        plt.plot(df["epoch"], df[col], marker='o', linestyle='-')
        plt.xlabel("epoch")
        plt.ylabel("normalized loss")
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(outdir / fname, dpi=150)
        plt.close()

    plot_single("val_q_norm", "Normalized Validation Q Bellman Loss", "iql_q_norm.png")
    plot_single("val_pi_norm", "Normalized Validation Behavior Cloning Loss", "iql_pi_norm.png")

    print("Saved:")
    print(" -", outdir / "metrics_from_log.csv")
    print(" -", outdir / "iql_q_norm.png")
    print(" -", outdir / "iql_pi_norm.png")

if __name__ == "__main__":
    main()