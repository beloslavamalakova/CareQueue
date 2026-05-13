#!/usr/bin/env python3

import argparse
import pandas as pd
from pathlib import Path

def compute_top10_initial_sofa_survival(
    data_path: str,
    sample_size: int = 10_000,
    seed: int = 42,
):
    # Load BC/sepsis transitions parquet
    df = pd.read_parquet(data_path)

    required_cols = {"stay_id", "bin", "sofa_total", "reward"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort so first row per stay_id is the initial patient state
    df = df.sort_values(["stay_id", "bin"]).reset_index(drop=True)

    # Initial severity
    initial_df = (
        df.groupby("stay_id")
        .first()[["sofa_total"]]
    )

    # Final outcome
    terminal_df = (
        df.groupby("stay_id")
        .last()[["reward"]]
    )

    # Combine them
    patient_df = initial_df.join(terminal_df)

    # Same cohort as training/preprocessing: SOFA >= 2
    cohort_df = patient_df[patient_df["sofa_total"] >= 2].copy()

    if len(cohort_df) == 0:
        raise ValueError("No patients found with sofa_total >= 2")

    # Random 10k subset from training cohort
    actual_sample_size = min(sample_size, len(cohort_df))

    sample_df = cohort_df.sample(
        n=actual_sample_size,
        random_state=seed
    ).reset_index(drop=True)

    # Rank random subset by initial SOFA, highest first
    sample_df = sample_df.sort_values(
        by="sofa_total",
        ascending=False
    ).reset_index(drop=True)

    # Select top 10% highest SOFA patients
    n_top10 = max(1, int(len(sample_df) * 0.10))
    top10_df = sample_df.head(n_top10).copy()

    # reward = +1 means survived, reward = -1 means died
    top10_df["survived"] = (top10_df["reward"] > -1).astype(int)

    survival_rate = top10_df["survived"].mean() * 100

    print("\n=== RANDOM 10K → TOP 10% INITIAL SOFA SURVIVAL ===")
    print(f"Total initial patients: {len(patient_df)}")
    print(f"Training cohort patients, SOFA >= 2: {len(cohort_df)}")
    print(f"Random subset size: {len(sample_df)}")
    print(f"Top 10% selected: {len(top10_df)}")
    print(f"Mean initial SOFA in top 10%: {top10_df['sofa_total'].mean():.2f}")
    print(f"Survival percentage: {survival_rate:.2f}%")

    return survival_rate, top10_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=r"C:\Users\20231942\Desktop\Central Folder\TUe\Year 3\Honors\Code\CareQueue\mimic-iv-3.1\iql\Version 3\Processed\transitions.parquet",
        help="Path to BC sepsis parquet file, e.g. 25_actions_bc_sepsis.parquet"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=10_000
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    parser.add_argument(
        "--out",
        default="top10_initial_sofa_from_random_10k.csv"
    )

    args = parser.parse_args()

    survival_rate, top10_df = compute_top10_initial_sofa_survival(
        data_path=args.data,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    # Save CSV in same folder as the script
    output_path = Path(__file__).resolve().parent / args.out

    top10_df.to_csv(output_path, index=False)

    print(f"\nSaved selected patients to: {output_path}")