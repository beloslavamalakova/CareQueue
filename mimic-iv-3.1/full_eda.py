import os
import pandas as pd

IN_CSV = "/home/beloslava/Downloads/mimic-iv-3.1/interim/full_aggregated.csv"
OUT_SUMMARY_CSV = "/home/beloslava/Downloads/mimic-iv-3.1/interim/full_eda_summary.csv"

# Define tokens used in your aggregated file
MISSING_TOKEN = "nan"
DROPPED_TOKEN = "dropped"

def main():
    df = pd.read_csv(IN_CSV, dtype=str, keep_default_na=False, na_filter=False)

    feature_cols = [c for c in df.columns if c != "subject_id"]

    is_missing = df[feature_cols].eq("nan") | df[feature_cols].eq("")  # "" catches empty cells
    is_dropped = df[feature_cols].eq("dropped")
    is_bad = is_missing | is_dropped
    is_good = ~is_bad


    # Per-patient completeness
    good_counts = is_good.sum(axis=1)
    bad_counts = is_bad.sum(axis=1)

    fully_complete_mask = bad_counts.eq(0)
    n_fully_complete = int(fully_complete_mask.sum())
    pct_fully_complete = (n_fully_complete / len(df)) * 100

    # Summaries
    print("FULL EDA SUMMARY")
    print(f"File: {IN_CSV}")
    print(f"Rows (patients): {len(df):,}")
    print(f"Unique subject_id: {n_patients:,}")
    print(f"Number of feature columns: {n_features:,}")
    print("")
    print(f" Fully complete patients (no '{MISSING_TOKEN}' and no '{DROPPED_TOKEN}'): {n_fully_complete:,} ({pct_fully_complete:.2f}%)")

    # Completeness distribution
    completeness_pct = (good_counts / n_features) * 100
    print("\n--- Completeness distribution (per patient) ---")
    print(f"Mean completeness:   {completeness_pct.mean():.2f}%")
    print(f"Median completeness: {completeness_pct.median():.2f}%")
    print(f"Min completeness:    {completeness_pct.min():.2f}%")
    print(f"Max completeness:    {completeness_pct.max():.2f}%")

    # How many patients have at least X% completeness
    for thr in [50, 75, 90, 95, 99, 100]:
        cnt = int((completeness_pct >= thr).sum())
        print(f"Patients with >= {thr:>3}% completeness: {cnt:,} ({(cnt/len(df))*100:.2f}%)")

    # Per-feature badness
    missing_per_feature = is_missing.sum(axis=0).sort_values(ascending=False)
    dropped_per_feature = is_dropped.sum(axis=0).sort_values(ascending=False)
    bad_per_feature = (missing_per_feature + dropped_per_feature).sort_values(ascending=False)

    print("\n--- Top 15 features by TOTAL bad entries (missing + dropped) ---")
    top_bad = bad_per_feature.head(15)
    for feat, cnt in top_bad.items():
        miss = int(missing_per_feature[feat])
        drop = int(dropped_per_feature[feat])
        print(f"{feat:25s}  bad={int(cnt):>8,}  (missing={miss:>8,}, dropped={drop:>8,})")

    print("\n--- Top 15 features by MISSING ('nan') ---")
    for feat, cnt in missing_per_feature.head(15).items():
        print(f"{feat:25s}  missing={int(cnt):>8,}")

    print("\n--- Top 15 features by DROPPED (outliers) ---")
    for feat, cnt in dropped_per_feature.head(15).items():
        print(f"{feat:25s}  dropped={int(cnt):>8,}")

    # Which features most often break "full completeness"?
    # Look only at patients who are NOT complete, count features that are bad among them
    incomplete_df = df.loc[~fully_complete_mask, feature_cols]
    incomplete_bad_counts = (incomplete_df.eq(MISSING_TOKEN) | incomplete_df.eq(DROPPED_TOKEN)).sum(axis=0).sort_values(ascending=False)

    print("\n--- Top 15 features most responsible for incompleteness (among incomplete patients) ---")
    for feat, cnt in incomplete_bad_counts.head(15).items():
        print(f"{feat:25s}  bad_in_incomplete={int(cnt):>8,}")

    # Save a compact summary table
    summary = pd.DataFrame({
        "feature": feature_cols,
        "missing_nan": missing_per_feature.reindex(feature_cols).values,
        "dropped_outlier": dropped_per_feature.reindex(feature_cols).values,
    })
    summary["bad_total"] = summary["missing_nan"] + summary["dropped_outlier"]
    summary["bad_rate"] = summary["bad_total"] / len(df)
    summary = summary.sort_values(["bad_total", "feature"], ascending=[False, True])

    os.makedirs(os.path.dirname(OUT_SUMMARY_CSV), exist_ok=True)
    summary.to_csv(OUT_SUMMARY_CSV, index=False)

    print("Saved feature-level summary to:")
    print(OUT_SUMMARY_CSV)

if __name__ == "__main__":
    main()
