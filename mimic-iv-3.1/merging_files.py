import os
import pandas as pd
import numpy as np

MIMIC_ROOT = "/home/beloslava/Downloads/mimic-iv-3.1"
TOP5_PATH  = os.path.join(MIMIC_ROOT, "interim", "feature_itemid_top5.csv")

ICU_CHARTEVENTS = os.path.join(MIMIC_ROOT, "icu", "chartevents.csv.gz")
HOSP_LABEVENTS  = os.path.join(MIMIC_ROOT, "hosp", "labevents.csv.gz")

OUT_PATH = os.path.join(MIMIC_ROOT, "interim", "patient_feature_matrix_top5.csv")

# aggregation for many timepoints per patient+feature
AGG_FUNC = "median"   # alternatives: "mean", "max", "min", "last"


def sanitize_feature_name(s: str) -> str:
    """Make feature names safe as column names."""
    s = s.strip().lower()
    s = s.replace(" ", "_")
    s = s.replace("/", "_")
    return s

def build_itemid_maps(top5_df: pd.DataFrame):
    """
    Uses feature_itemid_top5.csv

    Expected columns:
      feature, source, id_type, id, ...

    Keeps ONLY rows where id_type == 'itemid'
    """

    required = {"feature", "source", "id_type", "id"}
    missing = required - set(top5_df.columns)
    if missing:
        raise ValueError(f"TOP5 file missing columns: {missing}. "
                         f"Found columns: {list(top5_df.columns)}")

    df = top5_df.copy()

    # only real numeric itemids
    df = df[df["id_type"] == "itemid"].copy()

    # sanitize feature names
    df["feature"] = (
        df["feature"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("/", "_", regex=False)
    )

    # numeric itemids
    df["itemid"] = pd.to_numeric(df["id"], errors="coerce")
    df = df.dropna(subset=["itemid"])
    df["itemid"] = df["itemid"].astype(int)

    # split by source
    icu_df = df[df["source"] == "icu/d_items"]
    hosp_df = df[df["source"] == "hosp/d_labitems"]

    icu_itemid_to_feature = dict(
        icu_df[["itemid", "feature"]].drop_duplicates().values
    )

    hosp_itemid_to_feature = dict(
        hosp_df[["itemid", "feature"]].drop_duplicates().values
    )

    features = sorted(df["feature"].unique())

    return icu_itemid_to_feature, hosp_itemid_to_feature, features


def aggregate_events_to_patient_features(
    events_path: str,
    itemid_to_feature: dict,
    usecols: list,
    chunksize: int = 2_000_000,
    agg: str = "median",
    table_name: str = "events"
) -> pd.DataFrame:
    """
    Reads a big events table in chunks, filters to itemids of interest,
    maps itemid -> feature, aggregates per (subject_id, feature) -> value.

    Returns long format:
      subject_id | feature | value
    """
    wanted_itemids = set(itemid_to_feature.keys())
    if not wanted_itemids:
        return pd.DataFrame(columns=["subject_id", "feature", "value"])

    partial = []

    reader = pd.read_csv(
        events_path,
        compression="gzip",
        usecols=usecols,
        chunksize=chunksize,
        low_memory=True
    )

    for i, chunk in enumerate(reader, start=1):
        # filter quickly by itemid
        chunk = chunk[chunk["itemid"].isin(wanted_itemids)]
        if chunk.empty:
            continue

        # keep numeric values only
        chunk["valuenum"] = pd.to_numeric(chunk["valuenum"], errors="coerce")
        chunk = chunk.dropna(subset=["valuenum", "subject_id", "itemid"])
        if chunk.empty:
            continue

        # map to feature
        chunk["feature"] = chunk["itemid"].map(itemid_to_feature)
        chunk = chunk.dropna(subset=["feature"])
        if chunk.empty:
            continue

        # aggregate within chunk to reduce memory
        if agg == "median":
            g = chunk.groupby(["subject_id", "feature"])["valuenum"].median()
        elif agg == "mean":
            g = chunk.groupby(["subject_id", "feature"])["valuenum"].mean()
        elif agg == "max":
            g = chunk.groupby(["subject_id", "feature"])["valuenum"].max()
        elif agg == "min":
            g = chunk.groupby(["subject_id", "feature"])["valuenum"].min()
        else:
            raise ValueError(f"Unsupported AGG_FUNC={agg}")

        partial.append(g.reset_index().rename(columns={"valuenum": "value"}))

        print(f"[{table_name}] chunk {i} -> kept {len(chunk):,} rows, partial groups {len(g):,}")

    if not partial:
        return pd.DataFrame(columns=["subject_id", "feature", "value"])

    long_df = pd.concat(partial, ignore_index=True)

    # final aggregate across chunks (same subject+feature can appear in many chunks)
    if agg == "median":
        out = long_df.groupby(["subject_id", "feature"])["value"].median().reset_index()
    elif agg == "mean":
        out = long_df.groupby(["subject_id", "feature"])["value"].mean().reset_index()
    elif agg == "max":
        out = long_df.groupby(["subject_id", "feature"])["value"].max().reset_index()
    elif agg == "min":
        out = long_df.groupby(["subject_id", "feature"])["value"].min().reset_index()

    return out


top5 = pd.read_csv(TOP5_PATH)
icu_map, hosp_map, features = build_itemid_maps(top5)

print(f"Top5 features found: {len(features)}")
print(f"ICU itemids: {len(icu_map)} | HOSP lab itemids: {len(hosp_map)}")

# ICU: chartevents
icu_long = aggregate_events_to_patient_features(
    events_path=ICU_CHARTEVENTS,
    itemid_to_feature=icu_map,
    usecols=["subject_id", "itemid", "valuenum", "charttime"],
    chunksize=2_000_000,
    agg=AGG_FUNC,
    table_name="icu/chartevents"
)
icu_long["source"] = "icu"

# HOSP: labevents
hosp_long = aggregate_events_to_patient_features(
    events_path=HOSP_LABEVENTS,
    itemid_to_feature=hosp_map,
    usecols=["subject_id", "itemid", "valuenum", "charttime"],
    chunksize=2_000_000,
    agg=AGG_FUNC,
    table_name="hosp/labevents"
)
hosp_long["source"] = "hosp"

# Pivot to wide, keeping ICU and HOSP separately
icu_wide = icu_long.pivot(index="subject_id", columns="feature", values="value")
icu_wide.columns = [f"{c}__icu" for c in icu_wide.columns]

hosp_wide = hosp_long.pivot(index="subject_id", columns="feature", values="value")
hosp_wide.columns = [f"{c}__hosp" for c in hosp_wide.columns]

# Merge wide tables
wide = icu_wide.join(hosp_wide, how="outer")

# Create "combined" columns per feature: ICU first, else HOSP
for f in features:
    icu_col = f"{f}__icu"
    hosp_col = f"{f}__hosp"
    if icu_col in wide.columns or hosp_col in wide.columns:
        wide[f] = np.nan
        if icu_col in wide.columns:
            wide[f] = wide[icu_col]
        if hosp_col in wide.columns:
            # fill NaNs in combined with hosp values
            wide[f] = wide[f].fillna(wide[hosp_col])

# Put subject_id back as a column
wide = wide.reset_index()

# Optional: order columns nicely (subject_id, combined features, then source-specific)
combined_cols = ["subject_id"] + features
source_cols = sorted([c for c in wide.columns if c.endswith("__icu") or c.endswith("__hosp")])
final_cols = [c for c in combined_cols if c in wide.columns] + source_cols
wide = wide[final_cols]

# Save
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
wide.to_csv(OUT_PATH, index=False)

print(f"\nSaved patient-level feature matrix to:\n{OUT_PATH}")
print(f"Shape: {wide.shape}")
