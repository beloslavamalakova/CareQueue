#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(".")

EVENTS_4H = ROOT / "cache" / "events_4h_long.parquet"      # your 4h-binned long events
TOP5_CSV  = ROOT / "interim" / "feature_itemid_top5.csv"   # already created by you
OUT_STATES = ROOT / "interim" / "states_4h.parquet"

# choose aggregation already present in your binned parquet:
BIN_VALUE_COL = "last_valuenum"  # or mean_valuenum/min_valuenum/max_valuenum


# ----------------------------
# SOFA scoring (standard thresholds)
# ----------------------------
def sofa_resp_score(pf, on_support):
    if pf is None or np.isnan(pf): return np.nan
    if pf > 400: return 0
    if pf > 300: return 1
    if pf > 200: return 2
    if pf > 100: return 3 if on_support else 2
    return 4 if on_support else 3

def sofa_coag_score(plt):
    if plt is None or np.isnan(plt): return np.nan
    if plt > 150: return 0
    if plt > 100: return 1
    if plt > 50:  return 2
    if plt > 20:  return 3
    return 4

def sofa_liver_score(bili):
    if bili is None or np.isnan(bili): return np.nan
    if bili < 1.2: return 0
    if bili < 2.0: return 1
    if bili < 6.0: return 2
    if bili < 12.0: return 3
    return 4

def sofa_cns_score(gcs):
    if gcs is None or np.isnan(gcs): return np.nan
    if gcs == 15: return 0
    if gcs >= 13: return 1
    if gcs >= 10: return 2
    if gcs >= 6:  return 3
    return 4

def sofa_renal_score(creat, urine_24h_equiv):
    c_score = np.nan
    u_score = np.nan

    if creat is not None and not np.isnan(creat):
        if creat < 1.2: c_score = 0
        elif creat < 2.0: c_score = 1
        elif creat < 3.5: c_score = 2
        elif creat < 5.0: c_score = 3
        else: c_score = 4

    if urine_24h_equiv is not None and not np.isnan(urine_24h_equiv):
        if urine_24h_equiv < 200: u_score = 4
        elif urine_24h_equiv < 500: u_score = 3
        else: u_score = 0

    if np.isnan(c_score) and np.isnan(u_score): return np.nan
    if np.isnan(c_score): return u_score
    if np.isnan(u_score): return c_score
    return max(c_score, u_score)

def norepi_equiv(norepi, epi, dopamine, phenyl, vasopressin):
    ne = 0.0
    for x in [norepi, epi]:
        if x is not None and not np.isnan(x):
            ne += float(x)
    if dopamine is not None and not np.isnan(dopamine):
        ne += float(dopamine) * (0.1 / 15.0)
    if phenyl is not None and not np.isnan(phenyl):
        ne += float(phenyl) * (0.1 / 1.0)
    if vasopressin is not None and not np.isnan(vasopressin):
        ne += float(vasopressin) * (0.1 / 0.04)
    return ne

def sofa_cv_score(map_mmHg, dobutamine_any, ne_eq, dopamine):
    if dobutamine_any:
        return 2
    if ne_eq is not None and not np.isnan(ne_eq) and ne_eq > 0:
        return 4 if ne_eq > 0.1 else 3
    if dopamine is not None and not np.isnan(dopamine) and dopamine > 0:
        if dopamine <= 5: return 2
        if dopamine <= 15: return 3
        return 4
    if map_mmHg is None or np.isnan(map_mmHg): return np.nan
    return 0 if map_mmHg >= 70 else 1


# ----------------------------
# Load top5 feature mapping
# ----------------------------
def normalize_feature_name(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace(" ", "_").replace("/", "_")
    return s

def load_top5(top5_path: Path) -> pd.DataFrame:
    top5 = pd.read_csv(top5_path)
    req = {"feature", "source", "id_type", "id"}
    missing = req - set(top5.columns)
    if missing:
        raise ValueError(f"TOP5 missing cols: {missing}. Found={list(top5.columns)}")

    df = top5[top5["id_type"].astype(str).str.lower().eq("itemid")].copy()
    df["feature"] = df["feature"].map(normalize_feature_name)
    df["itemid"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["itemid"])
    df["itemid"] = df["itemid"].astype(int)
    return df


# ----------------------------
# Derive SOFA itemids from top5 feature names
# (simple, transparent substring matching)
# ----------------------------
SOFA_KEYWORDS = {
    # respiratory
    "pao2": ["pao2", "pa_o2", "arterial_po2", "po2"],
    "fio2": ["fio2", "fi_o2", "fraction_inspired_oxygen"],
    "vent": ["vent", "mechanical_vent", "ventilation", "peep"],

    # coag/liver/renal
    "platelets": ["platelet", "plt"],
    "bilirubin": ["bilirubin", "bili"],
    "creatinine": ["creatinine", "creat", "scr"],
    "urine": ["urine", "uo", "urine_output"],

    # cns
    "gcs_total": ["gcs", "glasgow"],

    # cardio
    "map": ["map", "mean_arterial_pressure"],
    "norepi": ["norepi", "norepinephrine", "noradrenaline"],
    "epi": ["epi", "epinephrine", "adrenaline"],
    "dopamine": ["dopamine"],
    "dobutamine": ["dobutamine"],
    "phenyl": ["phenylephrine"],
    "vasopressin": ["vasopressin"],
}

def match_itemids(top5_df: pd.DataFrame, keys: list[str]) -> list[int]:
    feats = top5_df["feature"].tolist()
    mask = np.zeros(len(feats), dtype=bool)
    for k in keys:
        mask |= top5_df["feature"].str.contains(k, regex=False)
    return top5_df.loc[mask, "itemid"].drop_duplicates().tolist()

def derive_sofa_itemids_from_top5(top5_df: pd.DataFrame) -> dict:
    out = {}
    for var, keys in SOFA_KEYWORDS.items():
        out[var] = match_itemids(top5_df, keys)
    return out


# ----------------------------
# Build states
# ----------------------------
def load_events(events_path: Path) -> pd.DataFrame:
    ev = pd.read_parquet(events_path)
    req = {"stay_id", "bin_idx", "itemid", "source"}
    miss = req - set(ev.columns)
    if miss:
        raise ValueError(f"events_4h_long missing {miss}. Found={list(ev.columns)}")
    if BIN_VALUE_COL not in ev.columns:
        raise ValueError(f"{BIN_VALUE_COL} not in events columns.")
    ev = ev[["stay_id", "bin_idx", "itemid", "source", BIN_VALUE_COL]].rename(columns={BIN_VALUE_COL: "v"})
    return ev.dropna(subset=["v"])

def build_top5_wide(ev: pd.DataFrame, top5_df: pd.DataFrame) -> pd.DataFrame:
    # build mapping per source (chart vs lab) based on your top5 file's 'source'
    icu_map = dict(top5_df[top5_df["source"].eq("icu/d_items")][["itemid", "feature"]].drop_duplicates().values)
    lab_map = dict(top5_df[top5_df["source"].eq("hosp/d_labitems")][["itemid", "feature"]].drop_duplicates().values)

    df = ev.copy()
    src = df["source"].astype(str).str.lower()

    df["feature"] = np.where(
        src.eq("chart"),
        df["itemid"].map(icu_map),
        df["itemid"].map(lab_map),
    )
    df = df.dropna(subset=["feature"])

    wide = df.pivot_table(index=["stay_id", "bin_idx"], columns="feature", values="v", aggfunc="mean")
    wide.columns = [f"top5__{c}" for c in wide.columns]
    return wide.reset_index()

def build_sofa_4h(ev: pd.DataFrame, sofa_itemids: dict) -> pd.DataFrame:
    out = []
    for (stay_id, bin_idx), g in ev.groupby(["stay_id", "bin_idx"], sort=False):
        g_chart = g[g["source"].astype(str).str.lower().eq("chart")]
        g_lab   = g[g["source"].astype(str).str.lower().eq("lab")]
        g_out   = g[g["source"].astype(str).str.lower().eq("output")]

        def agg(gx, itemids, how):
            if not itemids: return np.nan
            x = gx.loc[gx["itemid"].isin(itemids), "v"].astype(float)
            if x.empty: return np.nan
            if how == "min": return float(x.min())
            if how == "max": return float(x.max())
            if how == "mean": return float(x.mean())
            if how == "sum": return float(x.sum())
            raise ValueError(how)

        def any_present(gx, itemids):
            return bool(itemids) and gx["itemid"].isin(itemids).any()

        # Resp
        pao2 = agg(g_chart, sofa_itemids["pao2"], "min")
        fio2 = agg(g_chart, sofa_itemids["fio2"], "max")
        if fio2 is not None and not np.isnan(fio2) and fio2 > 1.5:
            fio2 = fio2 / 100.0
        pf = np.nan
        if pao2 is not None and not np.isnan(pao2) and fio2 is not None and not np.isnan(fio2) and fio2 > 0:
            pf = pao2 / fio2
        on_vent = any_present(g_chart, sofa_itemids["vent"])
        resp_s = sofa_resp_score(pf, on_vent)

        # Coag/Liver
        plt = agg(g_lab, sofa_itemids["platelets"], "min")
        bili = agg(g_lab, sofa_itemids["bilirubin"], "max")
        coag_s = sofa_coag_score(plt)
        liver_s = sofa_liver_score(bili)

        # Renal (urine output usually in outputevents; if not present in your pipeline, it'll be NaN)
        creat = agg(g_lab, sofa_itemids["creatinine"], "max")
        urine_4h = agg(g_out, sofa_itemids["urine"], "sum")
        urine_24h_equiv = urine_4h * 6 if urine_4h is not None and not np.isnan(urine_4h) else np.nan
        renal_s = sofa_renal_score(creat, urine_24h_equiv)

        # CNS
        gcs = agg(g_chart, sofa_itemids["gcs_total"], "min")
        cns_s = sofa_cns_score(gcs)

        # CV
        map_mmHg = agg(g_chart, sofa_itemids["map"], "min")
        norepi = agg(g_chart, sofa_itemids["norepi"], "max")
        epi = agg(g_chart, sofa_itemids["epi"], "max")
        dopamine = agg(g_chart, sofa_itemids["dopamine"], "max")
        phenyl = agg(g_chart, sofa_itemids["phenyl"], "max")
        vasop = agg(g_chart, sofa_itemids["vasopressin"], "max")
        dobut_any = any_present(g_chart, sofa_itemids["dobutamine"])
        ne_eq = norepi_equiv(norepi, epi, dopamine, phenyl, vasop)
        cv_s = sofa_cv_score(map_mmHg, dobut_any, ne_eq, dopamine)

        subs = [resp_s, coag_s, liver_s, renal_s, cns_s, cv_s]
        sofa_total = float(np.nansum(subs)) if not all(np.isnan(x) for x in subs) else np.nan

        out.append({
            "stay_id": stay_id,
            "bin_idx": bin_idx,
            "sofa_resp": resp_s,
            "sofa_coag": coag_s,
            "sofa_liver": liver_s,
            "sofa_renal": renal_s,
            "sofa_cns": cns_s,
            "sofa_cv": cv_s,
            "sofa_total_4h": sofa_total,
            "pf_ratio": pf,
        })
    return pd.DataFrame(out)

def main():
    top5 = load_top5(TOP5_CSV)
    sofa_itemids = derive_sofa_itemids_from_top5(top5)

    # quick visibility: what SOFA signals exist in your top5 list
    print("SOFA itemids found from top5 feature names:")
    for k, v in sofa_itemids.items():
        print(f"  {k:12s}: {len(v)} itemids")

    ev = load_events(EVENTS_4H)

    top5_wide = build_top5_wide(ev, top5)
    sofa_df = build_sofa_4h(ev, sofa_itemids)

    states = sofa_df.merge(top5_wide, on=["stay_id", "bin_idx"], how="left")

    OUT_STATES.parent.mkdir(parents=True, exist_ok=True)
    states.to_parquet(OUT_STATES, index=False)
    print("Saved:", OUT_STATES, "shape:", states.shape)

if __name__ == "__main__":
    main()
