from pathlib import Path
import pandas as pd

def build_state_action_table(
    mimic_root: Path,
    top5_csv: Path,
    cache_dir: Path,
    # You must define itemids for SOFA variables you want to use:
    sofa_itemids: dict[str, list[int]],
    # For actions:
    fluid_itemids: list[int],
    vaso_itemids: dict[str, list[int]],
):
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- top5 itemids (your existing top5 mapping file) :contentReference[oaicite:25]{index=25}
    icu_top5_ids, lab_top5_ids, top5_map = load_top5_itemids(top5_csv)

    # include SOFA itemids too so the events parquet contains them
    icu_needed = sorted(set(icu_top5_ids + sum((sofa_itemids.get(k, []) for k in sofa_itemids), [])))
    lab_needed = sorted(set(lab_top5_ids + sum((sofa_itemids.get(k, []) for k in sofa_itemids), [])))

    bins_pq, events_pq = duckdb_make_4h_bins_and_agg(
        mimic_root=mimic_root,
        out_dir=cache_dir,
        bin_hours=4,
        icu_itemids=icu_needed,
        lab_itemids=lab_needed,
        include_labs=True,
    )

    bins = pd.read_parquet(bins_pq)
    events = pd.read_parquet(events_pq)

    # ---- top5 features wide
    top5_wide = events_long_to_feature_wide(events, top5_map)

    # ---- extract SOFA inputs from events (you map itemids -> variable names)
    # We'll use "worst in bin" = min for PaO2/FiO2 ratio? Actually ratio "lower is worse",
    # MAP lower is worse, platelets lower worse, GCS lower worse;
    # creatinine higher worse, bilirubin higher worse, pressor dose higher worse.
    # We'll pick appropriate aggregates from events parquet.
    def pull_var(itemids: list[int], source_kind: str, agg_col: str) -> pd.DataFrame:
        if not itemids:
            return pd.DataFrame(columns=["stay_id","bin_idx","value"])
        sub = events[(events["source"] == source_kind) & (events["itemid"].isin(itemids))].copy()
        if sub.empty:
            return pd.DataFrame(columns=["stay_id","bin_idx","value"])
        sub["value"] = sub[agg_col]
        # if multiple itemids map to same variable, choose worst (depends on variable)
        return sub.groupby(["stay_id","bin_idx"])["value"].agg("max").reset_index()

    # You decide which itemids correspond to each clinical variable.
    # Example expectations in sofa_itemids:
    #  'pao2': [...], 'fio2': [...], 'map': [...], 'gcs': [...],
    #  'bilirubin': [...], 'platelets': [...], 'creatinine': [...], 'urine': [...],
    #  'resp_support': [...] (optional binary)
    #
    # NOTE: PaO2/FiO2 ratio: use PaO2 (min worse) and FiO2 (max worse) in bin.
    pao2 = pull_var(sofa_itemids.get("pao2", []), "chart", "min_valuenum").rename(columns={"value":"pao2"})
    fio2 = pull_var(sofa_itemids.get("fio2", []), "chart", "max_valuenum").rename(columns={"value":"fio2"})

    map_df = pull_var(sofa_itemids.get("map", []), "chart", "min_valuenum").rename(columns={"value":"map"})
    gcs_df = pull_var(sofa_itemids.get("gcs", []), "chart", "min_valuenum").rename(columns={"value":"gcs"})

    bili = pull_var(sofa_itemids.get("bilirubin", []), "lab", "max_valuenum").rename(columns={"value":"bilirubin"})
    plt  = pull_var(sofa_itemids.get("platelets", []), "lab", "min_valuenum").rename(columns={"value":"platelets_k"})
    cr   = pull_var(sofa_itemids.get("creatinine", []), "lab", "max_valuenum").rename(columns={"value":"creatinine"})
    uo   = pull_var(sofa_itemids.get("urine_ml_day", []), "chart", "min_valuenum").rename(columns={"value":"urine_ml_day"})

    # resp support indicator (optional); if absent, set False and you’ll under-score 3–4 a bit.
    rs = pull_var(sofa_itemids.get("resp_support", []), "chart", "max_valuenum").rename(columns={"value":"resp_support_raw"})

    sofa_inputs = bins[["stay_id","bin_idx"]].copy()
    for d in [pao2,fio2,map_df,gcs_df,bili,plt,cr,uo,rs]:
        sofa_inputs = sofa_inputs.merge(d, on=["stay_id","bin_idx"], how="left")

    sofa_inputs["pao2_fio2"] = sofa_inputs["pao2"] / sofa_inputs["fio2"]
    sofa_inputs["on_resp_support"] = sofa_inputs.get("resp_support_raw", 0).fillna(0).astype(float) > 0

    # For CV SOFA pressor columns here are placeholders (you can pipe in your action extractor rates if desired)
    sofa_inputs["dopamine"] = np.nan
    sofa_inputs["dobutamine_any"] = False
    sofa_inputs["epinephrine"] = np.nan
    sofa_inputs["norepinephrine"] = np.nan

    sofa_scored = compute_sofa_per_bin(sofa_inputs)

    # ---- actions
    actions_cont_pq = cache_dir / "actions_cont.parquet"
    extract_actions_continuous_per_bin(
        mimic_root=mimic_root,
        bins_parquet=bins_pq,
        fluid_itemids=fluid_itemids,
        vaso_itemids=vaso_itemids,
        out_path=actions_cont_pq,
        bin_hours=4
    )
    actions_cont = pd.read_parquet(actions_cont_pq)
    actions_disc = build_25_action_space(actions_cont)

    # ---- final merge: state = (top5 features + SOFA), action = action_id
    state = bins[["stay_id","bin_idx","bin_start","bin_end"]].merge(top5_wide, on=["stay_id","bin_idx"], how="left")
    state = state.merge(sofa_scored[["stay_id","bin_idx","sofa_total","sofa_resp","sofa_neuro","sofa_cv","sofa_liver","sofa_coag","sofa_renal"]],
                        on=["stay_id","bin_idx"], how="left")
    out = state.merge(actions_disc[["stay_id","bin_idx","fluid_level","vaso_level","action_id"]],
                      on=["stay_id","bin_idx"], how="left")

    return out
