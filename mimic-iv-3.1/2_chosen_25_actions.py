import re
import pandas as pd
from pathlib import Path

def find_itemids_in_d_items(mimic_root: Path, patterns: list[str]) -> pd.DataFrame:
    d_items = pd.read_csv(mimic_root / "icu" / "d_items.csv.gz", compression="gzip", low_memory=False)

    for c in ["label", "abbreviation", "category", "unitname", "linksto"]:
        if c not in d_items.columns:
            d_items[c] = ""

    text = (
        d_items["label"].astype(str) + " | " +
        d_items["abbreviation"].astype(str) + " | " +
        d_items["category"].astype(str) + " | " +
        d_items["unitname"].astype(str) + " | " +
        d_items["linksto"].astype(str)
    ).str.lower()

    mask = False
    for p in patterns:
        mask = mask | text.str.contains(p, regex=True)

    cols = ["itemid","label","abbreviation","category","unitname","linksto"]
    return d_items.loc[mask, cols].sort_values(["category","label"]).drop_duplicates()

def suggest_action_itemids(mimic_root: Path) -> dict[str, pd.DataFrame]:
    # vasopressors from your note: norepi, epi, vasopressin, dopamine, phenylephrine
    vaso_patterns = [
        r"\bnorepinephrine\b|\bnoradrenaline\b|\blevophed\b",
        r"\bepinephrine\b|\badrenaline\b",
        r"\bvasopressin\b|\bpitressin\b",
        r"\bdopamine\b",
        r"\bphenylephrine\b|\bneo[-\s]?synephrine\b",
    ]
    # fluids from your note: crystalloids, colloids, blood products
    fluid_patterns = [
        r"\bnormal saline\b|\b0\.9% saline\b|\bns\b",
        r"\blactated ringer\b|\bring(er)?'s\b|\blr\b",
        r"\bplasmalyte\b|\bplasma-lyte\b",
        r"\balbumin\b",           # colloid
        r"\bpacked red\b|\bprbc\b|\brbc\b|\bplatelet\b|\bffp\b|\bplasma\b",  # blood products
    ]

    return {
        "vasopressors_candidates": find_itemids_in_d_items(mimic_root, vaso_patterns),
        "fluids_candidates": find_itemids_in_d_items(mimic_root, fluid_patterns),
    }
