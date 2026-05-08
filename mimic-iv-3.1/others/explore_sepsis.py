import re
import pandas as pd
from pathlib import Path

BASE_DIR = Path("/home/beloslava/Downloads/mimic-iv-3.1")
ICU_DITEMS = BASE_DIR / "icu" / "d_items.csv.gz"
HOSP_DLAB = BASE_DIR / "hosp" / "d_labitems.csv.gz"
HOSP_PAT = BASE_DIR / "hosp" / "patients.csv.gz"

OUT_PATH = BASE_DIR / "interim" / "feature_itemid_candidates.csv"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ----------------------------
# Feature search spec
# - include: regex patterns that should match label/abbrev
# - exclude: regex patterns that if present => throw away (to reduce junk matches)
# ----------------------------
FEATURES = {
    "heart rate": {
        "include": [r"\bheart\s*rate\b", r"\bhr\b"],
        "exclude": [r"alarm", r"orthostatic", r"infusion", r"ml/hr"],  # keep core HR
    },
    "respiratory rate": {
        "include": [r"\bresp(iratory)?\s*rate\b", r"\brr\b"],
        "exclude": [r"alarm", r"activity", r"aerobic", r"rest"],
    },
    "systolic bp": {
        "include": [r"\bsystolic\b", r"\bsbp\b", r"\bart\s*bp\s*systolic\b", r"\bnon\s*invasive\b.*\bsystolic\b"],
        "exclude": [r"pressure\s*support", r"vent"],
    },
    "diastolic bp": {
        "include": [r"\bdiastolic\b", r"\bdbp\b", r"\bart\s*bp\s*diastolic\b", r"\bnon\s*invasive\b.*\bdiastolic\b"],
        "exclude": [],
    },
    "mean bp": {
        "include": [r"\bmean\b.*\bbp\b", r"\bmap\b", r"\bmean\s*arterial\b"],
        "exclude": [r"\bmap\b.*\bbrain\b"],  # ICD “map brain” garbage
    },
    "spo2": {
        "include": [r"\bspo2\b", r"\boxygen\s*saturation\b", r"\bpulseox\b"],
        "exclude": [r"alarm", r"desat\s*limit"],
    },
    "temperature": {
        "include": [r"\btemperature\b", r"\btemp\b"],
        "exclude": [r"pacemaker", r"threshold", r"av\s*interval"],
    },
    "gcs": {
        "include": [r"\bgcs\b", r"\bglasgow\b.*\bcoma\b", r"\bgcs\s*-\s*eye\b", r"\bgcs\s*-\s*verbal\b", r"\bgcs\s*-\s*motor\b"],
        "exclude": [],
    },
    "mechanical ventilation": {
        "include": [r"\binvasive\s*ventilation\b", r"\bmechanical\b.*\bvent\b", r"\bnon[-\s]?invasive\s*ventilation\b"],
        "exclude": [],
    },
    "max vaso": {
        "include": [r"\bnorepinephrine\b", r"\bepinephrine\b", r"\bdopamine\b", r"\bphenylephrine\b", r"\bvasopressor\b"],
        "exclude": [r"epi\s*pen", r"ophth", r"lidocaine"],  # meds not pressors
    },

    # Labs (d_labitems)
    "ph": {
        "include": [r"\bpH\b"],
        "exclude": [r"\bpharm", r"\bpharmacy\b"],
    },
    "lactate": {
        "include": [r"\blactate\b"],
        "exclude": [r"lactate\s*dehydrogenase", r"\bld\b"],  # LDH != lactate
    },
    "creatinine": {"include": [r"\bcreatinine\b"], "exclude": []},
    "bun": {"include": [r"\bbun\b", r"\burea\s*nitrogen\b"], "exclude": []},
    "glucose": {"include": [r"\bglucose\b"], "exclude": []},
    "potassium": {
        "include": [r"\bpotassium\b", r"\bk\b"],
        "exclude": [r"hydroxide", r"\bkoh\b", r"penicillin"],  # KOH test / penicillin G potassium etc.
    },
    "sodium": {"include": [r"\bsodium\b", r"\bna\b"], "exclude": []},
    "chloride": {"include": [r"\bchloride\b", r"\bcl\b"], "exclude": []},
    "calcium": {
        "include": [r"\bcalcium\b", r"\bca\b"],
        "exclude": [r"ca-125", r"carbonate\s*crystals", r"oxalate\s*crystals"],
    },
    "ionised calcium": {"include": [r"\bionized\b.*\bcalcium\b", r"\bionised\b.*\bcalcium\b", r"\bionized\s*calcium\b"], "exclude": []},
    "co2": {"include": [r"\btotal\s*co2\b", r"\btco2\b", r"\bco2\b"], "exclude": [r"production"]},
    "bicarbonate": {"include": [r"\bbicarbonate\b", r"\bhco3\b"], "exclude": []},
    "base excess": {"include": [r"\bbase\s*excess\b"], "exclude": []},
    "hemoglobin": {"include": [r"\bhemoglobin\b", r"\bhgb\b"], "exclude": [r"a1c"]},
    "wbc": {"include": [r"\bwbc\b", r"\bwbc\s*count\b", r"white\s*blood\s*cell"], "exclude": [r"casts", r"clumps"]},
    "platelets": {"include": [r"\bplatelet\b", r"\bplt\b", r"platelet\s*count"], "exclude": [r"clumps", r"smear"]},
    "ptt": {"include": [r"\bptt\b"], "exclude": [r"\bla\b"]},
    "pt": {
        "include": [r"\bpt\b", r"\bprothrombin\b.*\btime\b"],
        "exclude": [r"physical\s*therapy", r"\bchest\s*pt\b", r"\bsplint\b", r"\bpt\s*category\b"],
    },
    "inr": {"include": [r"\binr\b"], "exclude": []},
}

def norm(s: str) -> str:
    s = "" if pd.isna(s) else str(s)
    s = s.lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def matches(text: str, include, exclude) -> bool:
    if not any(re.search(p, text, flags=re.IGNORECASE) for p in include):
        return False
    if any(re.search(p, text, flags=re.IGNORECASE) for p in exclude):
        return False
    return True

def extract_candidates_from_d_items(path: Path):
    df = pd.read_csv(path, compression="infer", low_memory=False)
    # expected cols: itemid, label, abbreviation, category, unitname, linksto...
    for c in ["label", "abbreviation", "category", "unitname", "linksto"]:
        if c not in df.columns:
            df[c] = ""
    df["_text"] = (
        df["label"].astype(str) + " | " +
        df["abbreviation"].astype(str) + " | " +
        df["category"].astype(str) + " | " +
        df["unitname"].astype(str) + " | " +
        df["linksto"].astype(str)
    ).map(norm)
    return df

def extract_candidates_from_d_labitems(path: Path):
    df = pd.read_csv(path, compression="infer", low_memory=False)
    # expected: itemid, label, fluid, category, loinc_code...
    for c in ["label", "fluid", "category", "loinc_code"]:
        if c not in df.columns:
            df[c] = ""
    df["_text"] = (
        df["label"].astype(str) + " | " +
        df["fluid"].astype(str) + " | " +
        df["category"].astype(str) + " | " +
        df["loinc_code"].astype(str)
    ).map(norm)
    return df

def main():
    out_rows = []

    # 1) patients columns
    pat_cols = list(pd.read_csv(HOSP_PAT, compression="infer", nrows=1).columns)
    for feat in ["age", "gender"]:
        for col in pat_cols:
            if feat == "age" and col.lower() in {"age", "anchor_age"}:
                out_rows.append({"feature": "age", "source": "hosp/patients", "id_type": "column", "id": col, "label": col})
            if feat == "gender" and col.lower() in {"gender", "sex"}:
                out_rows.append({"feature": "gender", "source": "hosp/patients", "id_type": "column", "id": col, "label": col})

    # 2) ICU d_items
    di = extract_candidates_from_d_items(ICU_DITEMS)
    for feat, spec in FEATURES.items():
        # ICU features mostly for vitals/vents/vaso/gcs etc.
        mask = di["_text"].apply(lambda t: matches(t, spec["include"], spec["exclude"]))
        hits = di.loc[mask, ["itemid", "label", "abbreviation", "category", "unitname", "linksto"]].copy()
        for _, r in hits.iterrows():
            out_rows.append({
                "feature": feat,
                "source": "icu/d_items",
                "id_type": "itemid",
                "id": int(r["itemid"]),
                "label": r["label"],
                "extra": f"abbr={r['abbreviation']}; cat={r['category']}; unit={r['unitname']}; linksto={r['linksto']}"
            })

    # 3) HOSP d_labitems
    dl = extract_candidates_from_d_labitems(HOSP_DLAB)
    for feat, spec in FEATURES.items():
        mask = dl["_text"].apply(lambda t: matches(t, spec["include"], spec["exclude"]))
        hits = dl.loc[mask, ["itemid", "label", "fluid", "category", "loinc_code"]].copy()
        for _, r in hits.iterrows():
            out_rows.append({
                "feature": feat,
                "source": "hosp/d_labitems",
                "id_type": "itemid",
                "id": int(r["itemid"]),
                "label": r["label"],
                "extra": f"fluid={r.get('fluid','')}; cat={r.get('category','')}; loinc={r.get('loinc_code','')}"
            })

    out = pd.DataFrame(out_rows).drop_duplicates()
    out.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")
    print(out.groupby(["feature","source"]).size().sort_values(ascending=False).head(30))

if __name__ == "__main__":
    main()
