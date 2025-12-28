import re
import pandas as pd
from pathlib import Path

BASE_DIR = Path("/home/beloslava/Downloads/mimic-iv-3.1")
IN_PATH = BASE_DIR / "interim" / "feature_itemid_candidates.csv"
OUT_BEST = BASE_DIR / "interim" / "feature_itemid_best.csv"
OUT_TOP5 = BASE_DIR / "interim" / "feature_itemid_top5.csv"

# Canonical “good” phrases for each feature (used for scoring)
CANON = {
    "heart rate": ["heart rate", "hr"],
    "respiratory rate": ["respiratory rate", "rr"],
    "systolic bp": ["arterial blood pressure systolic", "art bp systolic", "non invasive blood pressure systolic", "systolic"],
    "diastolic bp": ["arterial blood pressure diastolic", "art bp diastolic", "non invasive blood pressure diastolic", "diastolic"],
    "mean bp": ["mean arterial pressure", "map", "mean bp"],
    "spo2": ["spo2", "oxygen saturation", "pulseox", "o2 saturation"],
    "temperature": ["temperature celsius", "temperature fahrenheit", "temperature"],
    "gcs": ["gcs - eye opening", "gcs - verbal response", "gcs - motor response", "gcs"],
    "mechanical ventilation": ["invasive ventilation", "non-invasive ventilation", "ventilation"],
    "max vaso": ["norepinephrine", "epinephrine", "dopamine", "phenylephrine", "vasopressor"],
    "ph": ["ph"],
    "lactate": ["lactate"],
    "creatinine": ["creatinine"],
    "bun": ["bun", "urea nitrogen"],
    "glucose": ["glucose"],
    "potassium": ["potassium"],
    "sodium": ["sodium"],
    "chloride": ["chloride"],
    "calcium": ["calcium", "ionized calcium", "ionised calcium"],
    "ionised calcium": ["ionized calcium", "ionised calcium"],
    "co2": ["total co2", "tco2", "co2"],
    "bicarbonate": ["bicarbonate", "hco3"],
    "base excess": ["base excess"],
    "hemoglobin": ["hemoglobin", "hgb"],
    "wbc": ["wbc", "white blood cell"],
    "platelets": ["platelet", "plt"],
    "ptt": ["ptt"],
    "pt": ["pt", "prothrombin time"],
    "inr": ["inr"],
    "age": ["anchor_age", "age"],
    "gender": ["gender", "sex"],
}

# Penalize obviously-non-feature rows (helps avoid alarms/orthostatics/infusion rates/etc.)
BAD_TOKENS = [
    "alarm", "orthostatic", "manual", "site", "change", "threshold", "pacemaker",
    "ml/hr", "infusion", "bolus", "challenge", "desat", "limit",
    "boost", "control", "diet", "aerobic", "activity", "rest"
]

GOOD_HINTS = [
    # ICU vital “usually good”
    "arterial", "non invasive", "invasive", "pulseox", "celsius", "fahrenheit",
    # Labs “usually good”
    "serum", "whole blood"
]

def norm(x: str) -> str:
    x = "" if pd.isna(x) else str(x)
    x = x.lower().strip()
    x = re.sub(r"\s+", " ", x)
    return x

def score_row(feature: str, label: str, extra: str, source: str) -> float:
    f = norm(feature)
    lbl = norm(label)
    ex = norm(extra)
    text = f"{lbl} | {ex}"

    score = 0.0

    # Strong preference for “dictionary-looking” sources
    if source == "icu/d_items":
        score += 2.0
    if source == "hosp/d_labitems":
        score += 2.0
    if source == "hosp/patients":
        score += 3.0

    # Canonical match scoring
    canon_list = [norm(c) for c in CANON.get(feature, [feature])]
    for c in canon_list:
        if not c:
            continue
        # exact label match
        if lbl == c:
            score += 10.0
        # startswith
        if lbl.startswith(c):
            score += 6.0
        # contains in label
        if c in lbl:
            score += 4.0
        # contains anywhere (label/extra)
        if c in text:
            score += 2.0

    # Prefer item labels that look “core”, not derived controls/alarms/etc.
    for bt in BAD_TOKENS:
        if bt in text:
            score -= 3.0

    for gh in GOOD_HINTS:
        if gh in text:
            score += 1.0

    # Feature-specific boosts (tiny but helpful)
    if feature in {"heart rate", "respiratory rate"} and ("alarm" not in text) and ("orthostatic" not in text):
        score += 1.0

    if feature in {"temperature"} and ("celsius" in text or "fahrenheit" in text):
        score += 1.0

    if feature == "max vaso":
        # prefer pressor names and avoid weird stuff
        if any(x in text for x in ["norepinephrine", "epinephrine", "dopamine", "phenylephrine"]):
            score += 2.0

    if feature == "pt":
        # avoid physical therapy
        if "physical therapy" in text or "chest pt" in text or "splint" in text:
            score -= 10.0
        if "prothrombin" in text:
            score += 3.0

    if feature == "potassium":
        # avoid KOH / penicillin potassium confusion if it sneaks in
        if "hydroxide" in text or "koh" in text:
            score -= 10.0

    return score

def main():
    df = pd.read_csv(IN_PATH)
    if "extra" not in df.columns:
        df["extra"] = ""

    df["label"] = df["label"].astype(str)
    df["extra"] = df["extra"].astype(str)
    df["score"] = df.apply(lambda r: score_row(r["feature"], r["label"], r["extra"], r["source"]), axis=1)

    # rank within (feature, source)
    df = df.sort_values(["feature", "source", "score"], ascending=[True, True, False])
    df["rank_within_source"] = df.groupby(["feature", "source"]).cumcount() + 1

    top5 = df[df["rank_within_source"] <= 5].copy()
    top5.to_csv(OUT_TOP5, index=False)

    best = df[df["rank_within_source"] == 1].copy()
    best.to_csv(OUT_BEST, index=False)

    print(f"Saved top5: {OUT_TOP5}")
    print(f"Saved best: {OUT_BEST}")
    print("\nBest picks per feature/source (preview):")
    print(best[["feature","source","id_type","id","label","score"]].sort_values(["feature","source"]).head(50))

if __name__ == "__main__":
    main()
