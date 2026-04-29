from pathlib import Path
import pandas as pd

MIMIC_ROOT = Path("/Users/20243322/OneDrive - TU Eindhoven/Desktop/A - honors/mimic-iv-3.1")

d_items_path = MIMIC_ROOT / "icu" / "d_items.csv.gz"

d_items = pd.read_csv(d_items_path)

terms = [
    "saline", "lactated", "ringer", "dextrose",
    "albumin", "plasma", "platelet", "cryoprecipitate",
    "packed red", "blood", "prbc"
]

mask = d_items["label"].str.lower().fillna("").apply(
    lambda x: any(t in x for t in terms)
)

results = d_items.loc[
    mask,
    ["itemid", "label", "category", "unitname"]
].sort_values(["category", "label"])

print(results.to_string(index=False))