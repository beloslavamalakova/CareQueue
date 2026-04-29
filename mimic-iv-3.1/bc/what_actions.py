"""

                WHEN MAX(CASE WHEN p.itemid IN (225794) THEN 1 ELSE 0 END) = 1 THEN 1
                WHEN MAX(CASE WHEN p.itemid IN (224263,224268,225752) THEN 1 ELSE 0 END) = 1 THEN 2
                WHEN MAX(CASE WHEN p.itemid IN (229351) THEN 1 ELSE 0 END) = 1 THEN 3
                WHEN MAX(CASE WHEN p.itemid IN (227194) THEN 1 ELSE 0 END) = 1 THEN 4
                WHEN MAX(CASE WHEN p.itemid IN (225802
"""

from pathlib import Path
import duckdb

MIMIC_ROOT = Path("/Users/20243322/OneDrive - TU Eindhoven/Desktop/A - honors/mimic-iv-3.1")
ITEMID = [
   225158,  # Normal Saline
    225828,  # Lactated Ringers
    220949,  # D5W / Dextrose 5%

    # Colloids
    220862,  # Albumin 5%
    220864,  # Albumin 25%

    # Blood products
    225168,  # Packed Red Blood Cells (PRBC)
    220970,  # Fresh Frozen Plasma (FFP)
    225170,  # Platelets
    225171  # Cryoprecipitate
]

D_ITEMS = MIMIC_ROOT / "icu" / "d_items.csv.gz"
CHARTEVENTS = MIMIC_ROOT / "icu" / "chartevents.csv.gz"
OUT_CSV = MIMIC_ROOT / "interim" / f"chartevents_itemid_{ITEMID}.csv"

OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()

# Optional: helps with large MIMIC files
con.execute("PRAGMA memory_limit='8GB';")
con.execute("PRAGMA threads=4;")
con.execute("PRAGMA preserve_insertion_order=false;")

# Convert list to SQL format
itemids_str = ",".join(map(str, ITEMID))

# Query all itemids at once
item_info = con.execute(f"""
    SELECT itemid, label, category, unitname
    FROM read_csv_auto('{D_ITEMS}', union_by_name=true)
    WHERE itemid IN ({itemids_str})
    ORDER BY itemid
""").df()

print("\n=== ITEM INFO FOR ALL ITEMIDS ===")
print(item_info)
