import duckdb
from pathlib import Path
import pandas as pd

def extract_actions_continuous_per_bin(
    mimic_root: Path,
    bins_parquet: Path,
    fluid_itemids: list[int],
    vaso_itemids: dict[str, list[int]],  # keys: norepi, epi, dopamine, phenyl, vasopressin
    out_path: Path,
    bin_hours: int = 4,
    threads: int = 4,
    mem: str = "8GB",
    tmpdir: Path | None = None,
):
    """
    Outputs one row per (stay_id, bin_idx) with:
      fluid_ml_bin (sum)
      norepi_max, epi_max, dopamine_max, phenyl_max, vasopressin_max (max rate within bin)
      norepi_equiv_max  (max NE-equivalent within bin; see conversion note)
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if tmpdir is None:
        tmpdir = out_path.parent / "duckdb_tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={int(threads)};")
    con.execute(f"PRAGMA memory_limit='{mem}';")
    con.execute(f"PRAGMA temp_directory='{str(tmpdir)}';")

    bin_seconds = bin_hours * 3600
    icustays_path = mimic_root / "icu" / "icustays.csv.gz"
    inputevents_path = mimic_root / "icu" / "inputevents.csv.gz"

    # stays
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW icu_stays AS
    SELECT stay_id::BIGINT AS stay_id, intime::TIMESTAMP AS intime, outtime::TIMESTAMP AS outtime
    FROM read_csv_auto('{str(icustays_path)}', union_by_name=true)
    WHERE intime IS NOT NULL AND outtime IS NOT NULL AND outtime > intime;
    """)

    con.execute(f"CREATE OR REPLACE TEMP VIEW bins AS SELECT * FROM read_parquet('{str(bins_parquet)}');")

    # filter lists
    fluid_sql = ",".join(map(str, sorted(set(fluid_itemids)))) if fluid_itemids else "-1"
    def ids_sql(x): return ",".join(map(str, sorted(set(x)))) if x else "-1"

    norepi_sql = ids_sql(vaso_itemids.get("norepi", []))
    epi_sql    = ids_sql(vaso_itemids.get("epi", []))
    dop_sql    = ids_sql(vaso_itemids.get("dopamine", []))
    phen_sql   = ids_sql(vaso_itemids.get("phenyl", []))
    vasp_sql   = ids_sql(vaso_itemids.get("vasopressin", []))

    # Inputevents: amount (mL) and/or rate; schemas vary a bit, but commonly:
    #   starttime/endtime, amount, amountuom, rate, rateuom, itemid, stay_id
    # We'll do pragmatic extraction:
    # - fluids: sum(amount) in mL when amountuom indicates mL
    # - pressors: take MAX(rate) within bin (units depend on charting; you may standardize later)
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW ie AS
    SELECT
      i.stay_id::BIGINT AS stay_id,
      i.itemid::BIGINT AS itemid,
      i.starttime::TIMESTAMP AS starttime,
      i.endtime::TIMESTAMP AS endtime,
      i.amount::DOUBLE AS amount,
      lower(COALESCE(i.amountuom,'')) AS amountuom,
      i.rate::DOUBLE AS rate,
      lower(COALESCE(i.rateuom,'')) AS rateuom
    FROM read_csv_auto('{str(inputevents_path)}', union_by_name=true) i
    JOIN icu_stays s ON i.stay_id::BIGINT = s.stay_id
    WHERE i.starttime IS NOT NULL;
    """)

    # assign a bin based on starttime (good enough for discretization; if you want exact overlap,
    # you can allocate proportionally, but this is usually fine for RL actions)
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW ie_binned AS
    SELECT
      stay_id,
      CAST(FLOOR(EXTRACT(EPOCH FROM (starttime - (SELECT intime FROM icu_stays s2 WHERE s2.stay_id=ie.stay_id))) / {bin_seconds}) AS BIGINT) AS bin_idx,
      itemid,
      amount,
      amountuom,
      rate,
      rateuom
    FROM ie;
    """)

    # fluids sum (mL)
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW fluids AS
    SELECT
      stay_id, bin_idx,
      SUM(CASE
        WHEN itemid IN ({fluid_sql}) AND amount IS NOT NULL AND (amountuom LIKE '%ml%') THEN amount
        ELSE 0
      END) AS fluid_ml_bin
    FROM ie_binned
    WHERE bin_idx IS NOT NULL AND bin_idx >= 0
    GROUP BY stay_id, bin_idx;
    """)

    # pressor max rates by drug (rate unit harmonization is dataset-dependent)
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW pressors AS
    SELECT
      stay_id, bin_idx,
      MAX(CASE WHEN itemid IN ({norepi_sql}) THEN rate ELSE NULL END) AS norepi_max,
      MAX(CASE WHEN itemid IN ({epi_sql})    THEN rate ELSE NULL END) AS epi_max,
      MAX(CASE WHEN itemid IN ({dop_sql})    THEN rate ELSE NULL END) AS dopamine_max,
      MAX(CASE WHEN itemid IN ({phen_sql})   THEN rate ELSE NULL END) AS phenyl_max,
      MAX(CASE WHEN itemid IN ({vasp_sql})   THEN rate ELSE NULL END) AS vasopressin_max
    FROM ie_binned
    WHERE bin_idx IS NOT NULL AND bin_idx >= 0
    GROUP BY stay_id, bin_idx;
    """)

    # norepi-equivalent (example conversion guidance exists; tune to your units!) :contentReference[oaicite:24]{index=24}
    # Here we just combine as a rough max across signals; you should standardize units first if you can.
    con.execute("""
    CREATE OR REPLACE TEMP VIEW actions_cont AS
    SELECT
      COALESCE(f.stay_id, p.stay_id) AS stay_id,
      COALESCE(f.bin_idx, p.bin_idx) AS bin_idx,
      COALESCE(f.fluid_ml_bin, 0.0) AS fluid_ml_bin,
      p.norepi_max, p.epi_max, p.dopamine_max, p.phenyl_max, p.vasopressin_max,
      GREATEST(
        COALESCE(p.norepi_max, 0.0),
        COALESCE(p.epi_max, 0.0),
        COALESCE(p.dopamine_max, 0.0),
        COALESCE(p.phenyl_max, 0.0),
        COALESCE(p.vasopressin_max, 0.0)
      ) AS norepi_equiv_max
    FROM fluids f
    FULL OUTER JOIN pressors p
      ON f.stay_id = p.stay_id AND f.bin_idx = p.bin_idx;
    """)

    con.execute(f"COPY (SELECT * FROM actions_cont) TO '{str(out_path)}' (FORMAT PARQUET);")
    con.close()
    return out_path
