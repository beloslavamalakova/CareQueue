import duckdb
from pathlib import Path

BASE = Path("/home/20243009/mimic-iv-3.1")
OUT = Path("/home/20243009/scripts/state_4h.parquet")

con = duckdb.connect()

print("Running DuckDB aggregation...")

query = f"""
COPY (
    SELECT
        ce.stay_id,
        FLOOR(
            EXTRACT(EPOCH FROM (ce.charttime - icu.intime)) / (4*3600)
        ) AS bin,

        AVG(CASE WHEN ce.itemid = 220045 THEN ce.valuenum END) AS HR,
        AVG(CASE WHEN ce.itemid = 223762 THEN ce.valuenum END) AS TEMP,
        AVG(CASE WHEN ce.itemid = 220277 THEN ce.valuenum END) AS SPO2,
        AVG(CASE WHEN ce.itemid = 220050 THEN ce.valuenum END) AS SBP,
        AVG(CASE WHEN ce.itemid = 220051 THEN ce.valuenum END) AS DBP,
        AVG(CASE WHEN ce.itemid = 220052 THEN ce.valuenum END) AS MBP

    FROM read_csv_auto('{BASE}/icu/chartevents.csv.gz') ce
    JOIN read_csv_auto('{BASE}/icu/icustays.csv.gz') icu
      ON ce.stay_id = icu.stay_id

    WHERE ce.itemid IN (
        220045, 223762, 220277,
        220050, 220051, 220052
    )

    GROUP BY ce.stay_id, bin
    ORDER BY ce.stay_id, bin
) TO '{OUT}' (FORMAT PARQUET);
"""

con.execute(query)

print("Done. Saved to:", OUT)