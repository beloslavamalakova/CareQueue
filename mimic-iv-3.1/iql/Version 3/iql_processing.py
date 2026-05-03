#!/usr/bin/env python3
"""
Lite MIMIC processing using DuckDB (fast), based on Bella's BCQ code

Outputs a transitions parquet with columns:
  stay_id, bin, action, reward, done,
  s_*, s_next_*

Action space (0..5) is procedure-based:
  0: none
  1: itemid 225794
  2: itemid 224263,224268,225752
  3: itemid 229351
  4: itemid 227194
  5: itemid 225802
`

State features are 4h-bin averages from chartevents:
  HR(220045), TEMP(223762), SPO2(220277), SBP(220050), DBP(220051), MBP(220052)
"""

import argparse
from pathlib import Path
import duckdb

STATE_ITEMIDS = {
    "HR": 220045,
    "TEMP": 223762,
    "SPO2": 220277,
    "SBP": 220050,
    "DBP": 220051,
    "MBP": 220052,
}

LAB_ITEMIDS = {
    "platelets": 51265,
    "bilirubin": 50885,
    "creatinine": 50912,
}

VASO_ITEMIDS = [
    221906,  # norepinephrine
    221289,  # epinephrine
    221749,  # phenylephrine
    221662,  # dopamine
    222315,  # vasopressin
]

FLUID_ITEMIDS = [
    #crystalloids, collloids and blood products
    225158,  # normal saline
    225828,  # lactated ringers
    220949,  # D5W / dextrose
    220862,  # albumin 5%
    220864,  # albumin 25%
    225168,  # packed red blood cells
    220970,  # fresh frozen plasma
    225170,  # platelets
    225171,  # cryoprecipitate
]

def build_states(con: duckdb.DuckDBPyConnection, base: Path, out_state: Path, bin_hours: int):
    chartevents = base / "icu" / "chartevents.csv.gz"
    icustays = base / "icu" / "icustays.csv.gz"

    item_list = ",".join(str(v) for v in STATE_ITEMIDS.values())

    avg_cols = []
    for name, itemid in STATE_ITEMIDS.items():
        avg_cols.append(
            f"AVG(CASE WHEN ce.itemid = {itemid} THEN ce.valuenum END) AS {name}"
        )

    avg_cols_sql = ",\n            ".join(avg_cols)

    con.execute(f"""
    COPY (
        SELECT
            ce.stay_id::BIGINT AS stay_id,
            FLOOR(
                EXTRACT(EPOCH FROM (ce.charttime::TIMESTAMP - icu.intime::TIMESTAMP))
                / ({bin_hours} * 3600)
            )::BIGINT AS bin,
            {avg_cols_sql}
        FROM read_csv_auto('{chartevents}', union_by_name=true) ce
        JOIN read_csv_auto('{icustays}', union_by_name=true) icu
          ON ce.stay_id = icu.stay_id
        WHERE ce.itemid IN ({item_list})
          AND ce.valuenum IS NOT NULL
          AND ce.charttime IS NOT NULL
          AND icu.intime IS NOT NULL
          AND ce.stay_id IS NOT NULL
          AND ce.charttime::TIMESTAMP >= icu.intime::TIMESTAMP
        GROUP BY ce.stay_id, bin
        ORDER BY ce.stay_id, bin
    )
    TO '{out_state}' (FORMAT PARQUET);
    """)


def build_transitions(con, base, state_file, out_file, bin_hours):
    inputevents = base / "icu" / "inputevents.csv.gz"
    admissions = base / "hosp" / "admissions.csv.gz"
    patients = base / "hosp" / "patients.csv.gz"
    icustays = base / "icu" / "icustays.csv.gz"
    labevents = base / "hosp" / "labevents.csv.gz"
    prescriptions = base / "hosp" / "prescriptions.csv.gz"
    micro = base / "hosp" / "microbiologyevents.csv.gz"

    platelet = LAB_ITEMIDS["platelets"]
    bilirubin = LAB_ITEMIDS["bilirubin"]
    creatinine = LAB_ITEMIDS["creatinine"]

    vaso_sql = ",".join(map(str, VASO_ITEMIDS))
    fluid_sql = ",".join(map(str, FLUID_ITEMIDS))
    all_items = ",".join(map(str, sorted(set(VASO_ITEMIDS + FLUID_ITEMIDS))))

    con.execute(f"""
    CREATE TABLE states AS
    SELECT * FROM read_parquet('{state_file}');
    """)

    con.execute(f"""
    CREATE TABLE icu AS
    SELECT
        stay_id::BIGINT AS stay_id,
        subject_id::BIGINT AS subject_id,
        hadm_id::BIGINT AS hadm_id,
        intime::TIMESTAMP AS intime
    FROM read_csv_auto('{icustays}', union_by_name=true)
    WHERE stay_id IS NOT NULL
      AND subject_id IS NOT NULL
      AND hadm_id IS NOT NULL
      AND intime IS NOT NULL;
    """)

    con.execute(f"""
    CREATE TABLE pat AS
    SELECT
        subject_id::BIGINT AS subject_id,
        anchor_age::DOUBLE AS age,
        CASE
            WHEN gender = 'M' THEN 1.0
            WHEN gender = 'F' THEN 0.0
            ELSE NULL
        END::DOUBLE AS sex
    FROM read_csv_auto('{patients}', union_by_name=true)
    WHERE subject_id IS NOT NULL;
    """)

    con.execute(f"""
    CREATE TABLE adm AS
    SELECT
        hadm_id::BIGINT AS hadm_id,
        hospital_expire_flag::INTEGER AS hospital_expire_flag
    FROM read_csv_auto('{admissions}', union_by_name=true)
    WHERE hadm_id IS NOT NULL;
    """)

    con.execute(f"""
    CREATE TABLE ie AS
    SELECT
        stay_id::BIGINT AS stay_id,
        starttime::TIMESTAMP AS starttime,
        itemid::BIGINT AS itemid,
        amount::DOUBLE AS amount,
        rate::DOUBLE AS rate,
        patientweight::DOUBLE AS patientweight
    FROM read_csv_auto('{inputevents}', union_by_name=true)
    WHERE stay_id IS NOT NULL
      AND starttime IS NOT NULL
      AND itemid IN ({all_items});
    """)

    con.execute(f"""
    CREATE TABLE labs AS
    SELECT
        hadm_id::BIGINT AS hadm_id,
        charttime::TIMESTAMP AS charttime,
        itemid::BIGINT AS itemid,
        valuenum::DOUBLE AS valuenum
    FROM read_csv_auto('{labevents}', union_by_name=true)
    WHERE hadm_id IS NOT NULL
      AND charttime IS NOT NULL
      AND valuenum IS NOT NULL
      AND itemid IN ({platelet},{bilirubin},{creatinine});
    """)

    con.execute(f"""
    CREATE TABLE abx AS
    SELECT
        hadm_id::BIGINT AS hadm_id,
        starttime::TIMESTAMP AS abx_time,
        LOWER(drug) AS drug
    FROM read_csv_auto('{prescriptions}', union_by_name=true)
    WHERE hadm_id IS NOT NULL
      AND starttime IS NOT NULL
      AND (
          LOWER(drug) LIKE '%cillin%'
       OR LOWER(drug) LIKE '%cef%'
       OR LOWER(drug) LIKE '%mycin%'
       OR LOWER(drug) LIKE '%penem%'
       OR LOWER(drug) LIKE '%floxacin%'
       OR LOWER(drug) LIKE '%metronidazole%'
       OR LOWER(drug) LIKE '%vancomycin%'
      );
    """)

    con.execute(f"""
    CREATE TABLE cx AS
    SELECT
        hadm_id::BIGINT AS hadm_id,
        charttime::TIMESTAMP AS cx_time
    FROM read_csv_auto('{micro}', union_by_name=true)
    WHERE hadm_id IS NOT NULL
      AND charttime IS NOT NULL;
    """)

    con.execute("""
    CREATE TABLE infection AS
    SELECT DISTINCT
        abx.hadm_id,
        LEAST(abx.abx_time, cx.cx_time) AS inf_time
    FROM abx
    JOIN cx USING (hadm_id)
    WHERE
      (
        cx.cx_time <= abx.abx_time
        AND abx.abx_time <= cx.cx_time + INTERVAL '72 hours'
      )
      OR
      (
        abx.abx_time < cx.cx_time
        AND cx.cx_time <= abx.abx_time + INTERVAL '24 hours'
      );
    """)

    con.execute("""
    CREATE TABLE ns AS
    SELECT
      stay_id,
      bin,

      HR,
      TEMP,
      SPO2,
      SBP,
      DBP,
      MBP,

      LEAD(HR)   OVER (PARTITION BY stay_id ORDER BY bin) AS next_HR,
      LEAD(TEMP) OVER (PARTITION BY stay_id ORDER BY bin) AS next_TEMP,
      LEAD(SPO2) OVER (PARTITION BY stay_id ORDER BY bin) AS next_SPO2,
      LEAD(SBP)  OVER (PARTITION BY stay_id ORDER BY bin) AS next_SBP,
      LEAD(DBP)  OVER (PARTITION BY stay_id ORDER BY bin) AS next_DBP,
      LEAD(MBP)  OVER (PARTITION BY stay_id ORDER BY bin) AS next_MBP,
      LEAD(bin)  OVER (PARTITION BY stay_id ORDER BY bin) AS next_bin
    FROM states;
    """)

    con.execute(f"""
    CREATE TABLE tr AS
    SELECT
      s.stay_id,
      s.bin,

      COALESCE(MAX(
        CASE
          WHEN ie.itemid IN ({vaso_sql})
           AND ie.rate IS NOT NULL
           AND ie.patientweight IS NOT NULL
           AND ie.patientweight > 0
          THEN ie.rate / ie.patientweight
          ELSE NULL
        END
      ), 0.0) AS vaso,

      COALESCE(SUM(
        CASE
          WHEN ie.itemid IN ({fluid_sql})
          THEN COALESCE(ie.amount, 0.0)
          ELSE 0.0
        END
      ), 0.0) AS fluid

    FROM ns s
    LEFT JOIN icu USING (stay_id)
    LEFT JOIN ie
      ON s.stay_id = ie.stay_id
     AND ie.starttime >= icu.intime + (s.bin * INTERVAL '{bin_hours} hours')
     AND ie.starttime <  icu.intime + ((s.bin + 1) * INTERVAL '{bin_hours} hours')
    GROUP BY s.stay_id, s.bin;
    """)

    con.execute(f"""
    CREATE TABLE lb AS
    SELECT
      s.stay_id,
      s.bin,

      AVG(CASE WHEN labs.itemid = {platelet} THEN labs.valuenum END) AS platelets,
      AVG(CASE WHEN labs.itemid = {bilirubin} THEN labs.valuenum END) AS bilirubin,
      AVG(CASE WHEN labs.itemid = {creatinine} THEN labs.valuenum END) AS creatinine

    FROM ns s
    LEFT JOIN icu USING (stay_id)
    LEFT JOIN labs
      ON icu.hadm_id = labs.hadm_id
     AND labs.charttime >= icu.intime + (s.bin * INTERVAL '{bin_hours} hours')
     AND labs.charttime <  icu.intime + ((s.bin + 1) * INTERVAL '{bin_hours} hours')
    GROUP BY s.stay_id, s.bin;
    """)

    con.execute("""
    CREATE TABLE sofa AS
    SELECT
      s.stay_id,
      s.bin,

      (
        CASE
          WHEN COALESCE(tr.vaso, 0) > 0.1 THEN 4
          WHEN COALESCE(tr.vaso, 0) > 0 THEN 3
          WHEN s.MBP < 70 THEN 1
          ELSE 0
        END
        +
        CASE
          WHEN lb.creatinine > 5 THEN 4
          WHEN lb.creatinine > 3.5 THEN 3
          WHEN lb.creatinine > 2 THEN 2
          WHEN lb.creatinine > 1.2 THEN 1
          ELSE 0
        END
        +
        CASE
          WHEN lb.platelets < 20 THEN 4
          WHEN lb.platelets < 50 THEN 3
          WHEN lb.platelets < 100 THEN 2
          WHEN lb.platelets < 150 THEN 1
          ELSE 0
        END
        +
        CASE
          WHEN lb.bilirubin > 12 THEN 4
          WHEN lb.bilirubin > 6 THEN 3
          WHEN lb.bilirubin > 2 THEN 2
          WHEN lb.bilirubin > 1.2 THEN 1
          ELSE 0
        END
      )::INTEGER AS sofa_total

    FROM ns s
    LEFT JOIN tr USING (stay_id, bin)
    LEFT JOIN lb USING (stay_id, bin);
    """)

    con.execute("""
    CREATE TABLE elig AS
    SELECT DISTINCT icu.stay_id
    FROM icu
    JOIN infection USING (hadm_id)
    JOIN sofa ON icu.stay_id = sofa.stay_id
    WHERE sofa.sofa_total >= 2;
    """)

    con.execute(f"""
    COPY (
    SELECT
      s.stay_id,
      s.bin,

      (
        CASE
          WHEN COALESCE(tr.vaso, 0) <= 0 THEN 0
          WHEN COALESCE(tr.vaso, 0) <= 0.08 THEN 1
          WHEN COALESCE(tr.vaso, 0) <= 0.22 THEN 2
          WHEN COALESCE(tr.vaso, 0) <= 0.45 THEN 3
          ELSE 4
        END
        * 5
        +
        CASE
          WHEN COALESCE(tr.fluid, 0) <= 0 THEN 0
          WHEN COALESCE(tr.fluid, 0) <= 50 THEN 1
          WHEN COALESCE(tr.fluid, 0) <= 180 THEN 2
          WHEN COALESCE(tr.fluid, 0) <= 530 THEN 3
          ELSE 4
        END
      )::INTEGER AS action,

      COALESCE(tr.vaso, 0.0)::DOUBLE AS vaso_amount,
      COALESCE(tr.fluid, 0.0)::DOUBLE AS fluid_amount,
      COALESCE(sofa.sofa_total, 0)::INTEGER AS sofa_total,

      COALESCE(s.HR,   0.0)::DOUBLE AS s_HR,
      COALESCE(s.TEMP, 0.0)::DOUBLE AS s_TEMP,
      COALESCE(s.SPO2, 0.0)::DOUBLE AS s_SPO2,
      COALESCE(s.SBP,  0.0)::DOUBLE AS s_SBP,
      COALESCE(s.DBP,  0.0)::DOUBLE AS s_DBP,
      COALESCE(s.MBP,  0.0)::DOUBLE AS s_MBP,
      COALESCE(pat.age, 0.0)::DOUBLE AS s_age,
      COALESCE(pat.sex, 0.0)::DOUBLE AS s_sex,

      COALESCE(s.next_HR,   0.0)::DOUBLE AS s_next_HR,
      COALESCE(s.next_TEMP, 0.0)::DOUBLE AS s_next_TEMP,
      COALESCE(s.next_SPO2, 0.0)::DOUBLE AS s_next_SPO2,
      COALESCE(s.next_SBP,  0.0)::DOUBLE AS s_next_SBP,
      COALESCE(s.next_DBP,  0.0)::DOUBLE AS s_next_DBP,
      COALESCE(s.next_MBP,  0.0)::DOUBLE AS s_next_MBP,
      COALESCE(pat.age, 0.0)::DOUBLE AS s_next_age,
      COALESCE(pat.sex, 0.0)::DOUBLE AS s_next_sex,

      CASE WHEN s.next_bin IS NULL THEN 1 ELSE 0 END::INTEGER AS done,

      CASE
        WHEN s.next_bin IS NULL AND COALESCE(adm.hospital_expire_flag, 0) = 1 THEN -100.0
        WHEN s.next_bin IS NULL AND COALESCE(adm.hospital_expire_flag, 0) = 0 THEN  100.0
        ELSE 0.0
      END::DOUBLE AS reward

    FROM ns s
    JOIN elig USING (stay_id)
    LEFT JOIN tr USING (stay_id, bin)
    LEFT JOIN sofa USING (stay_id, bin)
    LEFT JOIN icu USING (stay_id)
    LEFT JOIN pat USING (subject_id)
    LEFT JOIN adm USING (hadm_id)
    ORDER BY s.stay_id, s.bin
    )
    TO '{out_file}' (FORMAT PARQUET);
    """)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default= r"C:\Users\20231942\Desktop\Central Folder\TUe\Year 3\Honors\Data Processing\mimic\raw", help="Path to mimic-iv-3.1 directory")
    ap.add_argument("--out", type=str, default= r"C:\Users\20231942\Desktop\Central Folder\TUe\Year 3\Honors\Code\CareQueue\mimic-iv-3.1\iql\Version 3\Processed\transitions.parquet", help="Output transitions parquet path")
    ap.add_argument("--bin_hours", type=int, default=4, help="Time bin size in hours (default 4)")
    ap.add_argument("--threads", type=int, default=8, help="DuckDB threads")
    ap.add_argument("--mem_limit", type=str, default="8GB", help="DuckDB memory limit, e.g. 8GB, 32GB")
    ap.add_argument("--tmp", type=str, default="./duckdb_tmp", help="DuckDB temp directory")
    args = ap.parse_args()

    base = Path(args.base).resolve()
    out_file = Path(args.out).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # We'll also write a state parquet next to the output
    out_state = out_file.with_name(out_file.stem.replace("transitions", "state") + ".parquet")

    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={args.threads};")
    con.execute(f"PRAGMA memory_limit='{args.mem_limit}';")
    con.execute(f"PRAGMA temp_directory='{Path(args.tmp).resolve()}';")

    print(f"[lite] BASE={base}")
    print(f"[lite] bin_hours={args.bin_hours} threads={args.threads} mem_limit={args.mem_limit}")
    print(f"[lite] building states -> {out_state}")
    build_states(con, base, out_state, args.bin_hours)

    print(f"[lite] building transitions -> {out_file}")
    build_transitions(con, base, out_state, out_file, args.bin_hours)

    print("[lite] Done.")


if __name__ == "__main__":
    main()