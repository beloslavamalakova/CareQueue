#!/usr/bin/env python3
"""
Create behavior cloning parquet for bc_discrete.py.

Outputs columns:
  stay_id, bin, action, reward, done,
  vaso_bin, fluid_bin, vaso_amount, fluid_amount,
  s_*, s_next_*

Includes static state features:
  s_age, s_sex
  s_next_age, s_next_sex

sex encoding:
  male = 1.0
  female = 0.0
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


def build_transitions(
    con: duckdb.DuckDBPyConnection,
    base: Path,
    state_file: Path,
    out_file: Path,
    bin_hours: int,
):
    inputevents_file = base / "icu" / "inputevents.csv.gz"
    admissions_file = base / "hosp" / "admissions.csv.gz"
    patients_file = base / "hosp" / "patients.csv.gz"
    icustays_file = base / "icu" / "icustays.csv.gz"

    vaso_sql = ",".join(str(x) for x in VASO_ITEMIDS)
    fluid_sql = ",".join(str(x) for x in FLUID_ITEMIDS)
    all_treatment_sql = ",".join(str(x) for x in sorted(set(VASO_ITEMIDS + FLUID_ITEMIDS)))

    con.execute(f"""
    CREATE TABLE states AS
    SELECT * FROM read_parquet('{state_file}');
    """)

    con.execute(f"""
    CREATE TABLE icustays AS
    SELECT
        stay_id::BIGINT AS stay_id,
        subject_id::BIGINT AS subject_id,
        hadm_id::BIGINT AS hadm_id,
        intime::TIMESTAMP AS intime
    FROM read_csv_auto('{icustays_file}', union_by_name=true)
    WHERE stay_id IS NOT NULL
      AND subject_id IS NOT NULL
      AND hadm_id IS NOT NULL
      AND intime IS NOT NULL;
    """)

    con.execute(f"""
    CREATE TABLE patients AS
    SELECT
        subject_id::BIGINT AS subject_id,
        anchor_age::DOUBLE AS age,
        CASE
            WHEN gender = 'M' THEN 1.0
            WHEN gender = 'F' THEN 0.0
            ELSE NULL
        END::DOUBLE AS sex
    FROM read_csv_auto('{patients_file}', union_by_name=true)
    WHERE subject_id IS NOT NULL;
    """)

    con.execute(f"""
    CREATE TABLE admissions AS
    SELECT
        hadm_id::BIGINT AS hadm_id,
        hospital_expire_flag::INTEGER AS hospital_expire_flag
    FROM read_csv_auto('{admissions_file}', union_by_name=true)
    WHERE hadm_id IS NOT NULL;
    """)

    con.execute(f"""
    CREATE TABLE inputevents AS
    SELECT
        stay_id::BIGINT AS stay_id,
        starttime::TIMESTAMP AS starttime,
        endtime::TIMESTAMP AS endtime,
        itemid::BIGINT AS itemid,
        amount::DOUBLE AS amount,
        rate::DOUBLE AS rate,
        patientweight::DOUBLE AS patientweight
    FROM read_csv_auto('{inputevents_file}', union_by_name=true)
    WHERE stay_id IS NOT NULL
      AND starttime IS NOT NULL
      AND itemid IN ({all_treatment_sql});
    """)

    con.execute("""
    CREATE TABLE next_states AS
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
    CREATE TABLE treatments AS
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
      ), 0.0) AS vaso_amount,

      COALESCE(SUM(
        CASE
          WHEN ie.itemid IN ({fluid_sql})
          THEN COALESCE(ie.amount, 0.0)
          ELSE 0.0
        END
      ), 0.0) AS fluid_amount

    FROM next_states s
    LEFT JOIN icustays icu
      ON s.stay_id = icu.stay_id
    LEFT JOIN inputevents ie
      ON s.stay_id = ie.stay_id
     AND ie.starttime >= icu.intime + (s.bin * INTERVAL '{bin_hours} hours')
     AND ie.starttime <  icu.intime + ((s.bin + 1) * INTERVAL '{bin_hours} hours')

    GROUP BY s.stay_id, s.bin;
    """)

    con.execute("""
    CREATE TABLE treatment_actions AS
    SELECT
      stay_id,
      bin,
      vaso_amount,
      fluid_amount,

      CASE
        WHEN vaso_amount <= 0 THEN 0
        WHEN vaso_amount <= 0.08 THEN 1
        WHEN vaso_amount <= 0.22 THEN 2
        WHEN vaso_amount <= 0.45 THEN 3
        ELSE 4
      END::INTEGER AS vaso_bin,

      CASE
        WHEN fluid_amount <= 0 THEN 0
        WHEN fluid_amount <= 50 THEN 1
        WHEN fluid_amount <= 180 THEN 2
        WHEN fluid_amount <= 530 THEN 3
        ELSE 4
      END::INTEGER AS fluid_bin

    FROM treatments;
    """)

    con.execute(f"""
    CREATE TABLE transitions AS
    SELECT
      s.stay_id,
      s.bin,

      COALESCE((ta.vaso_bin * 5 + ta.fluid_bin), 0)::INTEGER AS action,

      COALESCE(ta.vaso_bin, 0)::INTEGER AS vaso_bin,
      COALESCE(ta.fluid_bin, 0)::INTEGER AS fluid_bin,
      COALESCE(ta.vaso_amount, 0.0)::DOUBLE AS vaso_amount,
      COALESCE(ta.fluid_amount, 0.0)::DOUBLE AS fluid_amount,

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

      CASE
        WHEN s.next_bin IS NULL THEN 1
        ELSE 0
      END::INTEGER AS done,

      CASE
        WHEN s.next_bin IS NULL AND COALESCE(a.hospital_expire_flag, 0) = 1 THEN -1.0
        WHEN s.next_bin IS NULL AND COALESCE(a.hospital_expire_flag, 0) = 0 THEN  1.0
        ELSE 0.0
      END::DOUBLE AS reward

    FROM next_states s
    LEFT JOIN treatment_actions ta
      ON s.stay_id = ta.stay_id
     AND s.bin = ta.bin
    LEFT JOIN icustays icu
      ON s.stay_id = icu.stay_id
    LEFT JOIN patients pat
      ON icu.subject_id = pat.subject_id
    LEFT JOIN admissions a
      ON icu.hadm_id = a.hadm_id
    ORDER BY s.stay_id, s.bin;
    """)

    con.execute(f"""
    COPY transitions TO '{out_file}' (FORMAT PARQUET);
    """)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--bin_hours", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--mem_limit", type=str, default="8GB")
    ap.add_argument("--tmp", type=str, default="./duckdb_tmp")
    args = ap.parse_args()

    base = Path(args.base).resolve()
    out_file = Path(args.out).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(args.tmp).resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    out_state = out_file.with_name(out_file.stem + "_states_tmp.parquet")

    con = duckdb.connect(database=":memory:")
    con.execute(f"PRAGMA threads={args.threads};")
    con.execute(f"PRAGMA memory_limit='{args.mem_limit}';")
    con.execute(f"PRAGMA temp_directory='{tmp_dir}';")
    con.execute("PRAGMA preserve_insertion_order=false;")

    print(f"[bc-prep] BASE={base}")
    print(f"[bc-prep] OUT={out_file}")
    print(f"[bc-prep] building states -> {out_state}")
    build_states(con, base, out_state, args.bin_hours)

    print(f"[bc-prep] building transitions -> {out_file}")
    build_transitions(con, base, out_state, out_file, args.bin_hours)

    print("[bc-prep] Done.")


if __name__ == "__main__":
    main()