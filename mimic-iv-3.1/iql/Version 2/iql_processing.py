#!/usr/bin/env python3
"""
Lite MIMIC processing using DuckDB (fast).

Outputs a transitions parquet with columns:
  stay_id, bin, action, reward, done,
  s_*, s_next_*

Action space (0..5) is procedure-based (same mapping as your friend's script):
  0: none
  1: itemid 225794
  2: itemid 224263,224268,225752
  3: itemid 229351
  4: itemid 227194
  5: itemid 225802

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

# 6-action mapping (procedureevents itemid -> action idx)
ACTION_RULES = [
    (1, (225794,)),
    (2, (224263, 224268, 225752)),
    (3, (229351,)),
    (4, (227194,)),
    (5, (225802,)),
]


def build_states(con: duckdb.DuckDBPyConnection, base: Path, out_state: Path, bin_hours: int):
    chartevents = base / "icu" / "chartevents.csv.gz"
    icustays = base / "icu" / "icustays.csv.gz"

    item_list = ",".join(str(v) for v in STATE_ITEMIDS.values())

    # Build AVG(...) columns
    avg_cols = []
    for name, itemid in STATE_ITEMIDS.items():
        avg_cols.append(f"AVG(CASE WHEN ce.itemid = {itemid} THEN ce.valuenum END) AS {name}")
    avg_cols_sql = ",\n        ".join(avg_cols)

    query = f"""
    COPY (
        SELECT
            ce.stay_id,
            FLOOR(EXTRACT(EPOCH FROM (ce.charttime - icu.intime)) / ({bin_hours}*3600))::BIGINT AS bin,
            {avg_cols_sql}
        FROM read_csv_auto('{chartevents}', union_by_name=true) ce
        JOIN read_csv_auto('{icustays}', union_by_name=true) icu
          ON ce.stay_id = icu.stay_id
        WHERE ce.itemid IN ({item_list})
          AND ce.valuenum IS NOT NULL
          AND ce.charttime IS NOT NULL
          AND icu.intime IS NOT NULL
          AND ce.stay_id IS NOT NULL
        GROUP BY ce.stay_id, bin
        ORDER BY ce.stay_id, bin
    ) TO '{out_state}' (FORMAT PARQUET);
    """
    con.execute(query)


def build_transitions(con: duckdb.DuckDBPyConnection, base: Path, state_file: Path, out_file: Path, bin_hours: int):
    procedure_file = base / "icu" / "procedureevents.csv.gz"
    admissions_file = base / "hosp" / "admissions.csv.gz"
    icustays_file = base / "icu" / "icustays.csv.gz"

    con.execute(f"CREATE TABLE states AS SELECT * FROM read_parquet('{state_file}');")

    con.execute(f"""
    CREATE TABLE procedureevents AS
    SELECT stay_id::BIGINT AS stay_id,
           starttime::TIMESTAMP AS starttime,
           itemid::BIGINT AS itemid
    FROM read_csv_auto('{procedure_file}', union_by_name=true);
    """)

    con.execute(f"""
    CREATE TABLE admissions AS
    SELECT hadm_id::BIGINT AS hadm_id,
           hospital_expire_flag::INTEGER AS hospital_expire_flag
    FROM read_csv_auto('{admissions_file}', union_by_name=true);
    """)

    con.execute(f"""
    CREATE TABLE icustays AS
    SELECT stay_id::BIGINT AS stay_id,
           hadm_id::BIGINT AS hadm_id,
           intime::TIMESTAMP AS intime
    FROM read_csv_auto('{icustays_file}', union_by_name=true);
    """)

    # Next-state table (LEAD over bins)
    con.execute("""
    CREATE TABLE next_states AS
    SELECT
      stay_id,
      bin,
      HR, TEMP, SPO2, SBP, DBP, MBP,
      LEAD(HR)   OVER (PARTITION BY stay_id ORDER BY bin) AS next_HR,
      LEAD(TEMP) OVER (PARTITION BY stay_id ORDER BY bin) AS next_TEMP,
      LEAD(SPO2) OVER (PARTITION BY stay_id ORDER BY bin) AS next_SPO2,
      LEAD(SBP)  OVER (PARTITION BY stay_id ORDER BY bin) AS next_SBP,
      LEAD(DBP)  OVER (PARTITION BY stay_id ORDER BY bin) AS next_DBP,
      LEAD(MBP)  OVER (PARTITION BY stay_id ORDER BY bin) AS next_MBP,
      LEAD(bin)  OVER (PARTITION BY stay_id ORDER BY bin) AS next_bin
    FROM states;
    """)

    # Build CASE priority for action
    # NOTE: first matching rule wins (same as your friend's logic)
    action_case_lines = []
    for action_idx, itemids in ACTION_RULES:
        ids = ",".join(str(x) for x in itemids)
        action_case_lines.append(
            f"WHEN MAX(CASE WHEN p.itemid IN ({ids}) THEN 1 ELSE 0 END) = 1 THEN {action_idx}"
        )
    action_case_sql = "\n      ".join(action_case_lines)

    # Create transitions
    con.execute(f"""
    CREATE TABLE transitions AS
    SELECT
      s.stay_id,
      s.bin,

      COALESCE(
        CASE
          {action_case_sql}
          ELSE 0
        END, 0
      )::INTEGER AS action,

      s.HR   AS s_HR,
      s.TEMP AS s_TEMP,
      s.SPO2 AS s_SPO2,
      s.SBP  AS s_SBP,
      s.DBP  AS s_DBP,
      s.MBP  AS s_MBP,

      COALESCE(s.next_HR,   0) AS s_next_HR,
      COALESCE(s.next_TEMP, 0) AS s_next_TEMP,
      COALESCE(s.next_SPO2, 0) AS s_next_SPO2,
      COALESCE(s.next_SBP,  0) AS s_next_SBP,
      COALESCE(s.next_DBP,  0) AS s_next_DBP,
      COALESCE(s.next_MBP,  0) AS s_next_MBP,

      CASE WHEN s.next_bin IS NULL THEN 1 ELSE 0 END AS done,

      CASE
        WHEN s.next_bin IS NULL AND COALESCE(a.hospital_expire_flag, 0) = 1 THEN -1
        WHEN s.next_bin IS NULL AND COALESCE(a.hospital_expire_flag, 0) = 0 THEN  1
        ELSE 0
      END AS reward

    FROM next_states s
    LEFT JOIN icustays icu
      ON s.stay_id = icu.stay_id
    LEFT JOIN admissions a
      ON icu.hadm_id = a.hadm_id
    LEFT JOIN procedureevents p
      ON s.stay_id = p.stay_id
     AND p.starttime >= icu.intime + (s.bin * INTERVAL '{bin_hours} hours')
     AND p.starttime <  icu.intime + ((s.bin + 1) * INTERVAL '{bin_hours} hours')

    GROUP BY
      s.stay_id, s.bin, s.HR, s.TEMP, s.SPO2, s.SBP, s.DBP, s.MBP,
      s.next_HR, s.next_TEMP, s.next_SPO2, s.next_SBP, s.next_DBP, s.next_MBP,
      s.next_bin, a.hospital_expire_flag;
    """)

    con.execute(f"COPY transitions TO '{out_file}' (FORMAT PARQUET);")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default= r"C:\Users\20231942\Desktop\Central Folder\TUe\Year 3\Honors\Data Processing\mimic\raw", help="Path to mimic-iv-3.1 directory")
    ap.add_argument("--out", type=str, default= r"C:\Users\20231942\Desktop\Central Folder\TUe\Year 3\Honors\Code\CareQueue\mimic-iv-3.1\iql\Version 2\Processed", help="Output transitions parquet path")
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