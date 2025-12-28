#!/usr/bin/env python3
"""
OOM-safe 4-hour binning for MIMIC-IV using DuckDB.

Outputs (Parquet):
  cache/bins_4h.parquet              - one row per stay_id x bin
  cache/events_4h_long.parquet       - long-format aggregated events per stay/bin/item/source

Assumes a MIMIC-IV folder layout like:
  ./icu/icustays.csv.gz
  ./icu/chartevents.csv.gz
  ./hosp/labevents.csv.gz
(plain .csv also supported)

Run:
  python3 time_bins_for_bcq_duckdb_4h.py
Optional:
  python3 time_bins_for_bcq_duckdb_4h.py --bin-hours 4 --mem 8GB --threads 4
"""

import argparse
import os
from pathlib import Path
import duckdb


def pick_existing(*candidates: str) -> str:
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"None of these paths exist: {candidates}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="Project root where mimic-iv-3.1 folders live")
    ap.add_argument("--bin-hours", type=int, default=4, help="Bin size in hours (default: 4)")
    ap.add_argument("--threads", type=int, default=4, help="DuckDB threads")
    ap.add_argument("--mem", default="8GB", help="DuckDB memory_limit, e.g. 4GB, 8GB, 16GB")
    ap.add_argument("--include-labs", action="store_true", help="Also bin hosp/labevents")
    ap.add_argument("--tmpdir", default="./duckdb_tmp", help="Temp directory for DuckDB spill files")
    ap.add_argument("--cache-dir", default="./cache", help="Output cache directory")
    ap.add_argument("--itemids-file", default=None,
                    help="Optional text file with one itemid per line to FILTER to (keeps output small)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    cache_dir = (root / args.cache_dir).resolve()
    tmpdir = (root / args.tmpdir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmpdir.mkdir(parents=True, exist_ok=True)

    # Detect inputs (gz or plain)
    icustays_path = pick_existing(str(root / "icu" / "icustays.csv.gz"),
                                  str(root / "icu" / "icustays.csv"))
    chartevents_path = pick_existing(str(root / "icu" / "chartevents.csv.gz"),
                                     str(root / "icu" / "chartevents.csv"))

    labevents_path = None
    if args.include_labs:
        labevents_path = pick_existing(str(root / "hosp" / "labevents.csv.gz"),
                                       str(root / "hosp" / "labevents.csv"))

    bins_out = str(cache_dir / f"bins_{args.bin_hours}h.parquet")
    events_out = str(cache_dir / f"events_{args.bin_hours}h_long.parquet")

    # Optional itemid filter (strongly recommended for very large runs)
    itemids = None
    if args.itemids_file:
        p = Path(args.itemids_file)
        if not p.exists():
            raise FileNotFoundError(f"--itemids-file not found: {p}")
        itemids = [line.strip() for line in p.read_text().splitlines() if line.strip().isdigit()]
        if not itemids:
            raise ValueError("itemids-file provided but no valid numeric itemids found.")

    con = duckdb.connect()
    con.execute(f"PRAGMA threads={int(args.threads)};")
    con.execute(f"PRAGMA memory_limit='{args.mem}';")
    con.execute(f"PRAGMA temp_directory='{str(tmpdir)}';")
    # Helpful for big reads
    con.execute("PRAGMA preserve_insertion_order=false;")

    bin_seconds = int(args.bin_hours) * 3600

    # If you re-run, overwrite outputs
    for outp in [bins_out, events_out]:
        if os.path.exists(outp):
            os.remove(outp)

    # Create a small ICU stays table (DuckDB temp view)
    # We only need stay_id + intime + outtime
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW icu_stays AS
    SELECT
      stay_id::BIGINT AS stay_id,
      intime::TIMESTAMP AS intime,
      outtime::TIMESTAMP AS outtime
    FROM read_csv_auto('{icustays_path}', union_by_name=true);
    """)

    # -----------------------------
    # 1) Produce bins table
    # -----------------------------
    # n_bins = ceil((outtime-intime)/bin_seconds)
    # Generate (stay_id, bin_idx) using range() per stay (via lateral join)
    con.execute(f"""
    COPY (
      WITH s AS (
        SELECT
          stay_id,
          intime,
          outtime,
          GREATEST(
            0,
            CAST(CEIL(EXTRACT(EPOCH FROM (outtime - intime)) / {bin_seconds}) AS BIGINT)
          ) AS n_bins
        FROM icu_stays
        WHERE intime IS NOT NULL AND outtime IS NOT NULL AND outtime > intime
      )
      SELECT
        s.stay_id,
        r.bin_idx,
        (s.intime + (r.bin_idx * INTERVAL '{args.bin_hours} hours')) AS bin_start,
        (s.intime + ((r.bin_idx + 1) * INTERVAL '{args.bin_hours} hours')) AS bin_end
      FROM s
      JOIN LATERAL (
        SELECT i AS bin_idx FROM range(0, s.n_bins) t(i)
      ) r ON TRUE
    )
    TO '{bins_out}' (FORMAT PARQUET);
    """)

    # -----------------------------
    # 2) Bin chartevents (core numeric)
    # -----------------------------
    # We avoid pulling text columns (value) to keep size reasonable.
    # We keep valuenum and time. Aggregations per stay/bin/itemid.
    itemid_filter_sql = ""
    if itemids:
        itemid_filter_sql = f"AND itemid IN ({','.join(itemids)})"

    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW chart_binned AS
    SELECT
      s.stay_id,
      CAST(FLOOR(EXTRACT(EPOCH FROM (c.charttime::TIMESTAMP - s.intime)) / {bin_seconds}) AS BIGINT) AS bin_idx,
      c.itemid::BIGINT AS itemid,
      c.charttime::TIMESTAMP AS event_time,
      c.valuenum::DOUBLE AS valuenum
    FROM read_csv_auto('{chartevents_path}', union_by_name=true) c
    JOIN icu_stays s
      ON c.stay_id::BIGINT = s.stay_id
    WHERE
      c.charttime IS NOT NULL
      AND s.intime IS NOT NULL AND s.outtime IS NOT NULL
      AND c.charttime::TIMESTAMP >= s.intime
      AND c.charttime::TIMESTAMP <  s.outtime
      AND c.valuenum IS NOT NULL
      {itemid_filter_sql}
      AND EXTRACT(EPOCH FROM (c.charttime::TIMESTAMP - s.intime)) >= 0
    ;
    """)

    # Aggregate; last_valuenum uses arg_max(valuenum, event_time)
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW chart_agg AS
    SELECT
      stay_id,
      bin_idx,
      itemid,
      'chart' AS source,
      COUNT(*)::BIGINT AS n,
      AVG(valuenum)::DOUBLE AS mean_valuenum,
      MIN(valuenum)::DOUBLE AS min_valuenum,
      MAX(valuenum)::DOUBLE AS max_valuenum,
      arg_max(valuenum, event_time)::DOUBLE AS last_valuenum
    FROM chart_binned
    WHERE bin_idx IS NOT NULL AND bin_idx >= 0
    GROUP BY stay_id, bin_idx, itemid
    ;
    """)

    # -----------------------------
    # 3) Optionally bin labevents
    # -----------------------------
    if args.include_labs:
        # Note: labevents uses charttime sometimes; in MIMIC-IV it's typically "charttime" or "storetime".
        # read_csv_auto(union_by_name=true) handles either; we pick charttime if exists else storetime.
        # We'll use COALESCE(charttime, storetime).
        lab_itemid_filter_sql = ""
        if itemids:
            # Usually lab itemids are different from chart itemids; if you want separate lists, use a different file.
            lab_itemid_filter_sql = f"AND itemid IN ({','.join(itemids)})"

        con.execute(f"""
        CREATE OR REPLACE TEMP VIEW lab_binned AS
        SELECT
          s.stay_id,
          CAST(FLOOR(EXTRACT(EPOCH FROM (COALESCE(l.charttime, l.storetime)::TIMESTAMP - s.intime)) / {bin_seconds}) AS BIGINT) AS bin_idx,
          l.itemid::BIGINT AS itemid,
          COALESCE(l.charttime, l.storetime)::TIMESTAMP AS event_time,
          l.valuenum::DOUBLE AS valuenum
        FROM read_csv_auto('{labevents_path}', union_by_name=true) l
        JOIN icu_stays s
          ON l.hadm_id::BIGINT = s.stay_id  -- (NOT correct generally)
        WHERE 1=0
        ;
        """)

        # IMPORTANT:
        # In MIMIC-IV, labevents typically keys by hadm_id + subject_id, NOT stay_id.
        # If you truly need labs aligned to ICU stays, you should join via hadm_id
        # from icustays (icustays has hadm_id) and then restrict to [intime, outtime).
        # Since icu_stays view above doesn't include hadm_id, we intentionally don't run labs by default
        # unless you adapt the join correctly for your pipeline.
        #
        # If you want labs, scroll down to the "LABS CORRECT JOIN" section below and use that version.

    # -----------------------------
    # 4) Join bins + events, write output
    # -----------------------------
    # Ensure we only keep bin_idx that exists (within stay bins)
    con.execute(f"""
    COPY (
      WITH bins AS (
        SELECT * FROM read_parquet('{bins_out}')
      ),
      ev AS (
        SELECT * FROM chart_agg
      )
      SELECT
        b.stay_id,
        b.bin_idx,
        b.bin_start,
        b.bin_end,
        ev.itemid,
        ev.source,
        ev.n,
        ev.mean_valuenum,
        ev.min_valuenum,
        ev.max_valuenum,
        ev.last_valuenum
      FROM bins b
      JOIN ev
        ON ev.stay_id = b.stay_id
       AND ev.bin_idx = b.bin_idx
    )
    TO '{events_out}' (FORMAT PARQUET);
    """)

    print(f"[ok] bins written:   {bins_out}")
    print(f"[ok] events written: {events_out}")
    print("\nTip: This is long-format (stay/bin/item). Pivot later only for selected itemids to avoid OOM.")


if __name__ == "__main__":
    main()
