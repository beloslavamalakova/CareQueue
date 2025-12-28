import os
import pandas as pd

HOSP_ROOT = os.path.expanduser("~/Downloads/mimic-iv-3.1")
HOSP_DATA_RAW = os.path.join(HOSP_ROOT, "icu")              # <- exists
HOSP_DATA_INTERIM = os.path.join(HOSP_ROOT, "interim", "icu")


def ensure_dirs():
    os.makedirs(HOSP_DATA_INTERIM, exist_ok=True)


def extract_first_1000_chartevents():
    if not os.path.isdir(HOSP_DATA_RAW):
        print(f"HOSP raw data folder not found: {HOSP_DATA_RAW}")
        return

    chartevents_path = None
    for fname in os.listdir(HOSP_DATA_RAW):
        if fname == "procedureevents.csv.gz":
            chartevents_path = os.path.join(HOSP_DATA_RAW, fname)
            break

    if chartevents_path is None:
        print("procedureevents.csv.gz not found.")
        return

    print(f"\nFound file: {chartevents_path}")

    chunksize = 200_000
    collected = []
    total_rows = 0

    try:
        for chunk in pd.read_csv(chartevents_path, chunksize=chunksize):
            needed = 1000 - total_rows
            if needed <= 0:
                break

            # Take rows by position only (no filtering)
            take = chunk.iloc[:needed]
            collected.append(take)

            total_rows += len(take)
            print(f"Collected rows: {total_rows}")

            if total_rows >= 1000:
                break

    except Exception as e:
        print(f"Error reading procedureevents.csv.gz: {e}")
        return

    if not collected:
        print("No rows collected.")
        return

    final_df = pd.concat(collected, ignore_index=True)

    out_path = os.path.join(HOSP_DATA_INTERIM, "procedureevents_first_1000.csv")
    final_df.to_csv(out_path, index=False)

    print(f"\nDone. Saved first 1000 rows to: {out_path}")


if __name__ == "__main__":
    ensure_dirs()
    extract_first_1000_chartevents()
