import duckdb
from pathlib import Path

STATE_FILE = Path("/home/20243009/scripts/state_4h.parquet")
PROCEDURE_FILE = Path("/home/20243009/mimic-iv-3.1/icu/procedureevents.csv.gz")
ADMISSIONS_FILE = Path("/home/20243009/mimic-iv-3.1/hosp/admissions.csv.gz")
OUT_FILE = Path("/home/20243009/scripts/transitions_duckdb.parquet")

# Not sure what it means with database=':memory:', but I think it might just point to the database which is in memory...
con = duckdb.connect(database=':memory:')

# Creating tables now for the files listed above
con.execute(f"""
    CREATE TABLE states AS SELECT * FROM read_parquet('{STATE_FILE}');
""")
con.execute(f"""
    CREATE TABLE procedureevents AS SELECT * FROM read_csv_auto('{PROCEDURE_FILE}');
""")
con.execute(f"""
    CREATE TABLE admissions AS SELECT hadm_id, hospital_expire_flag FROM read_csv_auto('{ADMISSIONS_FILE}');
""")

con.execute("""
    CREATE TABLE next_states AS
    SELECT 
        stay_id,
        bin, 
        HR, TEMP, SPO2, SBP, DBP, MBP,
        LEAD(HR) OVER (PARTITION BY stay_id ORDER BY bin) AS next_HR,
        LEAD(TEMP) OVER (PARTITION BY stay_id ORDER BY bin) AS next_TEMP,
        LEAD(SPO2) OVER (PARTITION BY stay_id ORDER BY bin) AS next_SPO2,
        LEAD(SBP) OVER (PARTITION BY stay_id ORDER BY bin) AS next_SBP,
        LEAD(DBP) OVER (PARTITION BY stay_id ORDER BY bin) AS next_DBP,
        LEAD(MBP) OVER (PARTITION BY stay_id ORDER BY bin) AS next_MBP,
        LEAD(bin) OVER (PARTITION BY stay_id ORDER BY bin) AS next_bin
    FROM states
""")

con.execute("""
    CREATE TABLE transitions AS
    SELECT
        s.stay_id,
        s.bin,
        COALESCE(
            CASE 
                WHEN MAX(CASE WHEN p.itemid IN (225794) THEN 1 ELSE 0 END) = 1 THEN 1
                WHEN MAX(CASE WHEN p.itemid IN (224263,224268,225752) THEN 1 ELSE 0 END) = 1 THEN 2
                WHEN MAX(CASE WHEN p.itemid IN (229351) THEN 1 ELSE 0 END) = 1 THEN 3
                WHEN MAX(CASE WHEN p.itemid IN (227194) THEN 1 ELSE 0 END) = 1 THEN 4
                WHEN MAX(CASE WHEN p.itemid IN (225802) THEN 1 ELSE 0 END) = 1 THEN 5
                ELSE 0
            END, 0) AS action,
        s.HR AS s_HR,
        s.TEMP AS s_TEMP,
        s.SPO2 AS s_SPO2,
        s.SBP AS s_SBP,
        s.DBP AS s_DBP,
        s.MBP AS s_MBP,
        COALESCE(s.next_HR, 0) AS s_next_HR,
        COALESCE(s.next_TEMP, 0) AS s_next_TEMP,
        COALESCE(s.next_SPO2, 0) AS s_next_SPO2,
        COALESCE(s.next_SBP, 0) AS s_next_SBP,
        COALESCE(s.next_DBP, 0) AS s_next_DBP,
        COALESCE(s.next_MBP, 0) AS s_next_MBP,
        CASE WHEN s.next_bin IS NULL THEN 1 ELSE 0 END AS done,
        CASE WHEN s.next_bin IS NULL AND COALESCE(a.hospital_expire_flag,0)=1 THEN -1
             WHEN s.next_bin IS NULL AND COALESCE(a.hospital_expire_flag,0)=0 THEN 1
             ELSE 0
        END AS reward
    FROM next_states s
    LEFT JOIN procedureevents p
        ON s.stay_id = p.stay_id AND p.starttime >= s.bin AND p.starttime < COALESCE(s.next_bin, s.bin+1)
    LEFT JOIN admissions a
        ON s.stay_id = a.hadm_id
    GROUP BY s.stay_id, s.bin, s.HR, s.TEMP, s.SPO2, s.SBP, s.DBP, s.MBP,
             s.next_HR, s.next_TEMP, s.next_SPO2, s.next_SBP, s.next_DBP, s.next_MBP,
             s.next_bin, a.hospital_expire_flag
""")

con.execute(f"COPY transitions TO '{OUT_FILE}' (FORMAT PARQUET)")
print(f"Saved transitions to {OUT_FILE}")