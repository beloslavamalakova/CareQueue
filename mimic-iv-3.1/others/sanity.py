import duckdb
con=duckdb.connect()
print(con.execute("""
  SELECT * FROM read_parquet('cache/states_4h_simple.parquet')
  WHERE stay_id=30000153
  ORDER BY bin_idx
  LIMIT 20
""").df())
