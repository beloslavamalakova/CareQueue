import pandas as pd

df = pd.read_parquet(r"C:\Users\20231942\Desktop\Central Folder\TUe\Year 3\Honors\Code\CareQueue\sepsis_iql_actionvec_transitions.parquet")
print(df.shape)
print(df.head())

state_cols = [c for c in df.columns if c.startswith("s_")]
avg_nan = df[state_cols].isna().mean().mean()
print("Avg NaN fraction across state cols:", avg_nan)