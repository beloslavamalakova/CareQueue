import numpy as np
import pandas as pd

def discretize_0_plus_quartiles(x: pd.Series) -> pd.Series:
    """
    0: x == 0 (or missing treated as 0)
    1-4: quartiles of x among x>0
    """
    x = x.fillna(0.0)
    pos = x[x > 0]
    if len(pos) == 0:
        return pd.Series(np.zeros(len(x), dtype=int), index=x.index)

    q = pos.quantile([0.25, 0.5, 0.75]).values  # three cut points
    # bins: (0)->0, (0,q1]->1, (q1,q2]->2, (q2,q3]->3, (q3,inf)->4
    def f(v: float) -> int:
        if v <= 0: return 0
        if v <= q[0]: return 1
        if v <= q[1]: return 2
        if v <= q[2]: return 3
        return 4

    return x.apply(f).astype(int)

def build_25_action_space(actions_cont_df: pd.DataFrame) -> pd.DataFrame:
    df = actions_cont_df.copy()

    df["fluid_level"] = discretize_0_plus_quartiles(df["fluid_ml_bin"])
    df["vaso_level"]  = discretize_0_plus_quartiles(df["norepi_equiv_max"])

    # 25 actions: (fluid_level 0..4) x (vaso_level 0..4)
    # First option is (0,0) -> action_id 0
    df["action_id"] = df["fluid_level"] * 5 + df["vaso_level"]

    return df[["stay_id","bin_idx","fluid_ml_bin","norepi_equiv_max","fluid_level","vaso_level","action_id"]]
