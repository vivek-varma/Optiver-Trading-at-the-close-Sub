"""Scratch harness for feature-engineering research (Optiver Close).

Provides:
  load_base()            -> df with base+temporal features, cached to parquet
  cv_skill(df, extra)    -> 3-fold walk-forward LGB-solo skill for feature set
  blend_gate(df, extra)  -> single 20% holdout, blend with saved NN preds
"""
import os
import numpy as np
import pandas as pd
import lightgbm as lgb

DIR = "/Users/vivekn/Code/Kaggle/Optiver_Close"
CACHE = os.path.join(DIR, "scratch_base_temporal.parquet")

LGB_PARAMS = dict(
    objective="mae", n_estimators=500, learning_rate=0.03, num_leaves=31,
    min_child_samples=1000, reg_alpha=1.0, reg_lambda=1.0, subsample=0.7,
    subsample_freq=1, colsample_bytree=0.7, random_state=42, n_jobs=4, verbose=-1,
)

# columns present after build_features + add_temporal (base+temporal)
def _base_temporal_cols(df):
    exclude = {"date_id", "seconds_in_bucket", "target", "time_id", "row_id"}
    # keep stock_id (categorical), drop other ids
    return [c for c in df.columns if c not in exclude]


def load_base():
    if os.path.exists(CACHE):
        return pd.read_parquet(CACHE)
    import features as F
    raw = pd.read_csv(os.path.join(DIR, "Data/train.csv"))
    df = F.build_features(raw)
    df = F.add_temporal_features(df)
    df.to_parquet(CACHE)
    return df


def _skill(y, p):
    zero = np.mean(np.abs(y))
    mae = np.mean(np.abs(y - p))
    return (zero - mae) / zero * 100.0


def _fit_predict(tr, va, feat_cols):
    Xtr, ytr = tr[feat_cols], tr["target"].values
    Xva = va[feat_cols]
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(Xtr, ytr, categorical_feature=["stock_id"])
    return m.predict(Xva)


def cv_skill(df, base_cols, extra_cols, val_days=50, n_folds=3):
    """Return (base_skills, full_skills) per fold. df must have target non-null filtered."""
    dates = np.sort(df.date_id.unique())
    n = len(dates)
    df = df[df.target.notna()].copy()
    df["stock_id"] = df["stock_id"].astype("category")
    base_res, full_res = [], []
    for k in range(n_folds):
        val_end_idx = n - 1 - (n_folds - 1 - k) * val_days
        val_start_idx = val_end_idx - val_days + 1
        vstart, vend = dates[val_start_idx], dates[val_end_idx]
        tr = df[df.date_id < vstart]
        va = df[(df.date_id >= vstart) & (df.date_id <= vend)]
        pb = _fit_predict(tr, va, base_cols)
        pf = _fit_predict(tr, va, base_cols + extra_cols)
        base_res.append(_skill(va.target.values, pb))
        full_res.append(_skill(va.target.values, pf))
    return base_res, full_res


def cv_one(df, feat_cols, val_days=50, n_folds=3):
    """Per-fold LGB-solo skill for a single feature list."""
    dates = np.sort(df.date_id.unique())
    n = len(dates)
    df = df[df.target.notna()].copy()
    df["stock_id"] = df["stock_id"].astype("category")
    res = []
    for k in range(n_folds):
        val_end_idx = n - 1 - (n_folds - 1 - k) * val_days
        val_start_idx = val_end_idx - val_days + 1
        vstart, vend = dates[val_start_idx], dates[val_end_idx]
        tr = df[df.date_id < vstart]
        va = df[(df.date_id >= vstart) & (df.date_id <= vend)]
        p = _fit_predict(tr, va, feat_cols)
        res.append(_skill(va.target.values, p))
    return res


def blend_gate(df, base_cols, extra_cols):
    """Retrain LGB on train portion, predict 20% val, blend with saved NN preds."""
    dates = np.sort(df.date_id.unique())
    n = len(dates)
    cutoff = dates[-int(0.2 * n)]
    df = df.copy()
    df["stock_id"] = df["stock_id"].astype("category")
    tr = df[(df.date_id < cutoff) & (df.target.notna())]
    va = df[df.date_id >= cutoff].copy()

    feat = base_cols + extra_cols
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(tr[feat], tr["target"].values, categorical_feature=["stock_id"])
    va["lgb"] = m.predict(va[feat])
    va["row_id"] = va.date_id.astype(str) + "_" + va.seconds_in_bucket.astype(str) + "_" + va.stock_id.astype(str)

    gru = pd.read_parquet(os.path.join(DIR, "preds_gru_val.parquet"))[["row_id", "target", "gru"]]
    tr_ = pd.read_parquet(os.path.join(DIR, "preds_transformer_val.parquet"))[["row_id", "transformer"]]
    m2 = va[["row_id", "lgb"]].merge(gru, on="row_id").merge(tr_, on="row_id")
    m2 = m2[m2.target.notna()].copy()

    def zerosum(col):
        # subtract per-(date_id, seconds_in_bucket) mean
        rid = m2.row_id.str.split("_", expand=True)
        key = rid[0] + "_" + rid[1]
        return m2[col] - m2.groupby(key)[col].transform("mean")

    m2["blend"] = 0.3 * m2.lgb + 0.1 * m2.gru + 0.6 * m2.transformer
    # zero-sum post-process on blend
    rid = m2.row_id.str.split("_", expand=True)
    m2["dk"] = rid[0].astype(str) + "_" + rid[1].astype(str)
    m2["blend_zs"] = m2.blend - m2.groupby("dk")["blend"].transform("mean")

    res = {
        "blend_skill": _skill(m2.target.values, m2.blend_zs.values),
        "lgb_solo_skill": _skill(m2.target.values, m2.lgb.values),
        "corr_lgb_gru": np.corrcoef(m2.lgb, m2.gru)[0, 1],
        "corr_lgb_tr": np.corrcoef(m2.lgb, m2.transformer)[0, 1],
    }
    return res, m2
