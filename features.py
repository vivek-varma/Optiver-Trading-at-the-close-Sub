"""Feature engineering for the Optiver 'Trading at the Close' dataset.

Two entry points:
  * build_features(df)       -> lean set for SEQUENCE models (RNN learns temporal
                                dynamics itself, so no hand-made diff/rolling).
  * build_features_gbdt(df)  -> rich set for TREE models: base + within-day
                                temporal + market-wide + cross-day history.

Leakage rule (holds for every feature here):
  a feature at time t uses only (a) the same row, (b) other stocks at the SAME
  (date_id, seconds_in_bucket), or (c) strictly EARLIER times/days. Never the
  future. The cross-day history features use only days < the current day.
"""
import numpy as np
import pandas as pd

RAW_FEATURES = [
    "imbalance_size", "imbalance_buy_sell_flag", "reference_price", "matched_size",
    "far_price", "near_price", "bid_price", "bid_size", "ask_price", "ask_size", "wap",
]
_CROSS_BASE = ["imb_flag_size", "book_imb", "near_wap", "imb_to_matched"]


# ── Base features (static + cross-sectional) — used by BOTH builders ──────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lean, leakage-safe base features. Sorted, index-reset copy."""
    df = df.sort_values(["stock_id", "date_id", "seconds_in_bucket"]).reset_index(drop=True).copy()

    df["mid_price"]       = (df.bid_price + df.ask_price) / 2
    df["spread"]          = df.ask_price - df.bid_price
    df["imb_flag_size"]   = df.imbalance_size * df.imbalance_buy_sell_flag
    df["imb_to_matched"]  = df.imb_flag_size / df.matched_size
    df["book_total"]      = df.bid_size + df.ask_size
    df["book_imb"]        = (df.bid_size - df.ask_size) / df.book_total
    df["auction_vs_book"] = df.matched_size / df.book_total
    df["ref_wap"]         = df.reference_price - df.wap
    df["near_wap"]        = df.near_price - df.wap
    df["far_wap"]         = df.far_price - df.wap
    df["far_near"]        = df.far_price - df.near_price
    df["ref_depth"]       = (df.reference_price - df.bid_price) / df.spread
    df["wap_depth"]       = (df.wap - df.bid_price) / df.spread
    df["auction_frac"]    = df.seconds_in_bucket / 540

    grp = df.groupby(["date_id", "seconds_in_bucket"], observed=True)
    for col in _CROSS_BASE:
        df[f"{col}_rank"] = grp[col].rank(pct=True)
        df[f"{col}_z"]    = (df[col] - grp[col].transform("mean")) / grp[col].transform("std")
    return df


# ── Within-day temporal dynamics (backward-looking only) ─────────────────
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """diff / momentum / rolling stats within each (stock, day). Uses only
    PAST steps of the same auction -> leakage-safe. Trees can't derive these."""
    g = df.groupby(["stock_id", "date_id"], observed=True)

    # one-step changes
    for col in ["wap", "imbalance_size", "imb_flag_size", "matched_size",
                "reference_price", "book_imb"]:
        df[f"{col}_d1"] = g[col].diff()

    # multi-step momentum (change vs 6 steps = 60s ago)
    df["wap_mom6"] = df["wap"] - g["wap"].shift(6)
    df["imb_mom6"] = df["imb_flag_size"] - g["imb_flag_size"].shift(6)

    # rolling realized vol / smoothed imbalance (min_periods=1 -> no NaN storm early)
    df["wap_rvol6"]     = g["wap"].transform(lambda s: s.rolling(6, min_periods=1).std())
    df["imb_rmean6"]    = g["imb_flag_size"].transform(lambda s: s.rolling(6, min_periods=1).mean())
    df["bookimb_rmean6"]= g["book_imb"].transform(lambda s: s.rolling(6, min_periods=1).mean())
    return df


# ── Market-wide (global) state at each instant ───────────────────────────
def add_global_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates across ALL stocks at the same (date, second). Because the
    target is index-relative, the market-wide level/dispersion is informative.
    Same-instant only -> leakage-safe."""
    g = df.groupby(["date_id", "seconds_in_bucket"], observed=True)
    df["mkt_imb_mean"]      = g["imb_flag_size"].transform("mean")
    df["mkt_imb_std"]       = g["imb_flag_size"].transform("std")
    df["mkt_bookimb_mean"]  = g["book_imb"].transform("mean")
    df["mkt_spread_mean"]   = g["spread"].transform("mean")
    df["mkt_matched_mean"]  = g["matched_size"].transform("mean")
    return df


# ── Cross-day per-stock history (strictly prior days) ────────────────────
def add_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per-stock daily aggregates, shifted so day d sees only days < d.
    Captures 'is this stock chronically volatile / imbalanced?'. Uses target
    of PAST days (available live via revealed_targets)."""
    daily = (df.groupby(["stock_id", "date_id"], observed=True)
               .agg(day_abs_tgt=("target", lambda s: s.abs().mean()),
                    day_wap_vol=("wap", "std"),
                    day_imb_mean=("imb_flag_size", "mean"))
               .reset_index()
               .sort_values(["stock_id", "date_id"]))

    gs = daily.groupby("stock_id", observed=True)
    for col in ["day_abs_tgt", "day_wap_vol", "day_imb_mean"]:
        daily[f"{col}_prev"] = gs[col].shift(1)                                   # yesterday
        daily[f"{col}_r5"]   = gs[col].transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())

    hist_cols = [c for c in daily.columns if c.endswith(("_prev", "_r5"))]
    return df.merge(daily[["stock_id", "date_id"] + hist_cols],
                    on=["stock_id", "date_id"], how="left")


def build_features_gbdt(df: pd.DataFrame) -> pd.DataFrame:
    """Feature set for tree models: base + within-day temporal dynamics.

    Family-attribution CV (3-fold) showed temporal carries the ENTIRE gain
    (base 1.78% -> base+temporal 2.01% skill), while add_global_features and
    add_historical_features added nothing and slightly diluted it. They're kept
    in this module for reference/experimentation but excluded from the default
    builder — temporal is also the only family that's cheap to compute live
    (within-day history only; no cross-day revealed_targets store needed).
    """
    df = build_features(df)
    df = add_temporal_features(df)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric feature columns: everything except identifiers, the target, and
    stock_id (which gets its own embedding in the sequence model)."""
    exclude = {"stock_id", "date_id", "seconds_in_bucket", "target", "time_id", "row_id"}
    return [c for c in df.columns if c not in exclude]
