"""Feature engineering for the Optiver 'Trading at the Close' dataset.

Shared by Baseline.ipynb (LightGBM) and LSTM.ipynb (sequence models).

Design notes
------------
* Every feature here is *leakage-safe*: it uses only information available at
  the current timestep, either from the same row or from other stocks at the
  SAME (date_id, seconds_in_bucket). Nothing looks forward in time.
* For sequence models we deliberately DROP hand-made temporal features
  (diff / momentum / cross-day history): a recurrent model learns those from
  the raw sequence itself. We keep the things it can't derive on its own —
  static price/size combinations and cross-sectional (peer-relative) features.
"""
import numpy as np
import pandas as pd

# Raw columns passed straight through as per-timestep features.
RAW_FEATURES = [
    "imbalance_size", "imbalance_buy_sell_flag", "reference_price", "matched_size",
    "far_price", "near_price", "bid_price", "bid_size", "ask_price", "ask_size", "wap",
]

# Cross-sectional features are built for each of these base columns.
_CROSS_BASE = ["imb_flag_size", "book_imb", "near_wap", "imb_to_matched"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features. Returns a sorted, index-reset copy."""
    df = df.sort_values(["stock_id", "date_id", "seconds_in_bucket"]).reset_index(drop=True).copy()

    # --- static per-row combinations ---------------------------------------
    df["mid_price"]       = (df.bid_price + df.ask_price) / 2
    df["spread"]          = df.ask_price - df.bid_price
    df["imb_flag_size"]   = df.imbalance_size * df.imbalance_buy_sell_flag   # signed imbalance
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

    # --- cross-sectional: a stock vs its peers at the SAME instant ----------
    # Safe from leakage: only uses rows sharing (date_id, seconds_in_bucket).
    grp = df.groupby(["date_id", "seconds_in_bucket"], observed=True)
    for col in _CROSS_BASE:
        df[f"{col}_rank"] = grp[col].rank(pct=True)
        df[f"{col}_z"]    = (df[col] - grp[col].transform("mean")) / grp[col].transform("std")

    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """The numeric per-timestep feature columns (everything the model sees
    except identifiers, the target, and stock_id — which gets its own
    embedding in the sequence model)."""
    exclude = {"stock_id", "date_id", "seconds_in_bucket", "target", "time_id", "row_id"}
    return [c for c in df.columns if c not in exclude]
