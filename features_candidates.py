"""Candidate feature families for FE research. Each function takes a df that
already has base+temporal columns, adds new columns IN PLACE, and returns the
list of new column names. Leakage-safe: same-row, same-(date,second) peers, or
strictly earlier steps/days only.
"""
import numpy as np
import pandas as pd


def add_multiwindow(df):
    """Family 1: rolling mean/std & momentum over windows {3,12}; EWMA span5."""
    g = df.groupby(["stock_id", "date_id"], observed=True)
    new = []
    for w in (3, 12):
        df[f"wap_rvol{w}"] = g["wap"].transform(lambda s: s.rolling(w, min_periods=1).std())
        df[f"imb_rmean{w}"] = g["imb_flag_size"].transform(lambda s: s.rolling(w, min_periods=1).mean())
        df[f"wap_mom{w}"] = df["wap"] - g["wap"].shift(w)
        df[f"imb_mom{w}"] = df["imb_flag_size"] - g["imb_flag_size"].shift(w)
        new += [f"wap_rvol{w}", f"imb_rmean{w}", f"wap_mom{w}", f"imb_mom{w}"]
    df["wap_ewma5"] = g["wap"].transform(lambda s: s.ewm(span=5, min_periods=1).mean())
    df["imb_ewma5"] = g["imb_flag_size"].transform(lambda s: s.ewm(span=5, min_periods=1).mean())
    new += ["wap_ewma5", "imb_ewma5"]
    return new


def add_flips(df):
    """Family 2: imbalance acceleration, sign-flip indicator, running flip count."""
    g = df.groupby(["stock_id", "date_id"], observed=True)
    new = []
    # acceleration = diff of diff of imb_flag_size (d1 already exists)
    df["imb_accel"] = g["imb_flag_size_d1"].diff()
    # sign-flip of the buy/sell flag vs previous step
    prev_flag = g["imbalance_buy_sell_flag"].shift(1)
    df["imb_flag_flip"] = (df["imbalance_buy_sell_flag"] != prev_flag).astype(float)
    df.loc[prev_flag.isna(), "imb_flag_flip"] = 0.0  # first step: no prior -> 0
    # running count of flips so far in the day (excludes current -> use cumsum shifted)
    df["imb_flip_count"] = g["imb_flag_flip"].cumsum()
    new += ["imb_accel", "imb_flag_flip", "imb_flip_count"]
    return new


def add_xs_temporal(df):
    """Family 3: cross-sectional rank/z-score of temporal features across stocks
    at each (date, second)."""
    grp = df.groupby(["date_id", "seconds_in_bucket"], observed=True)
    new = []
    for col in ["imb_mom6", "wap_rvol6", "wap_mom6", "imb_rmean6"]:
        df[f"{col}_xrank"] = grp[col].rank(pct=True)
        mean = grp[col].transform("mean")
        std = grp[col].transform("std")
        df[f"{col}_xz"] = (df[col] - mean) / std
        new += [f"{col}_xrank", f"{col}_xz"]
    return new


def add_microstructure(df):
    """Family 5: size-weighted micro-price vs wap, relative spread, near/far
    convergence, log-return realized vol. All same-row or backward-looking."""
    new = []
    # micro-price: size-weighted mid (heavier side pulls price) minus wap
    micro = (df["bid_price"] * df["ask_size"] + df["ask_price"] * df["bid_size"]) / df["book_total"]
    df["micro_wap"] = micro - df["wap"]
    df["rel_spread"] = df["spread"] / df["wap"]
    df["near_ref"] = df["near_price"] - df["reference_price"]
    new += ["micro_wap", "rel_spread", "near_ref"]
    # log-return realized vol within day (backward-looking)
    logwap = np.log(df["wap"].clip(lower=1e-9))
    ret = logwap.groupby([df["stock_id"], df["date_id"]]).diff()
    df["wap_logret"] = ret
    df["wap_ret_rvol6"] = ret.groupby([df["stock_id"], df["date_id"]]).transform(
        lambda s: s.rolling(6, min_periods=1).std())
    new += ["wap_logret", "wap_ret_rvol6"]
    return new


def add_close_interactions(df):
    """Family 4: key features x auction_frac and auction_frac**2."""
    af = df["auction_frac"]
    af2 = af * af
    new = []
    for col in ["imb_flag_size", "book_imb", "imb_to_matched"]:
        df[f"{col}_xaf"] = df[col] * af
        df[f"{col}_xaf2"] = df[col] * af2
        new += [f"{col}_xaf", f"{col}_xaf2"]
    return new
