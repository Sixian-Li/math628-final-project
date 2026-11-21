"""Load the frozen daily snapshots and build the weekly decision table.

Timing convention (Section 4 of the report):
  d_t     last session of week t; features and the base signal use its close
  e_t     first session of the next week; the target position is filled at its open
  e_{t+1} first session of the week after; the label return ends at its open
  R_t = O[e_{t+1}] / O[e_t] - 1,  y_t = 1{R_t > 0}
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

FEATURES = ["ret5", "ret20", "sma_spread", "price_dev", "sma60_slope", "vol20", "dd60"]
TREND_FEATURES = FEATURES[:5]
RISK_FEATURES = FEATURES[5:]


def load_daily(symbol: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"{symbol}.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    if df["date"].duplicated().any():
        raise ValueError(f"{symbol}: duplicate dates")
    if not np.isfinite(df[["open", "close"]].to_numpy()).all() or (df[["open", "close"]] <= 0).any().any():
        raise ValueError(f"{symbol}: non-positive or missing open/close")
    return df[["date", "open", "high", "low", "close"]]


def add_indicators(daily: pd.DataFrame) -> pd.DataFrame:
    """All rolling windows end at the current close (no future data)."""
    d = daily.copy()
    p = d["close"]
    for k in (10, 20, 30, 40, 60, 120, 140):
        d[f"sma{k}"] = p.rolling(k, min_periods=k).mean()
    d["ret5"] = p / p.shift(5) - 1
    d["ret20"] = p / p.shift(20) - 1
    d["ret60"] = p / p.shift(60) - 1
    d["sma_spread"] = d["sma20"] / d["sma60"] - 1
    d["price_dev"] = p / d["sma20"] - 1
    d["sma60_slope"] = d["sma60"] / d["sma60"].shift(5) - 1
    d["vol20"] = p.pct_change().rolling(20, min_periods=20).std(ddof=1)
    d["dd60"] = p / p.rolling(60, min_periods=60).max() - 1
    return d


def weekly_table(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per decision week with features, execution dates, and label."""
    d = add_indicators(daily).reset_index(drop=True)
    dates = d["date"]
    period = dates.dt.to_period("W-FRI")
    decision_idx = dates.groupby(period).idxmax().to_numpy()  # last session of each week
    # e_t is the session right after d_t, i.e. the first session of the next week.
    rows = []
    for k in range(len(decision_idx) - 2):
        i_d = int(decision_idx[k])
        i_e = i_d + 1
        i_e_next = int(decision_idx[k + 1]) + 1
        rows.append((i_d, i_e, i_e_next))
    idx = pd.DataFrame(rows, columns=["i_d", "i_e", "i_e_next"])
    w = d.loc[idx["i_d"]].reset_index(drop=True)
    w = w.rename(columns={"date": "d"})
    w["i_d"] = idx["i_d"].to_numpy()
    w["i_e"] = idx["i_e"].to_numpy()
    w["i_e_next"] = idx["i_e_next"].to_numpy()
    w["e"] = dates.iloc[w["i_e"]].to_numpy()
    w["e_next"] = dates.iloc[w["i_e_next"]].to_numpy()
    o = d["open"].to_numpy()
    w["R"] = o[w["i_e_next"]] / o[w["i_e"]] - 1
    w["y"] = (w["R"] > 0).astype(int)
    w["year"] = w["e"].dt.year  # a week belongs to the year in which its holding interval starts
    return w


def base_signal(weeks: pd.DataFrame, rule: str) -> pd.Series:
    """Base trend rule evaluated at d_t close: 'sma20_60', 'sma10_30', 'sma40_120', 'mom60'."""
    if rule == "mom60":
        s = weeks["ret60"] > 0
        valid = weeks["ret60"].notna()
    else:
        fast, slow = (int(x) for x in rule.removeprefix("sma").split("_"))
        s = weeks[f"sma{fast}"] > weeks[f"sma{slow}"]
        valid = weeks[f"sma{slow}"].notna()
    return (s & valid).astype(int)
