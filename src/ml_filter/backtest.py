"""Vectorized all-in / all-out daily ledger driven by weekly target positions.

Execution (Section 4 of the report): the weekly target w_t is decided at the close of
d_t and filled at the open of e_t. Costs are an adverse fill of ``cost_bps`` per
side, charged only when the position changes. The account starts in cash at the
open of the first evaluation week and is liquidated at the open of the last
label end, so every strategy shares the same start and end fills.
Cash earns zero. Equity is marked at each close (the last row is the
liquidation open).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def daily_positions(weeks: pd.DataFrame, w: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Expand weekly targets to sessions [i0, iN); returns (h, i0, iN)."""
    i_e = weeks["i_e"].to_numpy()
    i_next = weeks["i_e_next"].to_numpy()
    if not np.array_equal(i_e[1:], i_next[:-1]):
        raise ValueError("evaluation weeks must be contiguous")
    lengths = i_next - i_e
    h = np.repeat(np.asarray(w, dtype=float), lengths)
    return h, int(i_e[0]), int(i_next[-1])


def equity_curve(daily: pd.DataFrame, weeks: pd.DataFrame, w: np.ndarray, cost_bps: float) -> pd.DataFrame:
    h, i0, iN = daily_positions(weeks, w)
    c = cost_bps / 10_000.0
    o = daily["open"].to_numpy()[i0 : iN + 1]
    cl = daily["close"].to_numpy()[i0 - 1 : iN + 1]  # cl[k] is the close before session i0+k
    prev_close, close = cl[:-1], cl[1:]
    h_full = np.append(h, 0.0)  # liquidate at the open of iN
    h_prev = np.concatenate([[0.0], h_full[:-1]])
    hold = (h_prev == 1) & (h_full == 1)
    buy = (h_prev == 0) & (h_full == 1)
    sell = (h_prev == 1) & (h_full == 0)
    f = np.ones_like(o)
    f[hold] = close[hold] / prev_close[hold]
    f[buy] = close[buy] / (o[buy] * (1 + c))
    f[sell] = o[sell] * (1 - c) / prev_close[sell]
    equity = np.cumprod(f)
    return pd.DataFrame(
        {
            "date": daily["date"].to_numpy()[i0 : iN + 1],
            "equity": equity,
            "position": h_full,
            "trade": (buy | sell).astype(int),
        }
    )


def exposure_matched(bh: pd.DataFrame, exposure: float) -> pd.DataFrame:
    """Ex-post diagnostic: constant-mix of buy-and-hold at a given average exposure, no cost."""
    r = bh["equity"].pct_change().fillna(bh["equity"].iloc[0] - 1.0)
    eq = np.cumprod(1 + exposure * r.to_numpy())
    return pd.DataFrame({"date": bh["date"], "equity": eq, "position": exposure, "trade": 0})


def metrics(curve: pd.DataFrame) -> dict[str, float]:
    eq = curve["equity"].to_numpy()
    dates = pd.to_datetime(curve["date"])
    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    r = np.diff(np.concatenate([[1.0], eq])) / np.concatenate([[1.0], eq[:-1]])
    sd = r.std(ddof=1)
    dd = eq / np.maximum.accumulate(np.maximum(eq, 1.0)) - 1
    trades = int(curve["trade"].sum())
    pos = curve["position"].to_numpy()[:-1]
    held = int(pos.sum()) if np.isin(pos, (0.0, 1.0)).all() else 0
    # Holding-period CAGR: annualize net account growth over sessions actually invested, 252 sessions per year.
    holding_cagr = (eq[-1] ** (252 / held) - 1) * 100 if held else float("nan")
    return {
        "cagr_pct": ((eq[-1]) ** (1 / years) - 1) * 100,
        "holding_cagr_pct": holding_cagr,
        "holding_years": held / 252,
        "total_return_pct": (eq[-1] - 1) * 100,
        "ann_vol_pct": sd * math.sqrt(252) * 100,
        "sharpe": r.mean() / sd * math.sqrt(252) if sd > 0 else float("nan"),
        "max_dd_pct": dd.min() * 100,
        "exposure_pct": float(np.mean(curve["position"].to_numpy()[:-1])) * 100,
        "trades": trades,
        "trades_per_year": trades / years,
    }


def yearly_returns(curve: pd.DataFrame) -> pd.Series:
    s = curve.set_index(pd.to_datetime(curve["date"]))["equity"]
    year_end = s.groupby(s.index.year).last()
    prev = year_end.shift(1).fillna(1.0)
    return (year_end / prev - 1) * 100
