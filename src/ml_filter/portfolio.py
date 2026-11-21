"""Weekly-rebalanced multi-asset ledger.

Targets are set at the open of e_k and held, drifting, until the open of e_{k+1}.
Cost: ``cost_bps`` per side on traded notional, sum |target - drifted weight|,
taken from equity at the rebalance open. Cash earns 0. The account starts in cash
and is liquidated at the final e_next open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import metrics

CAP = 0.10


def capped_equal(sel: np.ndarray) -> np.ndarray:
    """Boolean selection (weeks x assets) -> weights min(1/n, 10%), rest in cash."""
    n = sel.sum(axis=1, keepdims=True)
    w = np.where(sel, np.minimum(1.0 / np.maximum(n, 1), CAP), 0.0)
    return w


def weekly_ledger(W: np.ndarray, G: np.ndarray, cost_bps: float):
    """W, G: (weeks x assets) target weights and open-to-open gross returns."""
    c = cost_bps / 1e4
    G = np.where(W > 0, np.nan_to_num(G, nan=1.0), 1.0)
    gross = 1 + (W * (G - 1)).sum(axis=1)
    drift = W * G / gross[:, None]
    prev = np.vstack([np.zeros((1, W.shape[1])), drift[:-1]])
    turn = np.abs(W - prev).sum(axis=1)
    final_turn = drift[-1].sum()
    after_cost = np.cumprod(np.concatenate([[1.0], (1 - c * turn[1:]) * gross[:-1]])) * (1 - c * turn[0])
    eq_end = after_cost * gross
    eq_end[-1] *= 1 - c * final_turn
    return dict(after_cost=after_cost, eq_end=eq_end, turnover=turn, final_turn=final_turn)


def daily_curve(W, weeks, Of, Cf, led, cal) -> pd.DataFrame:
    i_e, i_n = weeks.i_e.to_numpy(), weeks.i_e_next.to_numpy()
    dates, eq, pos, trade = [], [], [], []
    for k in range(len(weeks)):
        held = np.flatnonzero(W[k])
        s = np.arange(i_e[k], i_n[k])
        mark = 1 - W[k].sum() + (W[k, held] * Cf[s][:, held] / Of[i_e[k], held]).sum(axis=1)
        dates.append(cal[s]); eq.append(led["after_cost"][k] * mark)
        pos.append(np.full(len(s), W[k].sum())); trade.append(np.r_[led["turnover"][k] > 0, np.zeros(len(s) - 1)])
    dates.append(cal[[i_n[-1]]]); eq.append([led["eq_end"][-1]]); pos.append([0.0]); trade.append([1])
    return pd.DataFrame({"date": np.concatenate(dates), "equity": np.concatenate(eq),
                         "position": np.concatenate(pos), "trade": np.concatenate(trade).astype(int)})


def portfolio_metrics(curve: pd.DataFrame, W: np.ndarray, led: dict, weeks: pd.DataFrame) -> dict:
    m = metrics(curve)
    pos = curve["position"].to_numpy()[:-1]
    held = int((pos > 0).sum())
    m["holding_cagr_pct"] = (curve["equity"].iloc[-1] ** (252 / held) - 1) * 100 if held else float("nan")
    m["holding_years"] = held / 252
    years = (weeks.e_next.iloc[-1] - weeks.e.iloc[0]).days / 365.25
    r = np.diff(np.r_[1.0, led["eq_end"]]) / np.r_[1.0, led["eq_end"][:-1]]
    m["sharpe_weekly"] = r.mean() / r.std(ddof=1) * np.sqrt(52)
    m["avg_names"] = float((W > 0).sum(axis=1).mean())
    m["annual_turnover"] = float((led["turnover"].sum() + led["final_turn"]) / years)
    del m["trades"], m["trades_per_year"]
    return m


def weekly_metrics(led: dict, weeks: pd.DataFrame) -> dict:
    eq = led["eq_end"]
    years = (weeks.e_next.iloc[-1] - weeks.e.iloc[0]).days / 365.25
    r = np.diff(np.r_[1.0, eq]) / np.r_[1.0, eq[:-1]]
    return {"cagr_pct": (eq[-1] ** (1 / years) - 1) * 100, "sharpe_weekly": r.mean() / r.std(ddof=1) * np.sqrt(52)}
