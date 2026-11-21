"""Trade-level meta-labeling on the PIT Nasdaq-100 panel (scripts/run_meta_label.py, scripts/run_meta_grid.py).

A trade is a maximal run of consecutive candidate weeks (tradable member, features complete,
SMA_fast > SMA_slow at the d_t close); it is held from the open of e_t of its first week to the
open after its last week. Label variants:
  win    y = 1{trade return > 0}
  win_w  same label, training rows weighted by |trade return|
  mag    y = 1{trade return > median trade return of the training set}
  cap26  y = 1{return over the first <= 26 weeks > 0}; the label matures after at most 26 weeks
A label is usable for training only once it has matured (trade exit, or 26 weeks for cap26).
"""

from __future__ import annotations

import io
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FEATURES
from .portfolio import capped_equal, weekly_ledger
from .universe import DATA_END, ZIP

ROOT = Path(__file__).resolve().parents[2]

RK = [f + "_rk" for f in FEATURES]
QF = ["q_" + f for f in FEATURES]
X_COLS = FEATURES + RK + QF
CAP_WEEKS = 26
VARIANTS = {
    "win": dict(ret="ret", time="exit", rule="pos", weight=False),
    "win_w": dict(ret="ret", time="exit", rule="pos", weight=True),
    "mag": dict(ret="ret", time="exit", rule="median", weight=False),
    "cap26": dict(ret="ret_cap", time="t_cap", rule="pos", weight=False),
}
SMA_CACHE = ROOT / "data" / "processed" / "nasdaq100_sma_grid.pkl"


def sma_windows(P: dict, wins) -> dict[int, np.ndarray]:
    """SMA of each stock's own closes at every decision date d_t; cached and extended on demand."""
    cache = pickle.loads(SMA_CACHE.read_bytes()) if SMA_CACHE.exists() else {}
    missing = [int(k) for k in wins if int(k) not in cache]
    if missing:
        cal, weeks, files = P["cal"], P["weeks"], P["files"]
        new = {k: np.full((len(weeks), len(files)), np.nan) for k in missing}
        with zipfile.ZipFile(ZIP) as z:
            names = {n.split("/")[-1]: n for n in z.namelist()}
            for j, f in enumerate(files):
                raw = pd.read_csv(io.TextIOWrapper(z.open(names[f]), encoding="utf-8-sig"), usecols=["Date", "Open", "Close"])
                raw.columns = ["date", "open", "close"]
                raw["date"] = pd.to_datetime(raw["date"])
                raw = raw[(raw.date <= DATA_END) & (raw.open > 0) & (raw.close > 0)].drop_duplicates("date").sort_values("date")
                c = raw.set_index("date")["close"]
                for k in missing:
                    new[k][:, j] = c.rolling(k, min_periods=k).mean().reindex(cal).to_numpy()[weeks.i_d.to_numpy()]
        cache.update(new)
        tmp = SMA_CACHE.with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(cache, protocol=5))
        tmp.replace(SMA_CACHE)
    return {int(k): cache[int(k)] for k in wins}


def build_trades(P: dict, sma_f: np.ndarray, sma_s: np.ndarray, qfeat: pd.DataFrame, train_start: int = 2000):
    weeks, F, Of = P["weeks"], P["feat"], P["Of"]
    nW, nJ = P["R"].shape
    complete = np.all([~np.isnan(F[c]) for c in FEATURES], axis=0) & ~np.isnan(sma_s) & ~np.isnan(sma_f)
    cand = P["tradable"] & complete & (sma_f > sma_s) & (weeks.year >= train_start).to_numpy()[:, None]
    G = Of[weeks.i_e_next.to_numpy()] / Of[weeks.i_e.to_numpy()]
    LG = np.vstack([np.zeros((1, nJ)), np.cumsum(np.log(np.where(cand, np.nan_to_num(G, nan=1.0), 1.0)), axis=0)])

    kk, jj = np.nonzero(cand)
    L = pd.DataFrame({"k": kk, "j": jj, **{f: F[f][kk, jj] for f in FEATURES}})
    rk = L.groupby("k")[FEATURES].rank(pct=True)

    pad = np.vstack([np.zeros((1, nJ), bool), cand, np.zeros((1, nJ), bool)]).astype(np.int8)
    dif = np.diff(pad, axis=0).T  # (assets x weeks+1)
    js, k0 = np.nonzero(dif == 1)
    je, k1 = np.nonzero(dif == -1)
    k1 = k1 - 1
    T = pd.DataFrame({"j": js, "k0": k0, "k1": k1})
    T["ret"] = np.exp(LG[k1 + 1, js] - LG[k0, js]) - 1
    kc = np.minimum(k1, k0 + CAP_WEEKS - 1)
    T["ret_cap"] = np.exp(LG[kc + 1, js] - LG[k0, js]) - 1
    T["weeks_held"] = k1 - k0 + 1
    T["entry"] = weeks.d.to_numpy()[k0]
    T["entry_year"] = weeks.year.to_numpy()[k0]
    T["exit"] = weeks.e_next.to_numpy()[k1]
    T["t_cap"] = weeks.e_next.to_numpy()[kc]
    T["completed"] = k1 < nW - 1
    T["cap_known"] = T.completed | (T.weeks_held >= CAP_WEEKS)
    pos = pd.Series(np.arange(len(L)), index=pd.MultiIndex.from_arrays([L.k, L.j]))
    at = pos.reindex(pd.MultiIndex.from_arrays([T.k0, T.j])).to_numpy()
    for f in FEATURES:
        T[f] = L[f].to_numpy()[at]
        T[f + "_rk"] = rk[f].to_numpy()[at]
        T["q_" + f] = qfeat[f].reindex(T.entry).to_numpy()
    T = T.dropna(subset=X_COLS).reset_index(drop=True)
    return T, cand


def _fit(spec, X, y, w, Xt):
    m = spec.build()
    if spec.family == "rf":
        m.set_params(n_jobs=1)
    kw = {}
    if w is not None:
        kw = {"logisticregression__sample_weight": w} if spec.family == "logit" else {"sample_weight": w}
    return m.fit(X, y, **kw).predict_proba(Xt)[:, 1]


@dataclass
class Pred:
    p: pd.Series  # P(y = 1) at entry
    y: pd.Series  # label under the threshold of the model that made the forecast
    base: pd.Series  # training (weighted) mean of y: the `pbar` threshold


def _one_fit(T, v, spec, tr, te, p, y, base):
    ret = T[v["ret"]].to_numpy()
    thr = float(np.median(ret[tr])) if v["rule"] == "median" else 0.0
    ytr = (ret[tr] > thr).astype(int)
    w = np.abs(ret[tr]) if v["weight"] else None
    if spec is not None:
        p[te] = _fit(spec, T.loc[tr, X_COLS].to_numpy(), ytr, w, T.loc[te, X_COLS].to_numpy())
    y[te] = (ret[te] > thr).astype(float)
    base[te] = np.average(ytr, weights=w)


def rolling_predict(T, variant, spec, fd, years) -> Pred:
    v = VARIANTS[variant]
    p, y, base = (pd.Series(np.nan, index=T.index) for _ in range(3))
    tl, ey = T[v["time"]], T.entry_year
    for yr in years:
        tr = (tl <= fd[yr]).to_numpy()
        te = ((ey == yr) | ((yr == years[0]) & (ey < yr) & ~(T.exit <= fd[yr]))).to_numpy()
        if te.any() and tr.sum() >= 30:
            _one_fit(T, v, spec, tr, te, p, y, base)
    return Pred(p, y, base)


def fixed_predict(T, variant, spec, fd, fit_year) -> Pred:
    v = VARIANTS[variant]
    p, y, base = (pd.Series(np.nan, index=T.index) for _ in range(3))
    tr = (T[v["time"]] <= fd[fit_year]).to_numpy()
    te = ~(T.exit <= fd[fit_year]).to_numpy()
    _one_fit(T, v, spec, tr, te, p, y, base)
    return Pred(p, y, base)


def label_known(T, variant) -> pd.Series:
    return T.cap_known if variant == "cap26" else T.completed


class Holdings:
    """Fast trade-set -> (weeks x assets) capped-equal weights."""

    def __init__(self, T: pd.DataFrame, shape):
        lens = T.weeks_held.to_numpy()
        self.t = np.repeat(np.arange(len(T)), lens)
        self.k = T.k0.to_numpy()[self.t] + (np.arange(lens.sum()) - np.repeat(np.cumsum(lens) - lens, lens))
        self.j = T.j.to_numpy()[self.t]
        self.shape = shape

    def weights(self, keep) -> np.ndarray:
        M = np.zeros(self.shape, bool)
        c = np.asarray(keep, bool)[self.t]
        M[self.k[c], self.j[c]] = True
        return capped_equal(M)


def ledger_stats(W, rows, weeks, Of, cost):
    wk = weeks.iloc[rows]
    led = weekly_ledger(W[rows], Of[wk.i_e_next.to_numpy()] / Of[wk.i_e.to_numpy()], cost)
    eq = led["eq_end"]
    years = (wk.e_next.iloc[-1] - wk.e.iloc[0]).days / 365.25
    r = np.diff(np.r_[1.0, eq]) / np.r_[1.0, eq[:-1]]
    return {"cagr_pct": (eq[-1] ** (1 / years) - 1) * 100,
            "sharpe_weekly": r.mean() / r.std(ddof=1) * np.sqrt(52) if r.std() > 0 else np.nan,
            "max_dd_pct": (eq / np.maximum.accumulate(np.maximum(eq, 1.0)) - 1).min() * 100,
            "exposure_pct": W[rows].sum(1).mean() * 100}
