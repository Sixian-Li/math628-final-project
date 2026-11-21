"""Reverse-time test of the market-state filter: fit ONCE on late years, apply to earlier years that the fit never saw.

The walk-forward in scripts/run_qqq_state.py can only score 2005 onward (it needs earlier years to train on), so 2001-2004 was never
evaluated. Here the direction is flipped: one model fitted on the late period, frozen, applied to the early period.
  split A   train 2009-2025 -> test 2001-2008 (also reported for 2001-2004 and 2005-2008)
  split B   train 2008-2025 -> test 2001-2007
Everything else is the reference design: base rule SMA_fast > SMA_slow on the traded asset, 1-week target, hold a base week only if
the forecast passes the threshold, 5 bps. Features are market-level (VIX, Nasdaq-100 breadth, cross-asset) for every traded asset.
Reference configuration: new+cross | ridge10 | forecast > 0. Family: {ridge10, ridge100, lasso} x five new feature sets x {q33, q50, pos}.
Caveat: the feature groups were picked from a pre-screen on data up to 2018, so 2001-2008 is clean for the coefficients, not for the
choice of feature groups.
Usage: .venv/bin/python scripts/run_qqq_reverse.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import run_qqq_state as QS  # noqa: E402
from ml_filter.backtest import equity_curve, metrics  # noqa: E402
from ml_filter.data import load_daily, weekly_table  # noqa: E402
from ml_filter.meta import sma_windows  # noqa: E402
from ml_filter.universe import build_panel  # noqa: E402
from ml_filter.xfeat import market_features  # noqa: E402
from run_experiments import md_table  # noqa: E402

COST = 5
OUT = ROOT / "results" / "qqq_state" / "reverse"
SPLITS = {"A": ((2009, 2025), [(2001, 2008), (2001, 2004), (2005, 2008)]), "B": ((2008, 2025), [(2001, 2007)])}
CASES = [("QQQ", 20, 140), ("QQQ", 20, 60), ("QQQ", 40, 120), ("QQQ", 50, 200), ("QQQ", 10, 100), ("QQQ", 1, 200),
         ("SPY", 20, 140), ("SPY", 50, 200), ("XLK", 20, 140), ("IWM", 20, 140)]
LINEAR = ["ridge10", "ridge100", "lasso"]
NEWF = ["breadth", "breadth+st", "new", "new+cross", "new+cross+st"]
REF = ("new+cross", "ridge10", "pos")
N_PLACEBO = 300


def build(Xm, sym, f1, s1):
    D = load_daily(sym)
    W = weekly_table(D)
    X = Xm.set_index("d").reindex(W.d).reset_index()
    c = D.set_index("date")["close"]
    X["slow_up"] = (c.rolling(f1).mean() > c.rolling(s1).mean()).astype(float).where(c.rolling(s1).mean().notna()).reindex(W.d).to_numpy()
    X["R1"], X["year"] = W.R.to_numpy(), W.year.to_numpy()
    return D, W, X


def fit_apply(X, cand, feats, mk, tr_y, te_y):
    y, yr, A = X.R1.to_numpy(), X.year.to_numpy(), X[feats].to_numpy()
    tr = cand & (yr >= tr_y[0]) & (yr <= tr_y[1]) & np.isfinite(y)
    te = cand & (yr >= te_y[0]) & (yr <= te_y[1])
    m = QS.MODELS[mk]()
    m.fit(A[tr], y[tr] / y[tr].std())
    ins, pred = m.predict(A[tr]), np.full(len(X), np.nan)
    pred[te] = m.predict(A[te])
    return pred, {"q33": np.quantile(ins, 1 / 3), "q50": np.quantile(ins, 0.5), "pos": 0.0}, int(tr.sum())


def perf(D, W, w, y0, y1):
    ev = W[W.year.between(y0, y1) & (W.e_next <= pd.Timestamp(f"{y1}-12-31"))]
    m = metrics(equity_curve(D, ev, w[ev.index].astype(float), COST))
    return {"cagr": m["cagr_pct"], "hold_cagr": m["holding_cagr_pct"], "exposure": m["exposure_pct"], "sharpe": m["sharpe"], "maxdd": m["max_dd_pct"]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P = build_panel()
    Xm = market_features(P, sma_windows(P, [1, 50, 140, 200]))
    rng = np.random.default_rng(628)
    rules, fams, diag = [], [], []
    for sym, f1, s1 in CASES:
        D, W, X = build(Xm, sym, f1, s1)
        cand = (X.slow_up == 1).to_numpy()
        y, yr = X.R1.to_numpy(), X.year.to_numpy()
        case = f"{sym}_sma{f1}_{s1}"
        for sp, (tr_y, tests) in SPLITS.items():
            full = tests[0]
            preds = {(fn, mk): fit_apply(X, cand, QS.FEATS[fn], mk, tr_y, full) for fn in NEWF for mk in LINEAR}
            for te_y in tests:
                tag = {"case": case, "split": sp, "train": f"{tr_y[0]}-{tr_y[1]}", "test": f"{te_y[0]}-{te_y[1]}"}
                b = perf(D, W, cand, *te_y)
                rules.append({**tag, "rule": "buy and hold", **perf(D, W, np.ones(len(X), bool), *te_y)})
                rules.append({**tag, "rule": "base rule", **b})
                g = []
                for (fn, mk), (pred, thr, ntr) in preds.items():
                    for tn, t in thr.items():
                        keep = cand & ((pred > 0) if tn == "pos" else (pred >= t))
                        p = perf(D, W, keep, *te_y)
                        g.append({"feats": fn, "model": mk, "thr": tn, **p})
                        if (fn, mk, tn) == REF:
                            rules.append({**tag, "rule": "reference filter", **p})
                            m = cand & (yr >= te_y[0]) & (yr <= te_y[1]) & np.isfinite(y)
                            k, d = y[m & keep] * 1e4, y[m & ~keep] * 1e4
                            se = np.sqrt(k.var(ddof=1) / len(k) + d.var(ddof=1) / len(d)) if len(k) > 1 and len(d) > 1 else np.nan
                            r = np.corrcoef(pred[m], y[m])[0, 1]
                            idx = np.flatnonzero(cand & (yr >= te_y[0]) & (yr <= te_y[1]))
                            sim = []
                            for _ in range(N_PLACEBO):
                                k2 = np.zeros(len(X), bool)
                                k2[idx] = np.roll(keep[idx], rng.integers(13, len(idx) - 13))
                                sim.append(perf(D, W, k2, *te_y))
                            sim = pd.DataFrame(sim)
                            diag.append({**tag, "train_weeks": ntr, "kept_weeks": len(k), "kept_mean_bps": k.mean(), "dropped_weeks": len(d),
                                         "dropped_mean_bps": d.mean() if len(d) else np.nan, "diff_t": (k.mean() - d.mean()) / se if len(d) > 1 else np.nan,
                                         "corr": r, "corr_t": r * np.sqrt((m.sum() - 2) / (1 - r * r)),
                                         "pctile_hold": (sim.hold_cagr < p["hold_cagr"]).mean() * 100, "pctile_cagr": (sim.cagr < p["cagr"]).mean() * 100,
                                         "pctile_sharpe": (sim.sharpe < p["sharpe"]).mean() * 100})
                g = pd.DataFrame(g)
                fams.append({**tag, "n": len(g), "hold>base %": (g.hold_cagr > b["hold_cagr"]).mean() * 100, "cagr>base %": (g.cagr > b["cagr"]).mean() * 100,
                             "sharpe>base %": (g.sharpe > b["sharpe"]).mean() * 100, "d_hold": (g.hold_cagr - b["hold_cagr"]).mean(),
                             "d_cagr": (g.cagr - b["cagr"]).mean(), "d_maxdd": (g.maxdd - b["maxdd"]).mean(), "exposure": g.exposure.mean()})
    R, F, G = pd.DataFrame(rules), pd.DataFrame(fams), pd.DataFrame(diag)
    R.to_csv(OUT / "rules.csv", index=False)
    F.to_csv(OUT / "families.csv", index=False)
    G.to_csv(OUT / "reference_diagnostics.csv", index=False)
    na = Xm[(Xm.d >= "2001-01-01") & (Xm.d <= "2008-12-31")][QS.FEATS["new+cross"]].isna().mean() * 100
    txt = ["# Reverse-time test of the market-state filter (auto-generated by scripts/run_qqq_reverse.py)", "", __doc__.split("Usage")[0],
           "## Rules (percent; 5 bps)", "", md_table(R), "",
           "## Family: linear model x new feature sets x threshold, 1-week target (45 configurations; deltas vs the base rule in pp)", "", md_table(F, "{:.1f}"), "",
           "## Reference filter: kept vs dropped base weeks, forecast/return correlation, placebo percentiles (300 circular shifts)", "", md_table(G), "",
           "## Missing share of the reference features in 2001-2008 decision weeks (%; filled with the training median)", "",
           md_table(na[na > 0].rename("missing_pct").reset_index().rename(columns={"index": "feature"})) if (na > 0).any() else "none", ""]
    (OUT / "summary.md").write_text("\n".join(txt))
    print("\n".join(txt[4:]))


if __name__ == "__main__":
    main()
