"""QQQ: trend rule + ML filter on market-state features, predicting return SIZE (exploratory grid).

Base rule: SMA_fast > SMA_slow, default 20/140 (weekly decision, next-open fill, 5 bps). The filter holds a candidate week only if
the model's forecast is high enough.

Grid (every combination is run):
  candidates  base  = weeks with SMA20 > SMA140
              union = weeks with SMA20 > SMA140 or SMA20 > SMA60
              all   = every week (ML decides alone)
  features    price7 (the original seven) | breadth | breadth+st | new (VIX + breadth) | new+cross | new+cross+st | all | all+late
              (src/ml_filter/xfeat.py; `st` = short-term breadth)
  target      forward open-to-open return over 1, 4 or 13 weeks
  model       ridge10, ridge100, lasso, gb_d1, gb_d2, rf, knn, svr (regression on the return);
              logit (sign of the return), logit_w (sign, rows weighted by |return|)
  threshold   q33 / q50: forecast >= the 33rd / 50th percentile of the in-sample training forecasts;
              pos: forecast > 0 (or P >= 0.5)
Annual expanding refit from 2000; a label is used only once it is complete before the year's first decision.

DEV = 2005-2018 is for choosing; HOLDOUT = 2019-2025 is reported for every configuration but must not drive choices.
The script picks one configuration per candidate set mechanically from DEV only and marks it.
Usage: .venv/bin/python scripts/run_qqq_state.py [--sym QQQ] [--pair 20_140] [--fastpair 20_60]
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ml_filter.backtest import equity_curve, metrics  # noqa: E402
from ml_filter.data import load_daily, weekly_table  # noqa: E402
from ml_filter.meta import sma_windows  # noqa: E402
from ml_filter.universe import build_panel  # noqa: E402
from ml_filter.xfeat import FEATURE_SETS as FS  # noqa: E402
from ml_filter.xfeat import market_features  # noqa: E402
from run_experiments import md_table  # noqa: E402

warnings.filterwarnings("ignore")
COST = 5
YEARS = list(range(2005, 2026))
PERIODS = {"dev": (2005, 2018), "holdout": (2019, 2025), "full": (2005, 2025)}
FEATS = {
    "price7": FS["price7"],
    "new": FS["vix"] + FS["breadth"],
    "new+cross": FS["vix"] + FS["breadth"] + FS["cross"],
    "breadth": FS["breadth"],
    "breadth+st": FS["breadth"] + FS["breadth_st"],
    "new+cross+st": FS["vix"] + FS["breadth"] + FS["breadth_st"] + FS["cross"],
    "all": FS["price7"] + FS["vix"][:-1] + FS["breadth"] + FS["cross"] + FS["trend"],
    "all+late": FS["price7"] + FS["vix"][:-1] + FS["breadth"] + FS["cross"] + FS["trend"] + FS["late"],
}
HORIZONS = [1, 4, 13]
pre = lambda *steps: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), *steps)  # noqa: E731
MODELS = {
    "ridge10": lambda: pre(Ridge(alpha=10.0)),
    "ridge100": lambda: pre(Ridge(alpha=100.0)),
    "lasso": lambda: pre(Lasso(alpha=0.02)),
    "gb_d1": lambda: pre(GradientBoostingRegressor(n_estimators=50, max_depth=1, learning_rate=0.05, min_samples_leaf=20, random_state=42)),
    "gb_d2": lambda: pre(GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05, min_samples_leaf=20, random_state=42)),
    "rf": lambda: pre(RandomForestRegressor(n_estimators=300, max_depth=3, min_samples_leaf=20, max_features="sqrt", random_state=42, n_jobs=1)),
    "knn": lambda: pre(KNeighborsRegressor(n_neighbors=25)),
    "svr": lambda: pre(SVR(C=0.5, epsilon=0.5)),
    "logit": lambda: pre(LogisticRegression(C=0.1, max_iter=1000)),
    "logit_w": lambda: pre(LogisticRegression(C=0.1, max_iter=1000)),
}


def walk(X, cand, feats, h, mk, fd):
    """Returns forecast and the in-sample 33rd / 50th percentile thresholds per row."""
    pred, q33, q50 = (np.full(len(X), np.nan) for _ in range(3))
    y, tl, yr_ = X[f"R{h}"].to_numpy(), X[f"t{h}"], X["year"].to_numpy()
    A = X[feats].to_numpy()
    for yr in YEARS:
        tr = cand & (yr_ >= 2000) & np.isfinite(y) & (tl <= fd[yr]).to_numpy()
        te = cand & (yr_ == yr)
        if not te.any() or tr.sum() < 60:
            continue
        m = MODELS[mk]()
        if mk.startswith("logit"):
            kw = {"logisticregression__sample_weight": np.abs(y[tr])} if mk == "logit_w" else {}
            m.fit(A[tr], (y[tr] > 0).astype(int), **kw)
            f = lambda Z: m.predict_proba(Z)[:, 1] - 0.5  # noqa: E731
        else:
            m.fit(A[tr], y[tr] / y[tr].std())
            f = m.predict
        ins = f(A[tr])
        pred[te], q33[te], q50[te] = f(A[te]), np.quantile(ins, 1 / 3), np.quantile(ins, 0.5)
    return pred, q33, q50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sym", default="QQQ")
    ap.add_argument("--pair", default="20_140", help="base rule: SMA fast_slow")
    ap.add_argument("--fastpair", default="20_60", help="second, faster rule used by the `union` candidate set")
    a = ap.parse_args()
    sym = a.sym
    (f1, s1), (f2, s2) = (tuple(int(x) for x in v.split("_")) for v in (a.pair, a.fastpair))
    out = ROOT / "results" / "qqq_state" / f"{sym}_sma{f1}_{s1}"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    P = build_panel()
    X = market_features(P, sma_windows(P, [1, 50, 140, 200]))
    D = load_daily(sym)
    W = weekly_table(D)
    # market-state features are common; the traded asset, its trend state and its targets depend on --sym / --pair
    X = X.set_index("d").reindex(W.d).reset_index()
    c = D.set_index("date")["close"]
    sma = lambda k: c.rolling(k).mean()  # noqa: E731
    X["slow_up"] = (sma(f1) > sma(s1)).astype(float).where(sma(s1).notna()).reindex(W.d).to_numpy()
    X["fast_up"] = (sma(f2) > sma(s2)).astype(float).where(sma(s2).notna()).reindex(W.d).to_numpy()
    X["R1"], X["t1"] = W.R.to_numpy(), W.e_next.to_numpy()
    for h in (4, 13):
        X[f"R{h}"] = (1 + W.R).rolling(h).apply(np.prod, raw=True).shift(-(h - 1)).to_numpy() - 1
        X[f"t{h}"] = W.e_next.shift(-(h - 1)).to_numpy()
    X["year"] = W.year.to_numpy()
    fd = W.groupby("year")["d"].min().to_dict()
    slow, fast = (X.slow_up == 1).to_numpy(), (X.fast_up == 1).to_numpy()
    CANDS = {"base": slow, "union": slow | fast, "all": X.slow_up.notna().to_numpy()}

    def perf(w, y0, y1):
        ev = W[W.year.between(y0, y1) & (W.e_next <= pd.Timestamp(f"{y1}-12-31"))]
        m = metrics(equity_curve(D, ev, w[ev.index].astype(float), COST))
        return {"cagr": m["cagr_pct"], "hold_cagr": m["holding_cagr_pct"], "maxdd": m["max_dd_pct"], "sharpe": m["sharpe"],
                "exposure": m["exposure_pct"], "trades_py": m["trades_per_year"]}

    tasks = [(cn, fn, h, mk) for cn in CANDS for fn in FEATS for h in HORIZONS for mk in MODELS]
    res = Parallel(n_jobs=12)(delayed(walk)(X, CANDS[cn], FEATS[fn], h, mk, fd) for cn, fn, h, mk in tasks)
    print(f"[{time.time() - t0:5.1f}s] {len(tasks)} walk-forwards done")

    rows, keeps = [], {}
    for nm, w in {"buy_hold": np.ones(len(X), bool), "base": slow, "union": slow | fast}.items():
        for pn, (a, b) in PERIODS.items():
            rows.append({"cand": "-", "feats": "-", "h": 0, "model": nm, "thr": "-", "period": pn, **perf(w, a, b)})
    for (cn, fn, h, mk), (pred, q33, q50) in zip(tasks, res):
        for tn, thr in (("q33", q33), ("q50", q50), ("pos", np.zeros(len(X)))):
            keep = CANDS[cn] & (pred >= thr) if tn != "pos" else CANDS[cn] & (pred > 0)
            keeps[(cn, fn, h, mk, tn)] = keep
            for pn, (a, b) in PERIODS.items():
                rows.append({"cand": cn, "feats": fn, "h": h, "model": mk, "thr": tn, "period": pn, **perf(keep, a, b)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "grid.csv", index=False)
    wide = df.pivot_table(index=["cand", "feats", "h", "model", "thr"], columns="period", values=["cagr", "hold_cagr", "maxdd", "sharpe", "exposure"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    ref = wide[wide.cand == "-"].set_index("model")
    g = wide[wide.cand != "-"].copy()
    b = ref.loc["base"]
    for p in PERIODS:
        g[f"d_hold_{p}"] = g[f"hold_cagr_{p}"] - b[f"hold_cagr_{p}"]
        g[f"d_cagr_{p}"] = g[f"cagr_{p}"] - b[f"cagr_{p}"]
    g.to_csv(out / "grid_wide.csv", index=False)

    L = [f"# {sym}: SMA{f1}/{s1} + ML filter on market-state features (auto-generated by scripts/run_qqq_state.py)", "", __doc__.split("Usage")[0], ""]
    L += ["## Benchmarks", "", md_table(ref.reset_index()[["model"] + [f"{k}_{p}" for p in PERIODS for k in ("cagr", "hold_cagr", "maxdd", "sharpe", "exposure")]]), ""]
    ok = g[g.exposure_dev >= 35]  # a filter that keeps almost nothing makes holding CAGR meaningless
    L += [f"Configurations: {len(g)}; with DEV exposure >= 35%: {len(ok)}. Deltas are versus the base rule, in pp.", ""]

    def cell(by):
        t = ok.groupby(by).agg(n=("d_hold_dev", "size"), d_hold_dev=("d_hold_dev", "mean"), win_hold_dev=("d_hold_dev", lambda x: (x > 0).mean() * 100),
                               d_cagr_dev=("d_cagr_dev", "mean"), d_hold_holdout=("d_hold_holdout", "mean"),
                               win_hold_holdout=("d_hold_holdout", lambda x: (x > 0).mean() * 100), d_cagr_holdout=("d_cagr_holdout", "mean"),
                               exposure_dev=("exposure_dev", "mean")).reset_index()
        return md_table(t)

    for by in (["cand"], ["feats"], ["h"], ["model"], ["thr"], ["cand", "feats"], ["feats", "h"]):
        L += [f"## Average over configurations, by {' x '.join(by)}", "", cell(by), ""]
    L += [f"Correlation across configurations between DEV and HOLDOUT delta holding CAGR: {ok.d_hold_dev.corr(ok.d_hold_holdout):+.2f} "
          f"(rank: {ok.d_hold_dev.corr(ok.d_hold_holdout, method='spearman'):+.2f})", ""]

    # mechanical DEV-only selection: best DEV holding CAGR among configurations that also keep DEV CAGR within 1 pp of the base rule
    cols = ["cand", "feats", "h", "model", "thr", "cagr_dev", "hold_cagr_dev", "maxdd_dev", "sharpe_dev", "exposure_dev",
            "cagr_holdout", "hold_cagr_holdout", "maxdd_holdout", "sharpe_holdout", "exposure_holdout", "d_hold_holdout", "d_cagr_holdout"]
    sel = []
    for cn in CANDS:
        s = ok[(ok.cand == cn) & (ok.d_cagr_dev >= -1.0)].sort_values("hold_cagr_dev", ascending=False)
        if len(s):
            sel.append(s.iloc[0])
    sel = pd.DataFrame(sel)
    L += ["## Selected from DEV only (best DEV holding CAGR with DEV CAGR within 1 pp of the base rule), one per candidate set", "", md_table(sel[cols]), ""]
    L += ["## Top 15 by DEV holding CAGR (exposure >= 35%)", "", md_table(ok.sort_values("hold_cagr_dev", ascending=False).head(15)[cols]), ""]

    # placebo for the selected rules: circular shifts of the keep series inside the candidate weeks (keeps rate and persistence)
    rng = np.random.default_rng(628)
    prow = []
    ref = pd.DataFrame([{"cand": "base", "feats": "new+cross", "h": 1, "model": "ridge10", "thr": "pos"}])  # fixed reference, not chosen from results
    for _, r in pd.concat([ref, sel[["cand", "feats", "h", "model", "thr"]]], ignore_index=True).drop_duplicates().iterrows():
        keep, cm = keeps[(r.cand, r.feats, r.h, r.model, r.thr)], CANDS[r.cand]
        idx = np.flatnonzero(cm & (X.year >= 2005).to_numpy())
        for pn, (a, b_) in PERIODS.items():
            real = perf(keep, a, b_)
            sim = []
            for _ in range(300):
                k2 = np.zeros(len(X), bool)
                k2[idx] = np.roll(keep[idx], rng.integers(26, len(idx) - 26))
                sim.append(perf(k2, a, b_))
            sim = pd.DataFrame(sim)
            prow.append({"cand": r.cand, "feats": r.feats, "h": r.h, "model": r.model, "thr": r.thr, "period": pn,
                         "hold_cagr": real["hold_cagr"], "placebo_median_hold": sim.hold_cagr.median(), "pctile_hold": (sim.hold_cagr < real["hold_cagr"]).mean() * 100,
                         "cagr": real["cagr"], "placebo_median_cagr": sim.cagr.median(), "pctile_cagr": (sim.cagr < real["cagr"]).mean() * 100,
                         "pctile_sharpe": (sim.sharpe < real["sharpe"]).mean() * 100})
    pl = pd.DataFrame(prow)
    pl.to_csv(out / "selected_placebo.csv", index=False)
    L += ["## Placebo for the reference rule (base | new+cross | 1w | ridge10 | pos) and the selected rules (300 circular shifts of the keep series within candidate weeks)", "", md_table(pl), ""]
    (out / "summary.md").write_text("\n".join(L))
    pd.DataFrame({"d": X.d, **{"|".join(map(str, k)): v for k, v in keeps.items() if any((sel[["cand", "feats", "h", "model", "thr"]].values == list(k)).all(1))}}).to_csv(out / "selected_keeps.csv", index=False)
    print(f"[{time.time() - t0:5.1f}s] written -> {out}")


if __name__ == "__main__":
    main()
