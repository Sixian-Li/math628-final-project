"""Stocks: dual-SMA trend rule + ML ranking on long-horizon characteristics, 4-week target (exploratory grid).

Candidates: PIT Nasdaq-100 members with SMA_fast > SMA_slow (default 20/140). S0 holds all of them.
Each ML score predicts the next 4-week open-to-open return RELATIVE to the other candidates of the same week:
  target  rel  = return minus the week's candidate mean, clipped at the week's 2nd / 98th percentile
          rank = percentile rank of the return within the week
          top  = 1 if the return is in the week's top third (classification)
  features (all turned into within-week percentile ranks; src/ml_filter/xfeat.py)
          old7      the seven short-horizon features of the earlier studies
          mom       12-1 and 6-1 month momentum, 12- and 6-month return, nearness to the 52-week high
          mom+risk  mom + volatility (20/60/252 d), max daily return, beta, idiosyncratic vol, skewness
          all       every stock feature (adds reversal, liquidity, volume trend, path smoothness, trend strength / age)
          all+mkt   all + raw market state (VIX, breadth, QQQ drawdown and 60-day return); tree models only
  model   ridge, lasso, gb (histogram gradient boosting), rf for rel / rank; logit for top
Annual expanding refit from 2000 (first forecasts 2003); a label is used once complete before the year's first decision.

Portfolio rules (weights min(1/n, 10%), 5 bps):
  top20_4w        re-rank every 4 weeks, hold the top 20; a name is dropped earlier only if it stops being a candidate
  top20_4w_buf40  same, but a held name stays while it ranks in the top 40
  top10_4w, top30_4w
  top20_1w_buf40  weekly re-rank with the top-40 buffer
  keep50_4w       hold every candidate in the top half by score (a filter in the literal sense)
Benchmarks: S0, EW members, QQQ, single-characteristic rankings, random scores (100 seeds per rule).

DEV = 2005-2018 is for choosing; HOLDOUT = 2019-2025 is reported for everything but must not drive choices.
Usage: .venv/bin/python scripts/run_stock_rank.py [--sma 20_140]
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
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LogisticRegression, Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ml_filter.data import FEATURES  # noqa: E402
from ml_filter.meta import sma_windows  # noqa: E402
from ml_filter.portfolio import capped_equal, weekly_ledger  # noqa: E402
from ml_filter.universe import build_panel, load_qqq  # noqa: E402
from ml_filter.xfeat import STOCK_FEATURES, market_features, stock_features  # noqa: E402
from run_experiments import md_table  # noqa: E402

warnings.filterwarnings("ignore")
COST = 5
PRED_YEARS = list(range(2003, 2026))
PERIODS = {"dev": (2005, 2018), "holdout": (2019, 2025), "full": (2005, 2025)}
OLD7 = [f + "_old" for f in FEATURES]
MOM = ["mom12_1", "mom6_1", "ret252", "ret126", "hi52"]
RISK = ["vol20", "vol60", "vol252", "max21", "beta252", "ivol60", "skew60"]
ALL = STOCK_FEATURES + ["sma_spread_old", "sma60_slope_old", "trend_age"]
MKT = ["m_vix", "m_breadth140", "m_dd250", "m_ret60"]
FEATS = {"old7": OLD7, "mom": MOM, "mom+risk": MOM + RISK, "all": ALL, "all+mkt": ALL + MKT}
MODELS = {
    "ridge": lambda: Ridge(alpha=100.0),
    "lasso": lambda: Lasso(alpha=0.005),
    "gb": lambda: HistGradientBoostingRegressor(max_iter=100, max_depth=3, learning_rate=0.05, min_samples_leaf=200, random_state=42),
    "rf": lambda: RandomForestRegressor(n_estimators=200, max_depth=4, min_samples_leaf=200, max_features=0.33, max_samples=0.3, random_state=42, n_jobs=1),
    "logit": lambda: LogisticRegression(C=0.1, max_iter=500),
}
RULES = {
    "top20_4w": dict(k=20, every=4), "top20_4w_buf40": dict(k=20, every=4, buffer=40), "top10_4w": dict(k=10, every=4),
    "top30_4w": dict(k=30, every=4), "top20_1w_buf40": dict(k=20, every=1, buffer=40), "keep50_4w": dict(frac=0.5, every=4),
}
N_RANDOM = 100


def select(score, cand, k=20, every=4, buffer=None, frac=None, phase=0):
    nW, nJ = score.shape
    sel, cur = np.zeros((nW, nJ), bool), np.zeros(nJ, bool)
    for t in range(nW):
        ok = cand[t] & np.isfinite(score[t])
        if (t - phase) % every == 0:
            if not ok.any():
                cur = np.zeros(nJ, bool)
            elif frac is not None:
                cur = ok & (score[t] >= np.quantile(score[t][ok], 1 - frac))
            else:
                order = np.argsort(-np.where(ok, score[t], -np.inf), kind="stable")
                rank = np.empty(nJ, int)
                rank[order] = np.arange(nJ)
                new = cur & ok & (rank < buffer) if buffer else np.zeros(nJ, bool)
                for j in order:
                    if new.sum() >= k or not ok[j]:
                        break
                    new[j] = True
                cur = new
        cur = cur & cand[t]
        sel[t] = cur
    return sel


def fit_task(df, feats, target, mk, fd):
    out = np.full(len(df), np.nan)
    A, y = df[feats].to_numpy(), df[target].to_numpy()
    yr_, tl = df.year.to_numpy(), df.t4
    for yr in PRED_YEARS:
        tr = (yr_ >= 2000) & np.isfinite(y) & (tl <= fd[yr]).to_numpy()
        te = yr_ == yr
        if not te.any():
            continue
        m = MODELS[mk]()
        if mk == "logit":
            out[te] = m.fit(A[tr], y[tr].astype(int)).predict_proba(A[te])[:, 1]
        else:
            out[te] = m.fit(A[tr], y[tr] / y[tr].std()).predict(A[te])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sma", default="20_140")
    ap.add_argument("--cands", default="trend", choices=["trend", "all"], help="all = every tradable member, no trend rule")
    a = ap.parse_args()
    FAST, SLOW = (int(x) for x in a.sma.split("_"))
    out = ROOT / "results" / "stock_rank" / (f"sma{FAST}_{SLOW}" if a.cands == "trend" else "no_trend_rule")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    P = build_panel()
    weeks, F, Of = P["weeks"], P["feat"], P["Of"]
    nW, nJ = P["R"].shape
    S = sma_windows(P, [1, FAST, 50, SLOW, 140, 200])
    XF = stock_features(P)
    cand = P["tradable"] & ~np.isnan(S[SLOW]) & (weeks.year >= 2000).to_numpy()[:, None]
    if a.cands == "trend":
        cand &= S[FAST] > S[SLOW]
    ie = weeks.i_e.to_numpy()
    G = Of[weeks.i_e_next.to_numpy()] / Of[ie]
    G4 = np.full((nW, nJ), np.nan)
    G4[:-4] = Of[ie[4:]] / Of[ie[:-4]] - 1
    run_id = np.cumsum(~cand, axis=0)  # trend age: weeks since the stock last was not a candidate
    age = np.zeros((nW, nJ))
    for t in range(1, nW):
        age[t] = np.where(cand[t], age[t - 1] + 1, 0)
    del run_id
    M = market_features(P, S).set_index("d").reindex(weeks.d)

    kk, jj = np.nonzero(cand)
    df = pd.DataFrame({"k": kk, "j": jj, "year": weeks.year.to_numpy()[kk], "fwd4": G4[kk, jj], "t4": weeks.e.shift(-4).to_numpy()[kk]})
    raw = {**{f: XF[f][kk, jj] for f in STOCK_FEATURES}, **{f + "_old": F[f][kk, jj] for f in FEATURES}, "trend_age": np.minimum(age[kk, jj], 104)}
    g = df.groupby("k")
    for f, v in raw.items():
        df[f] = (pd.Series(v).groupby(df.k).rank(pct=True) - 0.5).fillna(0.0).to_numpy()
    for f, c in zip(MKT, ["vix", "breadth140", "dd250", "ret60"]):
        df[f] = M[c].to_numpy()[kk]
    df[MKT] = df[MKT].fillna(df[MKT].median())
    lo, hi = g.fwd4.transform(lambda x: x.quantile(0.02)), g.fwd4.transform(lambda x: x.quantile(0.98))
    df["rel"] = df.fwd4.clip(lo, hi) - g.fwd4.transform("mean")
    df["rank"] = g.fwd4.rank(pct=True) - 0.5
    df["top"] = (g.fwd4.rank(pct=True) > 2 / 3).astype(float).where(df.fwd4.notna())
    fd = weeks.groupby("year")["d"].min().to_dict()
    print(f"[{time.time() - t0:5.1f}s] panel rows {len(df):,}; candidates/week {g.size().mean():.1f}")

    tasks = [(fn, tg, mk) for fn in FEATS for tg, mks in (("rel", ["ridge", "lasso", "gb", "rf"]), ("rank", ["ridge", "gb", "rf"]), ("top", ["logit"]))
             for mk in mks if not (fn == "all+mkt" and mk in ("ridge", "lasso", "logit"))]
    res = Parallel(n_jobs=12)(delayed(fit_task)(df, FEATS[fn], tg, mk, fd) for fn, tg, mk in tasks)
    print(f"[{time.time() - t0:5.1f}s] {len(tasks)} walk-forward models done")

    def to_mat(v):
        Mx = np.full((nW, nJ), np.nan)
        Mx[kk, jj] = v
        return Mx

    scores = {f"ML|{fn}|{tg}|{mk}": to_mat(p) for (fn, tg, mk), p in zip(tasks, res)}
    # ensembles: mean within-week percentile rank of the regression forecasts of one feature set (no configuration picking)
    pr = {t: pd.Series(p).groupby(df.k.to_numpy()).rank(pct=True).to_numpy() for t, p in zip(tasks, res)}
    for fn in FEATS:
        grp = [pr[t] for t in tasks if t[0] == fn and t[1] != "top"]
        tree = [pr[t] for t in tasks if t[0] == fn and t[2] in ("gb", "rf")]
        scores[f"ENS|{fn}|all models"] = to_mat(np.mean(grp, axis=0))
        scores[f"ENS|{fn}|trees"] = to_mat(np.mean(tree, axis=0))
    for nm, f, sgn in [("mom12_1", "mom12_1", 1), ("ret252", "ret252", 1), ("mom6_1", "mom6_1", 1), ("hi52", "hi52", 1), ("lowvol", "vol60", -1),
                       ("highvol", "vol60", 1), ("ret20_old", "ret20_old", 1), ("reversal21", "ret21", -1)]:
        scores[f"single|{nm}"] = to_mat(sgn * df[f].to_numpy())

    def stats(sel, y0, y1, o=None, w=None):
        rows = np.flatnonzero(((weeks.year >= y0) & (weeks.year <= y1) & (weeks.e_next <= pd.Timestamp(f"{y1}-12-31"))).to_numpy())
        W = capped_equal(sel[rows]) if w is None else w[rows]
        led = weekly_ledger(W, (G if o is None else o)[rows], COST)
        eq, wk = led["eq_end"], weeks.iloc[rows]
        yrs = (wk.e_next.iloc[-1] - wk.e.iloc[0]).days / 365.25
        r = np.diff(np.r_[1.0, eq]) / np.r_[1.0, eq[:-1]]
        return {"cagr": (eq[-1] ** (1 / yrs) - 1) * 100, "sharpe": r.mean() / r.std(ddof=1) * np.sqrt(52),
                "maxdd": (eq / np.maximum.accumulate(np.maximum(eq, 1)) - 1).min() * 100,
                "turnover": (led["turnover"].sum() + led["final_turn"]) / yrs, "names": (W > 0).sum(1).mean(), "exposure": W.sum(1).mean() * 100}

    qo, _ = load_qqq(P["cal"])
    rows = []
    bench = {"S0_all_candidates": (cand, None, None), "EW_members": (P["tradable"] & (weeks.year >= 2000).to_numpy()[:, None], None, None),
             "QQQ_buy_hold": (None, qo[weeks.i_e_next.to_numpy()] / qo[ie], np.ones((nW, 1)))}
    for nm, (sel, o, w) in bench.items():
        for pn, (y0, y1) in PERIODS.items():
            rows.append({"score": nm, "rule": "-", "period": pn, **stats(sel, y0, y1, o, w)})
    rng = np.random.default_rng(628)
    plc = {}
    for rn, kw in RULES.items():
        sims = [select(np.where(cand, rng.random(cand.shape), np.nan), cand, **kw) for _ in range(N_RANDOM)]
        for pn, (y0, y1) in PERIODS.items():
            plc[(rn, pn)] = np.array([stats(s, y0, y1)["cagr"] for s in sims])
            rows.append({"score": "random_median", "rule": rn, "period": pn, "cagr": float(np.median(plc[(rn, pn)]))})
    print(f"[{time.time() - t0:5.1f}s] placebo done")
    for sn, Mx in scores.items():
        for rn, kw in RULES.items():
            sel = select(Mx, cand, **kw)
            for pn, (y0, y1) in PERIODS.items():
                st = stats(sel, y0, y1)
                rows.append({"score": sn, "rule": rn, "period": pn, **st, "placebo_pctile": float((plc[(rn, pn)] < st["cagr"]).mean() * 100)})
            if kw.get("every", 1) > 1:  # sensitivity to the rebalance week: average CAGR over the 4 possible phases
                alt = [select(Mx, cand, **kw, phase=ph) for ph in range(1, kw["every"])]
                for r_ in rows[-len(PERIODS):]:
                    a_, b_ = PERIODS[r_["period"]]
                    r_["cagr_phase_avg"] = float(np.mean([r_["cagr"]] + [stats(s_, a_, b_)["cagr"] for s_ in alt]))
    res_df = pd.DataFrame(rows)
    res_df.to_csv(out / "grid.csv", index=False)

    # rank IC of each score against the 4-week forward return (non-overlapping: every 4th week)
    ic_rows = []
    d4 = df[(df.k % 4 == 0) & df.fwd4.notna()]
    for sn, Mx in scores.items():
        s = pd.Series(Mx[d4.k, d4.j], index=d4.index)
        ic = pd.DataFrame({"k": d4.k, "s": s, "y": d4.fwd4, "year": d4.year}).dropna().groupby(["year", "k"]).apply(
            lambda x: x.s.corr(x.y, method="spearman"), include_groups=False).dropna()
        for pn, (y0, y1) in PERIODS.items():
            v = ic[(ic.index.get_level_values(0) >= y0) & (ic.index.get_level_values(0) <= y1)]
            ic_rows.append({"score": sn, "period": pn, "ic_mean": v.mean(), "ic_t": v.mean() / v.std(ddof=1) * np.sqrt(len(v)), "n": len(v)})
    ic = pd.DataFrame(ic_rows)
    ic.to_csv(out / "ic.csv", index=False)

    wide = res_df.pivot_table(index=["score", "rule"], columns="period", values=["cagr", "sharpe", "maxdd", "turnover", "placebo_pctile", "cagr_phase_avg"], sort=False)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    s0 = wide[wide.score == "S0_all_candidates"].iloc[0]
    for p in PERIODS:
        wide[f"d_cagr_{p}"] = wide[f"cagr_{p}"] - s0[f"cagr_{p}"]
    wide.to_csv(out / "grid_wide.csv", index=False)
    icw = ic.pivot(index="score", columns="period", values=["ic_mean", "ic_t"])
    icw.columns = [f"{a}_{b}" for a, b in icw.columns]

    cols = ["score", "rule", "cagr_dev", "d_cagr_dev", "sharpe_dev", "maxdd_dev", "turnover_dev", "placebo_pctile_dev",
            "cagr_holdout", "d_cagr_holdout", "sharpe_holdout", "maxdd_holdout", "placebo_pctile_holdout"]
    L = [f"# Stocks: SMA{FAST}/{SLOW} + ML ranking, 4-week target (auto-generated by scripts/run_stock_rank.py)", "", __doc__.split("Usage")[0], ""]
    L += ["## Benchmarks", "", md_table(wide[wide.rule == "-"][["score", "cagr_dev", "sharpe_dev", "maxdd_dev", "cagr_holdout", "sharpe_holdout", "maxdd_holdout", "cagr_full"]]), ""]
    L += ["Random-score median CAGR by rule:", "", md_table(wide[wide.score == "random_median"][["rule", "cagr_dev", "cagr_holdout", "cagr_full"]]), ""]
    ml = wide[wide.score.str.startswith("ML|")].copy()
    ml[["_", "feats", "target", "model"]] = ml.score.str.split("|", expand=True)
    for by in (["feats"], ["target"], ["model"], ["rule"], ["feats", "model"]):
        t = ml.groupby(by).agg(n=("d_cagr_dev", "size"), d_cagr_dev=("d_cagr_dev", "mean"), win_dev=("d_cagr_dev", lambda x: (x > 0).mean() * 100),
                               pctile_dev=("placebo_pctile_dev", "mean"), d_cagr_holdout=("d_cagr_holdout", "mean"),
                               win_holdout=("d_cagr_holdout", lambda x: (x > 0).mean() * 100), pctile_holdout=("placebo_pctile_holdout", "mean")).reset_index()
        L += [f"## ML configurations averaged by {' x '.join(by)} (delta CAGR vs S0, pp)", "", md_table(t), ""]
    L += [f"Correlation across ML configurations between DEV and HOLDOUT delta CAGR: {ml.d_cagr_dev.corr(ml.d_cagr_holdout):+.2f}", ""]
    pcols = cols + ["cagr_phase_avg_dev", "cagr_phase_avg_holdout"]
    L += ["## Ensembles (mean rank of all regression forecasts of one feature set; no configuration picking)", "",
          md_table(wide[wide.score.str.startswith("ENS|")][pcols]), ""]
    L += ["## Single-characteristic rankings", "", md_table(wide[wide.score.str.startswith("single|")][cols]), ""]
    best_ml = ml.sort_values("cagr_dev", ascending=False).iloc[0]
    sg = wide[wide.score.str.startswith("single|")].sort_values("cagr_dev", ascending=False).iloc[0]
    L += ["## Selected from DEV only (highest DEV CAGR): best ML configuration and best single characteristic", "",
          md_table(pd.DataFrame([best_ml, sg])[cols]), ""]
    L += ["## Top 15 ML configurations by DEV CAGR", "", md_table(ml.sort_values("cagr_dev", ascending=False).head(15)[cols]), ""]
    L += ["## Rank IC against the 4-week forward return (every 4th week)", "", md_table(icw.reset_index(), "{:.3f}"), ""]
    (out / "summary.md").write_text("\n".join(L))
    print(f"[{time.time() - t0:5.1f}s] written -> {out}")


if __name__ == "__main__":
    main()
