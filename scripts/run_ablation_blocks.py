"""Leave-one-period-out test of the three-component strategy (SMA rule x Ridge market-state filter x momentum ranking).

The sample is cut into four periods: 2001-2006, 2007-2012, 2013-2018 and 2019-2025. Each period is scored with a Ridge model
fitted on the other three (plus 2000) and then frozen, so every forecast comes from a model that never saw its own period.
Training weeks within 14 days of the test period are dropped. For 2019-2025 the model uses past data only (2000-2018); for the
three earlier periods it also uses later years, so those are cross-validation results and not a tradable history.
The row `2001-2025` chains the four sets of out-of-period forecasts into one account.

Components, books, ledger, costs and metrics are those of scripts/run_ablation.py (weekly open-to-open, 5 bps per side on every
weight change, Sharpe from weekly returns, holding CAGR over invested weeks only):
  SMA       SMA_fast > SMA_slow on QQQ decides which weeks may be invested; a stock book with SMA also restricts the
            candidates to members whose own SMA_fast > SMA_slow.
  Ridge     ridge regression on 16 market-state features forecasting next week's QQQ return; invested when the forecast is > 0.
            With SMA it is fitted on the QQQ uptrend weeks of that pair, without SMA on every week.
  Momentum  point-in-time Nasdaq-100 members ranked by 12-month return, top k at min(1/k, 10%), re-ranked every 4 weeks.

Part 1  reference pair 20/140: all eight combinations, top 10 / 20 / 30.
Part 2  222 SMA pairs (fast 1..51 step 5, slow 30..230 step 10): gain of each strategy over the plain SMA rule on the same
        pair, drawn as one heatmap panel per period, and how often the three-component book beats every other combination.
Usage: .venv/bin/python scripts/run_ablation_blocks.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import run_ablation as AB  # noqa: E402
import run_ablation_sma_grid as SG  # noqa: E402
import run_qqq_state as QS  # noqa: E402
from ml_filter.data import load_daily, weekly_table  # noqa: E402
from ml_filter.meta import sma_windows  # noqa: E402
from ml_filter.portfolio import capped_equal  # noqa: E402
from ml_filter.universe import build_panel  # noqa: E402
from ml_filter.xfeat import market_features, stock_features  # noqa: E402
from run_experiments import md_table  # noqa: E402
from run_stock_rank import select  # noqa: E402

OUT = ROOT / "results" / "ablation" / "blocks"
BLOCKS = {"2001-2006": (2001, 2006), "2007-2012": (2007, 2012), "2013-2018": (2013, 2018), "2019-2025": (2019, 2025)}
ALL = "2001-2025"
PERIODS = {**BLOCKS, ALL: (2001, 2025)}
EMBARGO = np.timedelta64(14, "D")
REF, FAST, SLOW = SG.REF, SG.FAST, SG.SLOW
BASE, RIDGE = "SMA (QQQ)", "SMA + Ridge (QQQ)"
METRICS = SG.METRICS


def three(k):
    return f"SMA + Ridge + momentum top {k}"


def fit_blocks(X, cand):
    """Out-of-period forecast for every candidate week of 2001-2025, and the number of training weeks per period."""
    y, yr, A, d = X.R1.to_numpy(), X.year.to_numpy(), X[AB.FEATS].to_numpy(), X.d.to_numpy().astype("datetime64[D]")
    pred, n = np.full(len(X), np.nan), {}
    for bn, (y0, y1) in BLOCKS.items():
        out = (d < np.datetime64(f"{y0}-01-01") - EMBARGO) | (d > np.datetime64(f"{y1}-12-31") + EMBARGO)
        tr = cand & (yr >= 2000) & (yr <= 2025) & np.isfinite(y) & out
        te = cand & (yr >= y0) & (yr <= y1)
        m = QS.MODELS[AB.MODEL]()
        m.fit(A[tr], y[tr] / y[tr].std())
        pred[te], n[bn] = m.predict(A[te]), int(tr.sum())
    return pred, n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    P = build_panel()
    weeks, Of = P["weeks"], P["Of"]
    S = sma_windows(P, sorted({1, 50, 140, 200, *REF, *FAST, *SLOW}))
    ret252 = stock_features(P)["ret252"]
    D = load_daily("QQQ")
    W = weekly_table(D)
    X = market_features(P, S).set_index("d").reindex(W.d).reset_index()
    X["R1"], X["year"] = W.R.to_numpy(), W.year.to_numpy()
    c = D.set_index("date")["close"]
    y2000 = (weeks.year >= 2000).to_numpy()[:, None]
    G = Of[weeks.i_e_next.to_numpy()] / Of[weeks.i_e.to_numpy()]
    Gq = (1 + W.R.to_numpy())[:, None]
    RQ = {pn: AB.rows_of(W, y0, y1) for pn, (y0, y1) in PERIODS.items()}
    RS = {pn: AB.rows_of(weeks, y0, y1) for pn, (y0, y1) in PERIODS.items()}

    def q_row(sig, pn):
        return AB.stats(sig[RQ[pn]][:, None].astype(float), Gq[RQ[pn]], W.iloc[RQ[pn]])

    def s_row(sel, sig, pn):
        on = pd.Series(sig, index=X.d).reindex(weeks.d).fillna(False).to_numpy(bool)
        return AB.stats(capped_equal(sel[RS[pn]] & on[RS[pn]][:, None]), G[RS[pn]], weeks.iloc[RS[pn]])

    # combinations without the SMA rule do not depend on the pair
    allw = X.slow_up.notna().to_numpy()
    always = np.ones(len(X), bool)
    pred_all, n_all = fit_blocks(X, allw)
    ridge_only = allw & (pred_all > 0)
    c_all = P["tradable"] & y2000
    mom_all = {k: select(np.where(c_all, ret252, np.nan), c_all, k=k, every=4) for k in (10, 20, 30)}
    fixed = []
    for pn in PERIODS:
        fixed.append({"period": pn, "strategy": "QQQ buy-and-hold", **q_row(always, pn)})
        fixed.append({"period": pn, "strategy": "Ridge (QQQ)", **q_row(ridge_only, pn)})
        for k, sel in mom_all.items():
            fixed.append({"period": pn, "strategy": f"Momentum top {k}", **s_row(sel, always, pn)})
            fixed.append({"period": pn, "strategy": f"Ridge + momentum top {k}", **s_row(sel, ridge_only, pn)})
    fixed = pd.DataFrame(fixed)
    print(f"[{time.time() - t0:5.1f}s] data and pair-independent rows ready; training weeks (all weeks): {n_all}")

    def run_pair(f, s, ks):
        ms = c.rolling(s).mean()
        up = (c.rolling(f).mean() > ms).astype(float).where(ms.notna()).reindex(W.d).to_numpy() == 1
        pred, n = fit_blocks(X, up)
        keep = up & (pred > 0)
        cand = P["tradable"] & ~np.isnan(S[s]) & (S[f] > S[s]) & y2000
        score = np.where(cand, ret252, np.nan)
        books = {k: select(score, cand, k=k, every=4) for k in ks}
        rows = []
        for pn in PERIODS:
            tag = {"fast": f, "slow": s, "period": pn, "train_weeks": n.get(pn, sum(n.values()) // len(n))}
            rows.append({**tag, "strategy": BASE, **q_row(up, pn)})
            rows.append({**tag, "strategy": RIDGE, **q_row(keep, pn)})
            for k, sel in books.items():
                rows.append({**tag, "strategy": f"SMA + momentum top {k}", **s_row(sel, up, pn)})
                rows.append({**tag, "strategy": three(k), **s_row(sel, keep, pn)})
        return rows

    # ---- Part 1: reference pair
    ref = pd.DataFrame(run_pair(*REF, (10, 20, 30)))
    ref_all = pd.concat([ref.drop(columns=["fast", "slow", "train_weeks"]), fixed], ignore_index=True)
    ref_all.to_csv(OUT / "reference_pair.csv", index=False)
    ix = ref_all.set_index(["strategy", "period"])

    # plumbing check: rows without a model must equal results/ablation/grid.csv for 2019-2025
    ab = pd.read_csv(ROOT / "results" / "ablation" / "grid.csv").set_index(["universe", "book", "timing", "period"])
    amap = {BASE: ("-", "QQQ ETF", "sma"), "QQQ buy-and-hold": ("-", "QQQ ETF", "always")}
    for k in (10, 20, 30):
        amap[f"SMA + momentum top {k}"] = ("trend", f"momentum top{k}", "sma")
        amap[f"Momentum top {k}"] = ("all", f"momentum top{k}", "always")
    gap = max(abs(ix.loc[(sn, "2019-2025"), m] - ab.loc[(*key, "holdout 2019-2025"), m]) for sn, key in amap.items() for m in ("cagr", "hold_cagr", "sharpe", "maxdd"))
    check = f"Check: the model-free rows of 2019-2025 at {REF[0]}/{REF[1]} reproduce results/ablation/grid.csv with a maximum absolute difference of {gap:.1e}."
    print(check)

    def table(k, cols=("cagr", "hold_cagr", "sharpe")):
        names = [three(k), f"SMA + momentum top {k}", f"Ridge + momentum top {k}", f"Momentum top {k}", RIDGE, BASE, "Ridge (QQQ)", "QQQ buy-and-hold"]
        t = []
        for sn in names:
            r = {"strategy": sn}
            for pn in PERIODS:
                v = ix.loc[(sn, pn)]
                r[pn] = " / ".join(f"{v[m]:.2f}" if m == "sharpe" else f"{v[m]:.1f}" for m in cols)
            t.append(r)
        return pd.DataFrame(t)

    heads = {k: table(k) for k in (10, 20, 30)}
    risk = table(10, ("invested", "maxdd"))
    heads[10].to_csv(OUT / "headline_top10.csv", index=False)
    print(md_table(heads[10]))
    print(f"[{time.time() - t0:5.1f}s] reference pair done")

    # ---- Part 2: SMA-pair surface
    pairs = [(f, s) for f in FAST for s in SLOW if f < s]
    rows = []
    for n, (f, s) in enumerate(pairs):
        rows += run_pair(f, s, (10, 20))
        if n % 20 == 0:
            print(f"[{time.time() - t0:5.1f}s] {n + 1}/{len(pairs)} pairs", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "grid.csv", index=False)
    fixed.to_csv(OUT / "pair_independent.csv", index=False)

    key = ["fast", "slow", "period"]
    base = df[df.strategy == BASE].set_index(key)[[*METRICS, "invested"]]
    d = df[df.strategy != BASE].join(base, on=key, rsuffix="_base")
    for m in METRICS:
        d["d_" + m] = d[m] - d[m + "_base"]
    d.to_csv(OUT / "gain_vs_sma.csv", index=False)

    SG.PERIODS = BLOCKS
    SG.PLABEL = {"2001-2006": "2001–2006 (model fitted on 2007–2025)", "2007-2012": "2007–2012 (model fitted on 2000–2006 and 2013–2025)",
                 "2013-2018": "2013–2018 (model fitted on 2000–2012 and 2019–2025)", "2019-2025": "2019–2025 (model fitted on 2000–2018)"}
    dh = d[d.period.isin(BLOCKS)]
    for strat, tag in ((three(10), "top10"), (three(20), "top20"), (RIDGE, "qqq")):
        for m in METRICS:
            SG.heat(dh, strat, m, OUT / f"heat_{m}_{tag}.png", "QQQ")

    summ = []
    for strat in (three(10), three(20), RIDGE):
        for pn in PERIODS:
            g = d[(d.strategy == strat) & (d.period == pn)]
            r = {"strategy": strat, "period": pn, "pairs": len(g), "invested % (median)": g.invested.median(), "base invested % (median)": g.invested_base.median()}
            for m, (label, _, _) in METRICS.items():
                v = g["d_" + m]
                r[f"{label}: better %"], r[f"{label}: median"], r[f"{label}: min"], r[f"{label}: max"] = (v > 0).mean() * 100, v.median(), v.min(), v.max()
            summ.append(r)
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "summary.csv", index=False)

    # the three-component book against every other combination, pair by pair
    vs = []
    fx = fixed.set_index(["strategy", "period"])
    for k in (10, 20):
        main_ = df[df.strategy == three(k)].set_index(key)
        alts = [(f"SMA + momentum top {k}", True), (f"Ridge + momentum top {k}", False), (f"Momentum top {k}", False), (RIDGE, True), (BASE, True),
                ("Ridge (QQQ)", False), ("QQQ buy-and-hold", False)]
        for an, by_pair in alts:
            for pn in PERIODS:
                a = main_[main_.index.get_level_values("period") == pn]
                r = {"strategy": three(k), "against": an, "period": pn, "pairs": len(a)}
                for m, (label, _, _) in METRICS.items():
                    other = df[df.strategy == an].set_index(key)[m].reindex(a.index) if by_pair else fx.loc[(an, pn), m]
                    v = a[m] - other
                    r[f"{label}: better %"], r[f"{label}: median gain"] = (v > 0).mean() * 100, v.median()
                vs.append(r)
    vs = pd.DataFrame(vs)
    vs.to_csv(OUT / "versus.csv", index=False)

    L = ["# Leave-one-period-out test: SMA x Ridge x momentum (auto-generated by scripts/run_ablation_blocks.py)", "", __doc__.split("Usage")[0], check, "",
         f"## Reference pair {REF[0]}/{REF[1]}, top 10: CAGR / holding CAGR / Sharpe (percent per year; 5 bps)", "", md_table(heads[10]), "",
         "Rows without SMA (`Ridge + momentum`, `Momentum`, `Ridge (QQQ)`, buy-and-hold) use every member and every week.", "",
         f"## Reference pair {REF[0]}/{REF[1]}, top 10: invested weeks % / max drawdown %", "", md_table(risk), "",
         f"## Reference pair {REF[0]}/{REF[1]}, top 20: CAGR / holding CAGR / Sharpe", "", md_table(heads[20]), "",
         f"## Reference pair {REF[0]}/{REF[1]}, top 30: CAGR / holding CAGR / Sharpe", "", md_table(heads[30]), "",
         f"## {len(pairs)} SMA pairs: gain over `{BASE}` on the same pair (share of pairs that improve; median / min / max; Sharpe difference, CAGR and holding CAGR in pp per year)", "",
         md_table(summ, "{:.2f}"), "",
         f"## {len(pairs)} SMA pairs: the three-component book against every other combination (share of pairs where it is better, and the median gain)", "",
         md_table(vs, "{:.2f}"), ""]
    (OUT / "summary.md").write_text("\n".join(L))
    print(md_table(summ, "{:.2f}"))
    print(f"[{time.time() - t0:5.1f}s] done")


if __name__ == "__main__":
    main()
