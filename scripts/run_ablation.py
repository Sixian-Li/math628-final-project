"""Component ablation: SMA trend rule x Ridge market-state filter x momentum ranking (exploratory).

Three components, each switched on or off:
  SMA       dual moving-average rule, SMA20 > SMA140. On QQQ it decides which weeks may be invested. In a stock book it also
            restricts the candidates to members whose own SMA20 > SMA140 (universe `trend`).
  Ridge     ridge regression on 16 market-state features (VIX, Nasdaq-100 breadth, cross-asset: `new+cross` in
            scripts/run_qqq_state.py) forecasting next week's QQQ return; invested only when the forecast is > 0.
            With SMA it is trained on QQQ uptrend weeks only (`sma+ridge`); without SMA on every week (`ridge_only`).
  Momentum  point-in-time Nasdaq-100 members ranked by 12-month return (`ret252`), top k held at min(1/k, 10%),
            re-ranked every 4 weeks. No model.

Books     QQQ ETF | every candidate | momentum top 10 / 20 / 30; stock books in universe `trend` or `all` (every member)
Timing    always | sma | sma+ridge | ridge_only   (off weeks are held in cash)
Periods   rev 2001-2008     one model fitted on 2009-2025 and frozen (the walk-forward cannot score years before 2005)
          dev 2005-2018, holdout 2019-2025, full 2005-2025     annual expanding walk-forward
          rev_b 2001-2007   sensitivity: the model fitted on 2008-2025

Every row uses the same weekly open-to-open ledger with 5 bps per side on every weight change, so moving the whole book in
and out is charged. Sharpe is from weekly returns, max drawdown is at weekly marks, and holding CAGR annualizes over invested
weeks only. QQQ rows therefore differ slightly from scripts/run_qqq_state.py, which marks the account daily.

Caveats: 12-month return and the top-10 size were picked after looking at 2005-2018 results; the feature groups of the Ridge
model were picked from a pre-screen on data up to 2018; all rows share one timing signal, so they are not independent evidence.
Usage: .venv/bin/python scripts/run_ablation.py
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

import run_qqq_reverse as RV  # noqa: E402
import run_qqq_state as QS  # noqa: E402
from ml_filter.data import load_daily, weekly_table  # noqa: E402
from ml_filter.meta import sma_windows  # noqa: E402
from ml_filter.portfolio import capped_equal, weekly_ledger  # noqa: E402
from ml_filter.universe import build_panel  # noqa: E402
from ml_filter.xfeat import market_features, stock_features  # noqa: E402
from run_experiments import md_table  # noqa: E402
from run_stock_rank import select  # noqa: E402

COST = 5
OUT = ROOT / "results" / "ablation"
FEATS, MODEL = QS.FEATS["new+cross"], "ridge10"
PERIODS = {"rev 2001-2008": ("rev", 2001, 2008), "dev 2005-2018": ("fwd", 2005, 2018), "holdout 2019-2025": ("fwd", 2019, 2025),
           "full 2005-2025": ("fwd", 2005, 2025), "rev_b 2001-2007": ("rev_b", 2001, 2007)}
MAIN = ["rev 2001-2008", "dev 2005-2018", "holdout 2019-2025"]
TIMINGS = ["always", "sma", "sma+ridge", "ridge_only"]
HEADLINE = [("SMA + Ridge + momentum top 10", ("trend", "momentum top10", "sma+ridge")),
            ("SMA + Ridge + momentum top 20", ("trend", "momentum top20", "sma+ridge")),
            ("SMA + momentum top 20", ("trend", "momentum top20", "sma")),
            ("Ridge + momentum top 20", ("all", "momentum top20", "ridge_only")),
            ("Momentum top 20", ("all", "momentum top20", "always")),
            ("SMA + Ridge (QQQ)", ("-", "QQQ ETF", "sma+ridge")),
            ("SMA (QQQ)", ("-", "QQQ ETF", "sma")),
            ("Ridge (QQQ)", ("-", "QQQ ETF", "ridge_only")),
            ("QQQ buy-and-hold", ("-", "QQQ ETF", "always"))]


def stats(Wt, G, wk):
    """Wt, G: (weeks x assets) weights and gross open-to-open returns of one period; wk: the matching weekly rows."""
    led = weekly_ledger(Wt, G, COST)
    eq = led["eq_end"]
    yrs = (wk.e_next.iloc[-1] - wk.e.iloc[0]).days / 365.25
    ret = np.diff(np.r_[1.0, eq]) / np.r_[1.0, eq[:-1]]
    inv = Wt.sum(1) > 0
    return {"cagr": (eq[-1] ** (1 / yrs) - 1) * 100, "hold_cagr": (eq[-1] ** (52 / max(inv.sum(), 1)) - 1) * 100,
            "invested": inv.mean() * 100, "sharpe": ret.mean() / ret.std(ddof=1) * np.sqrt(52),
            "maxdd": (eq / np.maximum.accumulate(np.maximum(eq, 1)) - 1).min() * 100,
            "names": float((Wt[inv] > 0).sum(1).mean()) if inv.any() else 0.0,
            "turnover": (led["turnover"].sum() + led["final_turn"]) / yrs}


def rows_of(wk, y0, y1):
    return np.flatnonzero((wk.year.between(y0, y1) & (wk.e_next <= pd.Timestamp(f"{y1}-12-31"))).to_numpy())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P = build_panel()
    weeks, Of = P["weeks"], P["Of"]
    S = sma_windows(P, [1, 20, 50, 140, 200])
    XF = stock_features(P)
    X = market_features(P, S)
    W = weekly_table(load_daily("QQQ"))
    X["year"] = W.year.to_numpy()
    fd = W.groupby("year")["d"].min().to_dict()
    up, allw = (X.slow_up == 1).to_numpy(), X.slow_up.notna().to_numpy()

    # timing signals on the QQQ decision weeks; one set per way of fitting the Ridge model
    def sigs(pb, pa):
        return {"always": np.ones(len(X), bool), "sma": up, "sma+ridge": up & (pb > 0), "ridge_only": allw & (pa > 0)}, pa

    SIG = {"fwd": sigs(QS.walk(X, up, FEATS, 1, MODEL, fd)[0], QS.walk(X, allw, FEATS, 1, MODEL, fd)[0])}
    for k, tr, te in (("rev", (2009, 2025), (2001, 2008)), ("rev_b", (2008, 2025), (2001, 2007))):
        SIG[k] = sigs(RV.fit_apply(X, up, FEATS, MODEL, tr, te)[0], RV.fit_apply(X, allw, FEATS, MODEL, tr, te)[0])

    y2000 = (weeks.year >= 2000).to_numpy()[:, None]
    UNI = {"trend": P["tradable"] & ~np.isnan(S[140]) & (S[20] > S[140]) & y2000, "all": P["tradable"] & y2000}
    books = {}
    for un, c in UNI.items():
        books[(un, "every candidate")] = c
        for k in (10, 20, 30):
            books[(un, f"momentum top{k}")] = select(np.where(c, XF["ret252"], np.nan), c, k=k, every=4)
    G = Of[weeks.i_e_next.to_numpy()] / Of[weeks.i_e.to_numpy()]
    Gq = (1 + W.R.to_numpy())[:, None]

    rows = []
    for pn, (sk, y0, y1) in PERIODS.items():
        rq, rs = rows_of(W, y0, y1), rows_of(weeks, y0, y1)
        for tn in TIMINGS:
            sig = SIG[sk][0][tn]
            rows.append({"universe": "-", "book": "QQQ ETF", "timing": tn, "period": pn, **stats(sig[rq][:, None].astype(float), Gq[rq], W.iloc[rq])})
            on = pd.Series(sig, index=X.d).reindex(weeks.d).fillna(False).to_numpy(bool)
            for (un, bn), sel in books.items():
                rows.append({"universe": un, "book": bn, "timing": tn, "period": pn, **stats(capped_equal(sel[rs] & on[rs][:, None]), G[rs], weeks.iloc[rs])})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "grid.csv", index=False)
    ix = df.set_index(["universe", "book", "timing", "period"])

    head = []
    for label, key in HEADLINE:
        r = {"strategy": label}
        for pn in MAIN:
            s = ix.loc[(*key, pn)]
            r[pn] = f"{s.cagr:.1f} / {s.hold_cagr:.1f} / {s.sharpe:.2f}"
        f = ix.loc[(*key, "full 2005-2025")]
        r["invested % (2005-2025)"], r["max DD % (2005-2025)"] = f"{f.invested:.0f}", f"{f.maxdd:.0f}"
        r["max DD % (2001-2008)"] = f"{ix.loc[(*key, 'rev 2001-2008')].maxdd:.0f}"
        head.append(r)
    head = pd.DataFrame(head)
    head.to_csv(OUT / "headline.csv", index=False)

    # how the Ridge-only signal treats QQQ uptrend and downtrend weeks
    y, yr, diag = X.R1.to_numpy(), X.year.to_numpy(), []
    for pn, (sk, y0, y1) in PERIODS.items():
        (sg, pa), m = SIG[sk], allw & (yr >= y0) & (yr <= y1) & np.isfinite(y)
        k, dn = sg["ridge_only"], m & ~up
        diag.append({"period": pn, "uptrend_weeks": int((m & up).sum()), "uptrend_kept_pct": k[m & up].mean() * 100, "downtrend_weeks": int(dn.sum()),
                     "downtrend_kept_pct": k[dn].mean() * 100, "downtrend_kept_mean_bps": y[dn & k].mean() * 1e4,
                     "downtrend_dropped_mean_bps": y[dn & ~k].mean() * 1e4 if (dn & ~k).any() else np.nan,
                     "corr_forecast_return": np.corrcoef(pa[m], y[m])[0, 1]})
    diag = pd.DataFrame(diag)
    diag.to_csv(OUT / "ridge_only_diagnostics.csv", index=False)

    def wide(v, fmt):
        t = df.pivot_table(index=["universe", "book", "timing"], columns="period", values=v, sort=False)[list(PERIODS)].reset_index()
        return md_table(t, fmt)

    L = ["# Component ablation: SMA x Ridge x momentum (auto-generated by scripts/run_ablation.py)", "", __doc__.split("Usage")[0],
         "## Headline: CAGR / holding CAGR / Sharpe (percent per year; 5 bps)", "", md_table(head), "",
         "Rows: `SMA + Ridge + momentum` = universe `trend`, timing `sma+ridge`; `SMA + momentum` = universe `trend`, timing `sma`; "
         "`Ridge + momentum` = universe `all`, timing `ridge_only`; `Momentum` = universe `all`, timing `always`.", ""]
    for v, title, fmt in (("cagr", "CAGR (%)", "{:.1f}"), ("hold_cagr", "Holding CAGR (%)", "{:.1f}"), ("sharpe", "Sharpe (weekly returns)", "{:.2f}"),
                          ("maxdd", "Max drawdown (%, weekly marks)", "{:.1f}"), ("invested", "Invested weeks (%)", "{:.0f}")):
        L += [f"## Full grid: {title}", "", wide(v, fmt), ""]
    L += ["## Ridge-only signal: share of QQQ uptrend and downtrend weeks it keeps, and next-week QQQ return of the downtrend weeks", "", md_table(diag), ""]
    (OUT / "summary.md").write_text("\n".join(L))
    print(md_table(head))
    print()
    print(md_table(diag))


if __name__ == "__main__":
    main()
