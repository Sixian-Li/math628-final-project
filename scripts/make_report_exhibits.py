"""Small data files behind Figures 1 to 5, Table 9 and the single-stock example of the report.

scripts/make_report_figures.py and scripts/make_report_tables.py only read saved results. The series below need the full pipeline
(stock panel, market-state features, the annual walk-forward Ridge filter), so they are written once here:
  qqq_daily.csv       QQQ close with SMA20 and SMA140
  timing_weeks.csv    one row per weekly decision: QQQ next-week return, SMA20/140 state, Ridge forecast and keep flag
  ridge_coefficients.csv   intercept and coefficients of each annual Ridge fit (standardized features, target in standard deviations)
  equity_weekly.csv   weekly equity of the eight ablation strategies, 2005-2025 (same ledger as scripts/run_ablation.py)
  example_daily.csv   PLTR open and close with SMA20 and SMA140, 2025 (the single-stock example in the report)
  example_weeks.csv   PLTR weekly opens and return, own trend state and momentum top-10 membership, next to the QQQ signals
Usage: .venv/bin/python scripts/make_report_exhibits.py
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

import run_ablation as AB  # noqa: E402
import run_qqq_state as QS  # noqa: E402
from ml_filter.data import load_daily, weekly_table  # noqa: E402
from ml_filter.meta import sma_windows  # noqa: E402
from ml_filter.portfolio import capped_equal, weekly_ledger  # noqa: E402
from ml_filter.universe import build_panel  # noqa: E402
from ml_filter.xfeat import market_features, stock_features  # noqa: E402
from run_stock_rank import select  # noqa: E402

OUT = ROOT / "reports" / "exhibits"
Y0, Y1, K = 2005, 2025, 10
EXAMPLE = "PLTR"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P = build_panel()
    weeks, Of, Cf, cal, tick = P["weeks"], P["Of"], P["Cf"], P["cal"], np.array(P["tickers"])
    S = sma_windows(P, [1, 20, 50, 140, 200])
    ret252 = stock_features(P)["ret252"]
    D = load_daily("QQQ")
    W = weekly_table(D)
    X = market_features(P, S)
    X["year"] = W.year.to_numpy()
    fd = W.groupby("year")["d"].min().to_dict()
    up, allw = (X.slow_up == 1).to_numpy(), X.slow_up.notna().to_numpy()
    pred, pred_all = QS.walk(X, up, AB.FEATS, 1, AB.MODEL, fd)[0], QS.walk(X, allw, AB.FEATS, 1, AB.MODEL, fd)[0]
    keep, ridge_only, always = up & (pred > 0), allw & (pred_all > 0), np.ones(len(X), bool)

    # the fitted model: one row per annual refit of the reference configuration
    y, yr_, A = X["R1"].to_numpy(), X["year"].to_numpy(), X[AB.FEATS].to_numpy()
    fits, gap = [], 0.0
    for yr in QS.YEARS:
        tr, te = up & (yr_ >= 2000) & np.isfinite(y) & (X["t1"] <= fd[yr]).to_numpy(), up & (yr_ == yr)
        if tr.sum() < 60 or not te.any():
            continue
        sd = y[tr].std()
        m = QS.MODELS[AB.MODEL]()
        m.fit(A[tr], y[tr] / sd)
        z, f = y[tr] / sd, m.predict(A[tr])
        gap = max(gap, np.abs(m.predict(A[te]) - pred[te]).max())          # must reproduce the walk-forward forecasts
        fits.append({"year": yr, "n_train": int(tr.sum()), "sd_target": sd, "intercept": m[-1].intercept_,
                     "r2_in_sample": 1 - ((z - f) ** 2).sum() / ((z - z.mean()) ** 2).sum(), **dict(zip(AB.FEATS, m[-1].coef_))})
    pd.DataFrame(fits).to_csv(OUT / "ridge_coefficients.csv", index=False, float_format="%.6f")
    print(f"ridge_coefficients.csv: {len(fits)} annual fits; they reproduce the walk-forward forecasts with a maximum absolute difference of {gap:.1e}")

    # QQQ daily series and weekly signals
    c = D.set_index("date")["close"]
    q = pd.DataFrame({"close": c, "sma20": c.rolling(20).mean(), "sma140": c.rolling(140).mean()}).loc["2004-01-01":f"{Y1}-12-31"]
    q.to_csv(OUT / "qqq_daily.csv", float_format="%.4f")
    tw = pd.DataFrame({"d": W.d, "e": W.e, "e_next": W.e_next, "year": W.year, "R": W.R, "up": up, "forecast": pred, "keep": keep})
    tw[tw.year.between(Y0, Y1)].to_csv(OUT / "timing_weeks.csv", index=False, float_format="%.6f")

    # weekly equity of the eight strategies on the ablation ledger
    y2000 = (weeks.year >= 2000).to_numpy()[:, None]
    trend, every = P["tradable"] & ~np.isnan(S[140]) & (S[20] > S[140]) & y2000, P["tradable"] & y2000
    book = {u: select(np.where(cnd, ret252, np.nan), cnd, k=K, every=4) for u, cnd in (("trend", trend), ("all", every))}
    G = Of[weeks.i_e_next.to_numpy()] / Of[weeks.i_e.to_numpy()]
    Gq = (1 + W.R.to_numpy())[:, None]
    rq, rs = AB.rows_of(W, Y0, Y1), AB.rows_of(weeks, Y0, Y1)

    def on(sig):
        return pd.Series(sig, index=X.d).reindex(weeks.d).fillna(False).to_numpy(bool)

    spec = {f"SMA + Ridge + momentum top {K}": ("trend", keep), f"SMA + momentum top {K}": ("trend", up),
            f"Ridge + momentum top {K}": ("all", ridge_only), f"Momentum top {K}": ("all", always),
            "SMA + Ridge (QQQ)": (None, keep), "SMA (QQQ)": (None, up), "Ridge (QQQ)": (None, ridge_only), "QQQ buy-and-hold": (None, always)}
    eq = {}
    for name, (u, sig) in spec.items():
        Wt = sig[rq][:, None].astype(float) if u is None else capped_equal(book[u][rs] & on(sig)[rs][:, None])
        eq[name] = weekly_ledger(Wt, Gq[rq] if u is None else G[rs], AB.COST)["eq_end"]
    E = pd.DataFrame(eq, index=pd.Index(W.e_next.iloc[rq].to_numpy(), name="date"))
    E.to_csv(OUT / "equity_weekly.csv", float_format="%.6f")

    # check against results/ablation/grid.csv (full 2005-2025)
    ab = pd.read_csv(ROOT / "results" / "ablation" / "grid.csv").set_index(["universe", "book", "timing", "period"])
    amap = {f"SMA + Ridge + momentum top {K}": ("trend", f"momentum top{K}", "sma+ridge"), f"SMA + momentum top {K}": ("trend", f"momentum top{K}", "sma"),
            f"Ridge + momentum top {K}": ("all", f"momentum top{K}", "ridge_only"), f"Momentum top {K}": ("all", f"momentum top{K}", "always"),
            "SMA + Ridge (QQQ)": ("-", "QQQ ETF", "sma+ridge"), "SMA (QQQ)": ("-", "QQQ ETF", "sma"), "Ridge (QQQ)": ("-", "QQQ ETF", "ridge_only"),
            "QQQ buy-and-hold": ("-", "QQQ ETF", "always")}
    yrs = (W.e_next.iloc[rq[-1]] - W.e.iloc[rq[0]]).days / 365.25
    gap = max(abs((E[n].iloc[-1] ** (1 / yrs) - 1) * 100 - ab.loc[(*k, "full 2005-2025"), "cagr"]) for n, k in amap.items())
    print(f"equity_weekly.csv reproduces the 2005-2025 CAGR of results/ablation/grid.csv; maximum absolute difference {gap:.1e}")

    # single-stock example: PLTR in 2025
    j = [i for i in np.flatnonzero(tick == EXAMPLE) if np.isfinite(Cf[cal.get_loc("2025-08-11"), i])][0]
    n = pd.Series(Cf[:, j], index=cal, name="close")
    nd = pd.DataFrame({"open": pd.Series(Of[:, j], index=cal), "close": n, "sma20": n.rolling(20).mean(), "sma140": n.rolling(140).mean()})
    nd.loc["2025-01-01":"2025-12-31"].to_csv(OUT / "example_daily.csv", float_format="%.4f", index_label="date")
    ie, ien = weeks.i_e.to_numpy(), weeks.i_e_next.to_numpy()
    nw = pd.DataFrame({"d": weeks.d, "e": weeks.e, "e_next": weeks.e_next, "open_e": Of[ie, j], "open_next": Of[ien, j], "ret": G[:, j] - 1,
                       "own_uptrend": trend[:, j], "in_top10": book["trend"][:, j], "qqq_up": on(up), "keep": on(keep),
                       "forecast": pd.Series(pred, index=X.d).reindex(weeks.d).to_numpy()})
    nw[(nw.e >= "2025-01-01") & (nw.e <= "2025-12-31")].to_csv(OUT / "example_weeks.csv", index=False, float_format="%.6f")
    print("wrote", ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
