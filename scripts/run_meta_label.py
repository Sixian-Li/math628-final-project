"""Trade-level meta-labeling on point-in-time Nasdaq-100 stocks (exploratory). Default SMA20/140.

A *trade* is a maximal run of consecutive weeks in which a stock is a candidate (tradable member,
features complete, SMA_fast > SMA_slow at the d_t close); it is entered at the open of e_t of its
first week and exited at the open after its last week. Weekly timing, capped equal weights
min(1/n, 10%) and the cost ledger are the same as in run_stocks.py.

At entry, a model predicts P(y = 1) from 7 own-stock features + their weekly cross-sectional ranks
among candidates + 7 QQQ features. A trade is taken (and held for its whole run) if p >= the
training mean of y (`pbar`). Label variants (src/ml_filter/meta.py):
  win    whole-trade return > 0
  win_w  same, training rows weighted by |trade return|
  mag    whole-trade return > median trade return of the training set
  cap26  return over the first <= 26 weeks > 0 (label matures after at most 26 weeks)
A label is usable for training only after it has matured.

Schemes:
  rolling  annual expanding refit 2003-2025; spec for year Y chosen by Brier on trades entered in
           the previous five years whose labels had matured before Y. Evaluated 2005-2025, 2019-2025.
  fixed    spec chosen on trades entered 2014-2018 (matured before 2019) with a model trained on
           labels matured before 2014; refit once on labels matured before 2019; frozen for 2019-2025.
Placebo: each live trade kept at random with the rule's own keep rate (200 seeds), same holding logic.
Also: `<model>_inv` takes exactly the trades the model rejects (p < pbar); `hivol` / `lovol` take trades
whose entry vol20 ranks in the top / bottom half of that week's candidates (no model).
Usage: .venv/bin/python scripts/run_meta_label.py [--sma 20_140]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ml_filter.data import load_daily, weekly_table  # noqa: E402
from ml_filter.meta import VARIANTS, Holdings, build_trades, fixed_predict, label_known, rolling_predict, sma_windows  # noqa: E402
from ml_filter.models import GB_GRID, LOGIT_GRID, RF_GRID  # noqa: E402
from ml_filter.portfolio import capped_equal, daily_curve, portfolio_metrics, weekly_ledger, weekly_metrics  # noqa: E402
from ml_filter.universe import build_panel, load_qqq  # noqa: E402
from run_experiments import md_table  # noqa: E402

PRED_YEARS = list(range(2003, 2026))
LOOKBACK_YEARS = 5
COSTS = [0, 5]
N_RANDOM = 200
GRIDS = {"logit": LOGIT_GRID, "rf": RF_GRID, "gb": GB_GRID}


def brier(p, y):
    return float(((p - y) ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sma", default="20_140")
    FAST, SLOW = (int(x) for x in ap.parse_args().sma.split("_"))
    OUT = ROOT / "results" / "meta_label" / f"sma{FAST}_{SLOW}"
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    P = build_panel()
    weeks, Of, Cf, cal = P["weeks"], P["Of"], P["Cf"], P["cal"]
    nW, nJ = P["R"].shape
    S = sma_windows(P, [FAST, SLOW])
    qd = load_daily("QQQ")
    qw = weekly_table(qd).set_index("d")
    qc_ = qd.set_index("date")["close"]
    q_sig = (qc_.rolling(FAST).mean() > qc_.rolling(SLOW).mean()).reindex(weeks.d).to_numpy()
    T, _ = build_trades(P, S[FAST], S[SLOW], qw)
    fd = weeks.groupby("year")["d"].min().to_dict()
    H = Holdings(T, (nW, nJ))
    print(f"[{time.time() - t0:5.1f}s] trades {len(T):,}; win rate {T.ret.gt(0).mean():.3f}; median weeks {T.weeks_held.median():.0f}")

    # ---- models ----
    specs = [(v, s) for v in VARIANTS for g in GRIDS.values() for s in g]
    roll = dict(Parallel(n_jobs=12)(delayed(lambda v, s: ((v, s.label), rolling_predict(T, v, s, fd, PRED_YEARS)))(v, s) for v, s in specs))
    preds, sel_rows = {}, []
    for v in VARIANTS:
        for fam, grid in GRIDS.items():
            p = pd.Series(np.nan, index=T.index)
            any_pred = roll[(v, grid[0].label)]
            for yr in PRED_YEARS:
                te = (T.entry_year == yr) | ((yr == PRED_YEARS[0]) & (T.entry_year < yr))
                win = T.entry_year.between(max(PRED_YEARS[0], yr - LOOKBACK_YEARS), yr - 1) & (T[VARIANTS[v]["time"]] <= fd[yr]) & any_pred.y.notna()
                best = grid[0] if win.sum() < 50 else min(grid, key=lambda s: (brier(roll[(v, s.label)].p[win], any_pred.y[win]), grid.index(s)))
                p[te] = roll[(v, best.label)].p[te]
                sel_rows.append({"variant": v, "scheme": "rolling", "family": fam, "year": yr, "spec": best.label})
            preds[("rolling", v, fam)] = (p, any_pred.y, any_pred.base)
            # fixed
            tr_sel = T.entry_year.between(2014, 2018) & (T[VARIANTS[v]["time"]] <= fd[2019])
            sc = {}
            for s in grid:
                pr = fixed_predict(T, v, s, fd, 2014)
                sc[s.label] = brier(pr.p[tr_sel], pr.y[tr_sel])
            best = min(grid, key=lambda s: (sc[s.label], grid.index(s)))
            for s in grid:
                sel_rows.append({"variant": v, "scheme": "fixed", "family": fam, "spec": s.label, "val_brier": sc[s.label], "selected": s is best})
            pr = fixed_predict(T, v, best, fd, 2019)
            preds[("fixed", v, fam)] = (pr.p, pr.y, pr.base)
    pd.DataFrame(sel_rows).to_csv(OUT / "selection.csv", index=False)
    print(f"[{time.time() - t0:5.1f}s] models done")

    # ---- portfolios ----
    qo, qc = load_qqq(cal)

    def period_rows(y0, y1):
        return np.flatnonzero(((weeks.year >= y0) & (weeks.year <= y1) & (weeks.e_next <= pd.Timestamp(f"{y1}-12-31"))).to_numpy())

    def run(W, rows, cost, o=Of, c=Cf, curve=True):
        wk = weeks.iloc[rows].reset_index(drop=True)
        led = weekly_ledger(W[rows], o[wk.i_e_next] / o[wk.i_e], cost)
        if not curve:
            return weekly_metrics(led, wk)
        cv = daily_curve(W[rows], wk, o, c, led, cal)
        return portfolio_metrics(cv, W[rows], led, wk), cv

    member = capped_equal(P["tradable"] & (weeks.year >= 2000).to_numpy()[:, None])
    s0 = H.weights(np.ones(len(T), bool))
    met, prow, curves = [], [], {}
    rng = np.random.default_rng(628)
    plans = [("rolling", "2005-2025", (2005, 2025)), ("rolling", "2019-2025", (2019, 2025)), ("fixed", "2019-2025", (2019, 2025))]
    for scheme, pname, (y0, y1) in plans:
        rows = period_rows(y0, y1)
        live = ((T.k1 >= rows[0]) & (T.k0 <= rows[-1])).to_numpy()
        bench = {"QQQ_buy_hold": (np.ones((nW, 1)), qo, qc), f"QQQ_sma{FAST}_{SLOW}": (q_sig.astype(float)[:, None], qo, qc),
                 "EW_members": (member, Of, Cf), "S0_all_trades": (s0, Of, Cf)}
        for cost in COSTS:
            for name, (W, o, c) in bench.items():
                m_, cv = run(W, rows, cost, o, c)
                met.append({"scheme": scheme, "period": pname, "cost_bps": cost, "variant": "-", "strategy": name, **m_})
                if cost == 5:
                    curves[(scheme, pname, name)] = cv
        # rules: (variant, strategy, keep). `<model>_inv` takes exactly the trades the model rejects;
        # hivol / lovol keep trades whose entry vol20 is in the top / bottom half of that week's candidates (no model).
        rules = [("-", "hivol", (T.vol20_rk >= 0.5).to_numpy() & live), ("-", "lovol", (T.vol20_rk < 0.5).to_numpy() & live)]
        for v in VARIANTS:
            for fam in GRIDS:
                p, y, base = preds[(scheme, v, fam)]
                keep = (p >= base).to_numpy() & live
                rules += [(v, fam, keep), (v, f"{fam}_inv", (p < base).to_numpy() & live)]
                ent = (T.entry_year.between(y0, y1) & label_known(T, v) & p.notna()).to_numpy()
                k = keep[ent]
                yy, pp, ret = y[ent].to_numpy(), p[ent].to_numpy(), T.ret[ent].to_numpy()
                prow.append({"scheme": scheme, "period": pname, "variant": v, "model": fam, "trades": int(ent.sum()),
                             "y_rate": yy.mean(), "auc": roc_auc_score(yy, pp),
                             "brier_skill_pct": (1 - brier(pp, yy) / brier(base[ent].to_numpy(), yy)) * 100,
                             "keep_pct": k.mean() * 100, "win_rate_kept": (ret[k] > 0).mean(), "win_rate_dropped": (ret[~k] > 0).mean(),
                             "mean_ret_kept_pct": ret[k].mean() * 100, "mean_ret_dropped_pct": ret[~k].mean() * 100})
        for v, name, keep in rules:
            q = keep.sum() / live.sum()
            draws = [H.weights((rng.random(len(T)) < q) & live) for _ in range(N_RANDOM)]
            W = H.weights(keep)
            for cost in COSTS:
                m_, cv = run(W, rows, cost)
                r = pd.DataFrame([run(d, rows, cost, curve=False) for d in draws])
                met.append({"scheme": scheme, "period": pname, "cost_bps": cost, "variant": v, "strategy": name, **m_,
                            "keep_pct_live": q * 100, "placebo_median_cagr": float(r.cagr_pct.median()),
                            "placebo_pctile_cagr": float((r.cagr_pct < m_["cagr_pct"]).mean() * 100),
                            "placebo_pctile_sharpe": float((r.sharpe_weekly < m_["sharpe_weekly"]).mean() * 100)})
                if cost == 5:
                    curves[(scheme, pname, f"{v}_{name}")] = cv
        print(f"[{time.time() - t0:5.1f}s] {scheme} {pname} done")

    met, prow = pd.DataFrame(met), pd.DataFrame(prow)
    met.to_csv(OUT / "metrics.csv", index=False)
    prow.to_csv(OUT / "prediction_metrics.csv", index=False)
    T[["j", "k0", "k1", "entry", "exit", "weeks_held", "ret", "ret_cap", "completed"]].assign(file=np.array(P["files"])[T.j]).to_csv(OUT / "trades.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, key in zip(axes, [("rolling", "2005-2025"), ("fixed", "2019-2025")]):
        for name in ["QQQ_buy_hold", f"QQQ_sma{FAST}_{SLOW}", "EW_members", "S0_all_trades", "win_gb", "mag_gb", "win_w_gb", "cap26_gb"]:
            cv = curves[(*key, name)]
            ax.plot(pd.to_datetime(cv.date), cv.equity, label=name, lw=1.6 if name in ("S0_all_trades", "QQQ_buy_hold") else 1.0)
        ax.set_yscale("log"); ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2)
        ax.set_title(f"Meta-label SMA{FAST}/{SLOW} — {key[0]}, {key[1]}, 5 bps")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "equity.png", dpi=150); plt.close(fig)

    tr = T[T.completed]
    ts = pd.DataFrame([{"entered": n, "trades": int(m.sum()), "win_rate": (tr.ret[m] > 0).mean(), "mean_ret_pct": tr.ret[m].mean() * 100,
                        "median_ret_pct": tr.ret[m].median() * 100, "median_weeks": tr.weeks_held[m].median()}
                       for n, m in [("2000-2025", tr.entry_year <= 2025), ("2005-2025", tr.entry_year.between(2005, 2025)),
                                    ("2019-2025", tr.entry_year.between(2019, 2025))]])
    cols = ["variant", "strategy", "cagr_pct", "holding_cagr_pct", "sharpe", "max_dd_pct", "exposure_pct", "avg_names", "annual_turnover",
            "placebo_median_cagr", "placebo_pctile_cagr", "placebo_pctile_sharpe"]
    Ls = [f"# Meta-labeling, PIT Nasdaq-100, SMA{FAST}/{SLOW} (auto-generated by scripts/run_meta_label.py)", "", __doc__.split("Usage")[0],
          "## Trades (completed, by entry year)", "", md_table(ts), ""]
    for scheme, pname, _ in plans:
        for cost in (5, 0):
            Ls += [f"## {scheme}, {pname}, {cost} bps", "", md_table(met[(met.scheme == scheme) & (met.period == pname) & (met.cost_bps == cost)][cols]), ""]
    Ls += ["## Trade-level prediction (trades entered in the period, label matured)", "", md_table(prow, "{:.3f}"), ""]
    (OUT / "summary.md").write_text("\n".join(Ls))
    print(f"[{time.time() - t0:5.1f}s] written -> {OUT}")


if __name__ == "__main__":
    main()
