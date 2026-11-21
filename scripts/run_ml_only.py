"""ML-only timing on QQQ: no SMA gate, the model decides every week (exploratory, added after the main study).

Every week with complete features is a candidate. The model predicts P(R > 0) for the next
open-to-open week and the account holds QQQ when the forecast passes a threshold:
  p50   hold if p >= 0.50
  pbar  hold if p >= the training up-share (i.e. "a better-than-average week")
Same 7 features, grids, timing, ledger and costs as the main study.

Two estimation schemes:
  fixed    train once, never refit. Spec chosen on validation 2014-2018 with a model trained on
           2000-2013; the chosen spec is refit once on 2000-2018 and frozen for 2019-2025.
  rolling  annual expanding-window refits 2003-2025; the spec for year Y has the lowest Brier
           over the previous five years of out-of-sample forecasts. Reported for 2005-2025
           and for the 2019-2025 slice.
Usage: .venv/bin/python scripts/run_ml_only.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ml_filter.backtest import equity_curve, exposure_matched, metrics, yearly_returns  # noqa: E402
from ml_filter.data import FEATURES, base_signal, load_daily, weekly_table  # noqa: E402
from ml_filter.models import GB_GRID, LOGIT_GRID, RF_GRID, brier, fit_predict, training_mask, walk_forward  # noqa: E402
from run_experiments import md_table, period_weeks  # noqa: E402

GRIDS = {"logit": LOGIT_GRID, "rf": RF_GRID, "gb": GB_GRID}
COSTS = [0, 5]
PRED_YEARS = list(range(2003, 2026))
LOOKBACK_YEARS = 5
OUT = ROOT / "results" / "ml_only"


def evaluate(W, D, ev, targets, period, scheme, rows, curves):
    idx = ev.index.to_numpy()
    for cost in COSTS:
        for name, w in targets.items():
            cur = equity_curve(D, ev, w[idx], cost)
            rows.append({"scheme": scheme, "period": period, "cost_bps": cost, "strategy": name, **metrics(cur)})
            if cost == 5:
                curves[(scheme, period, name)] = cur
            if cost == 0 and name == "buy_hold":
                bh0 = cur
    exps = {r["strategy"]: r["exposure_pct"] / 100 for r in rows[-len(targets):]}  # 5 bps rows of this block
    for name, exp in exps.items():
        if name == "buy_hold":
            continue
        rows.append({"scheme": scheme, "period": period, "cost_bps": 0, "strategy": f"expmatch_{name}",
                     **metrics(exposure_matched(bh0, exp))})


def pred_row(W, mask, p, base, scheme, period, key, spec_label):
    y = W.loc[mask, "y"].to_numpy()
    R = W.loc[mask, "R"].to_numpy()
    p = np.asarray(p)
    base = np.asarray(base)
    k50, kb = p >= 0.5, p >= base
    return {
        "scheme": scheme, "period": period, "model": key, "spec": spec_label, "n": len(y),
        "up_share_pct": y.mean() * 100, "auc": roc_auc_score(y, p),
        "brier": brier(p, y), "brier_const": brier(base, y), "brier_skill_pct": (1 - brier(p, y) / brier(base, y)) * 100,
        "p_mean": p.mean(), "p_sd": p.std(), "in_pct_p50": k50.mean() * 100, "in_pct_pbar": kb.mean() * 100,
        "meanR_in_pbar_bps": R[kb].mean() * 1e4 if kb.any() else np.nan,
        "meanR_out_pbar_bps": R[~kb].mean() * 1e4 if (~kb).any() else np.nan,
    }


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    D = load_daily("QQQ")
    W = weekly_table(D)
    allw = W[FEATURES].notna().all(axis=1).to_numpy()
    sma = base_signal(W, "sma20_60").to_numpy().astype(bool) & allw
    rows, preds, sel, curves = [], [], [], {}

    # ------------------------------------------------------------ fixed (train once)
    val = period_weeks(W, list(range(2014, 2019))).index.to_numpy()
    vmask = np.zeros(len(W), bool)
    vmask[val] = True
    vmask &= allw
    tr_sel = training_mask(W, allw, 2014)
    tr_fin = training_mask(W, allw, 2019)
    ev = period_weeks(W, list(range(2019, 2026)))
    tmask = np.zeros(len(W), bool)
    tmask[ev.index] = True
    tmask &= allw
    targets = {"buy_hold": np.ones(len(W)), "sma20_60": sma.astype(float)}
    base_fin = W.loc[tr_fin, "y"].mean()
    for key, grid in GRIDS.items():
        best, best_s = None, np.inf
        for spec in grid:
            s = brier(fit_predict(W, tr_sel, vmask, FEATURES, spec), W.loc[vmask, "y"].to_numpy())
            sel.append({"scheme": "fixed", "model": key, "spec": spec.label, "val_brier_2014_2018": s})
            if s < best_s - 1e-12:
                best, best_s = spec, s
        p = fit_predict(W, tr_fin, tmask, FEATURES, best)
        full = np.full(len(W), np.nan)
        full[tmask] = p
        targets[f"{key}_p50"] = (full >= 0.5).astype(float)
        targets[f"{key}_pbar"] = (full >= base_fin).astype(float)
        preds.append(pred_row(W, tmask, p, np.full(len(p), base_fin), "fixed", "2019-2025", key, best.label))
    evaluate(W, D, ev, targets, "2019-2025", "fixed", rows, curves)
    print(f"[{time.time() - t0:5.1f}s] fixed done")

    # ------------------------------------------------------------ rolling (annual refit)
    wf = {(key, s.label): walk_forward(W, allw, PRED_YEARS, FEATURES, s) for key, g in GRIDS.items() for s in g}
    base_rate = next(iter(wf.values())).base_rate
    proba = {}
    for key, grid in GRIDS.items():
        pr = pd.Series(np.nan, index=W.index)
        for yr in range(2005, 2026):
            win = W["year"].between(yr - LOOKBACK_YEARS, yr - 1) & (W["year"] >= PRED_YEARS[0])
            scores = {}
            for s in grid:
                p = wf[(key, s.label)].proba
                m = win & p.notna()
                scores[s.label] = brier(p[m].to_numpy(), W.loc[m, "y"].to_numpy())
            best = min(grid, key=lambda s: scores[s.label])  # first minimum = simplest
            m = W["year"] == yr
            pr[m] = wf[(key, best.label)].proba[m]
            sel.append({"scheme": "rolling", "model": key, "year": yr, "spec": best.label, "trailing_brier": scores[best.label]})
        proba[key] = pr
    for period, (y0, y1) in {"2005-2025": (2005, 2025), "2019-2025": (2019, 2025)}.items():
        ev = period_weeks(W, list(range(y0, y1 + 1)))
        targets = {"buy_hold": np.ones(len(W)), "sma20_60": sma.astype(float)}
        for key, pr in proba.items():
            targets[f"{key}_p50"] = (pr >= 0.5).to_numpy().astype(float)
            targets[f"{key}_pbar"] = (pr >= base_rate).to_numpy().astype(float)
            m = np.zeros(len(W), bool)
            m[ev.index] = True
            m &= pr.notna().to_numpy()
            preds.append(pred_row(W, m, pr[m], base_rate[m], "rolling", period, key, "trailing-5y Brier"))
        evaluate(W, D, ev, targets, period, "rolling", rows, curves)
    print(f"[{time.time() - t0:5.1f}s] rolling done")

    m = pd.DataFrame(rows)
    pred = pd.DataFrame(preds)
    m.to_csv(OUT / "metrics.csv", index=False)
    pred.to_csv(OUT / "prediction_metrics.csv", index=False)
    pd.DataFrame(sel).to_csv(OUT / "selection.csv", index=False)
    yr = [{"strategy": n, "year": y, "return_pct": r} for (s, p, n), c in curves.items() if s == "rolling" and p == "2005-2025"
          for y, r in yearly_returns(c).items()]
    pd.DataFrame(yr).pivot(index="year", columns="strategy", values="return_pct").to_csv(OUT / "rolling_yearly_returns_5bps.csv")

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, (scheme, period) in zip(axes, [("fixed", "2019-2025"), ("rolling", "2005-2025")]):
        for name in ["buy_hold", "sma20_60", "logit_p50", "rf_p50", "gb_p50", "logit_pbar", "rf_pbar", "gb_pbar"]:
            c = curves[(scheme, period, name)]
            ax.plot(pd.to_datetime(c["date"]), c["equity"], lw=1.6 if name in ("buy_hold", "sma20_60") else 0.9,
                    ls="--" if name.endswith("pbar") else "-", label=name)
        ax.set_yscale("log")
        ax.set_title(f"QQQ ML-only timing — {scheme}, {period}, 5 bps")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "equity.png", dpi=150)
    plt.close(fig)

    cols = ["strategy", "cagr_pct", "holding_cagr_pct", "max_dd_pct", "sharpe", "exposure_pct", "trades_per_year"]
    L = ["# ML-only timing on QQQ (auto-generated by scripts/run_ml_only.py)", "", __doc__.split("Usage")[0], ""]
    L += ["`expmatch_X` (0 bps rows) = buy-and-hold scaled to X's average exposure, rebalanced daily, no cost: "
          "what X's exposure alone would earn with zero timing skill.", ""]
    for (scheme, period) in [("fixed", "2019-2025"), ("rolling", "2019-2025"), ("rolling", "2005-2025")]:
        for cost in COSTS:
            sub = m[(m.scheme == scheme) & (m.period == period) & (m.cost_bps == cost)]
            L += [f"## {scheme}, {period}, {cost} bps", "", md_table(sub[cols]), ""]
    L += ["## Prediction quality", "", md_table(pred, "{:.3f}"), ""]
    (OUT / "summary.md").write_text("\n".join(L))
    print(f"[{time.time() - t0:5.1f}s] written -> {OUT}")


if __name__ == "__main__":
    main()
