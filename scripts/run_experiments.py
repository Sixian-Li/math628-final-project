"""Run every prespecified case of the proposal and write results/ (CSV, figures, summary.md).

Usage (from the project root):
    .venv/bin/python scripts/run_experiments.py
"""

from __future__ import annotations

import json
import os
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

from ml_filter.backtest import equity_curve, exposure_matched, metrics, yearly_returns  # noqa: E402
from ml_filter.data import FEATURES, RISK_FEATURES, TREND_FEATURES, base_signal, load_daily, weekly_table  # noqa: E402
from ml_filter.models import (  # noqa: E402
    GB_GRID,
    LOGIT_GRID,
    RF_GRID,
    THRESHOLD,
    brier,
    select,
    walk_forward,
)

TRAIN_YEARS = list(range(2000, 2014))
VAL_YEARS = list(range(2014, 2019))
TEST_YEARS = list(range(2019, 2026))
COSTS = [0, 5, 10]
MAIN_COST = 5
CASES = [
    ("QQQ_SMA20_60", "QQQ", "sma20_60"),
    ("QQQ_SMA10_30", "QQQ", "sma10_30"),
    ("QQQ_SMA40_120", "QQQ", "sma40_120"),
    ("SPY_SMA20_60", "SPY", "sma20_60"),
    ("QQQ_MOM60", "QQQ", "mom60"),
]
MAIN = "QQQ_SMA20_60"
OUT = ROOT / "results"
# Optional single-case rerun, e.g. ML_CASE=QQQ:sma20_140 (exploratory; writes to results/cases/<case>/)
if os.environ.get("ML_CASE"):
    _sym, _rule = os.environ["ML_CASE"].split(":")
    MAIN = f"{_sym}_{_rule.upper()}"
    CASES = [(MAIN, _sym, _rule)]
    OUT = ROOT / "results" / "cases" / MAIN
MAIN_SYM, MAIN_RULE = next((sym, rule) for c, sym, rule in CASES if c == MAIN)
FIG = OUT / "figures"


def period_weeks(weeks: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Weeks whose holding interval starts in ``years`` and whose label ends within the last year."""
    end = pd.Timestamp(f"{years[-1]}-12-31")
    m = weeks["year"].isin(years) & (weeks["e_next"] <= end)
    return weeks[m]


def candidates(weeks: pd.DataFrame, rule: str) -> np.ndarray:
    complete = weeks[FEATURES].notna().all(axis=1).to_numpy()
    return base_signal(weeks, rule).to_numpy().astype(bool) & complete


def prediction_row(case, name, weeks, ev_idx, cand, keep, proba=None, base_rate=None):
    m = cand[ev_idx]
    sub = weeks.loc[ev_idx[m]]
    k = keep[ev_idx][m].astype(bool)
    R, y = sub["R"].to_numpy(), sub["y"].to_numpy()
    row = {
        "case": case,
        "strategy": name,
        "n_candidates": int(m.sum()),
        "up_share_pct": y.mean() * 100,
        "retained_pct": k.mean() * 100,
        "mean_R_kept_pct": R[k].mean() * 100 if k.any() else np.nan,
        "mean_R_filtered_pct": R[~k].mean() * 100 if (~k).any() else np.nan,
        "filtered_up_n": int(((~k) & (R > 0)).sum()),
        "filtered_up_sum_pct": R[(~k) & (R > 0)].sum() * 100,
        "filtered_down_n": int(((~k) & (R <= 0)).sum()),
        "filtered_down_sum_pct": R[(~k) & (R <= 0)].sum() * 100,
    }
    if proba is not None:
        p = proba.loc[sub.index].to_numpy()
        c = base_rate.loc[sub.index].to_numpy()
        row.update(
            {
                "brier": brier(p, y),
                "brier_const": brier(c, y),
                "brier_skill_pct": (1 - brier(p, y) / brier(c, y)) * 100,
                "auc": roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan,
                "accuracy_pct": ((p >= THRESHOLD) == y).mean() * 100,
            }
        )
    return row


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    daily = {s: load_daily(s) for s in ("QQQ", "SPY")}
    weeks = {s: weekly_table(d) for s, d in daily.items()}

    # ---- 1. Hyperparameter selection on the main case, validation 2014-2018 (Brier) ----
    W = weeks[MAIN_SYM]
    cand_main = candidates(W, MAIN_RULE)
    chosen, sel_tables = {}, []
    for key, feats, grid in [
        ("logit", FEATURES, LOGIT_GRID),
        ("rf", FEATURES, RF_GRID),
        ("gb", FEATURES, GB_GRID),
        ("logit_trend", TREND_FEATURES, LOGIT_GRID),
        ("logit_risk", RISK_FEATURES, LOGIT_GRID),
    ]:
        spec, table = select(W, cand_main, VAL_YEARS, feats, grid)
        table.insert(0, "strategy", key)
        table["selected"] = table["model"] == spec.label
        chosen[key] = (spec, feats)
        sel_tables.append(table)
    selection = pd.concat(sel_tables, ignore_index=True)
    selection.to_csv(OUT / "validation_selection.csv", index=False)
    print(f"[{time.time() - t0:5.1f}s] selection done:", {k: v[0].label for k, v in chosen.items()})

    # ---- 2. Final evaluation 2019-2025 for every case ----
    metric_rows, pred_rows, count_rows, yearly_rows, curve_rows, coef_rows = [], [], [], [], [], []
    for case, sym, rule in CASES:
        Wc, D = weeks[sym], daily[sym]
        cand = candidates(Wc, rule)
        for label, yrs in [("train_2000_2013", TRAIN_YEARS), ("val_2014_2018", VAL_YEARS), ("test_2019_2025", TEST_YEARS)]:
            pw = period_weeks(Wc, yrs)
            c = cand[pw.index]
            count_rows.append(
                {"case": case, "period": label, "weeks": len(pw), "candidate_weeks": int(c.sum()),
                 "candidate_share_pct": c.mean() * 100, "candidate_up_share_pct": pw.loc[c, "y"].mean() * 100}
            )
        ev = period_weeks(Wc, TEST_YEARS)
        ev_idx = ev.index.to_numpy()
        model_keys = ["logit", "rf"] + (["gb", "logit_trend", "logit_risk"] if case == MAIN else [])
        wf0 = walk_forward(Wc, cand, TEST_YEARS, FEATURES, None)
        targets = {
            "buy_hold": np.ones(len(Wc)),
            "base": cand.astype(float),
            "vol_filter": (cand & (Wc["vol20"] <= wf0.vol_q).to_numpy()).astype(float),
        }
        wfs = {}
        for key in model_keys:
            spec, feats = chosen[key]
            wf = walk_forward(Wc, cand, TEST_YEARS, feats, spec)
            wfs[key] = wf
            targets[key] = (cand & (wf.proba >= THRESHOLD).to_numpy()).astype(float)
            if case == MAIN and key in ("logit", "rf", "gb"):
                mdl = wf.last_model
                if key == "logit":
                    vals = mdl[-1].coef_[0]
                    kind = "standardized_coef"
                else:
                    vals = mdl.feature_importances_
                    kind = "impurity_importance"
                for f, v in zip(feats, vals):
                    coef_rows.append({"model": key, "fit_year": TEST_YEARS[-1], "kind": kind, "feature": f, "value": v})

        curves = {}
        for cost in COSTS:
            for name, w in targets.items():
                cur = equity_curve(D, ev, w[ev_idx], cost)
                curves[(name, cost)] = cur
                metric_rows.append({"case": case, "strategy": name, "cost_bps": cost, **metrics(cur)})
        bh0 = curves[("buy_hold", 0)]
        for name in ["base", "vol_filter"] + model_keys:
            exp = metrics(curves[(name, MAIN_COST)])["exposure_pct"] / 100
            cur = exposure_matched(bh0, exp)
            curves[(f"exposure_matched_{name}", MAIN_COST)] = cur
            metric_rows.append({"case": case, "strategy": f"exposure_matched_{name}", "cost_bps": MAIN_COST, **metrics(cur)})

        for (name, cost), cur in curves.items():
            if cost != MAIN_COST:
                continue
            if not name.startswith("exposure_matched"):
                for yr, r in yearly_returns(cur).items():
                    yearly_rows.append({"case": case, "strategy": name, "year": yr, "return_pct": r})
            curve_rows.append(cur.assign(case=case, strategy=name)[["case", "strategy", "date", "equity", "position"]])

        pred_rows.append(prediction_row(case, "vol_filter", Wc, ev_idx, cand, targets["vol_filter"]))
        for key in model_keys:
            pred_rows.append(prediction_row(case, key, Wc, ev_idx, cand, targets[key], wfs[key].proba, wfs[key].base_rate))

        plot_case(case, curves, ["buy_hold", "base", "vol_filter"] + model_keys)
        print(f"[{time.time() - t0:5.1f}s] {case} done")

    m = pd.DataFrame(metric_rows)
    base = m[m.strategy == "base"].set_index(["case", "cost_bps"])["cagr_pct"]
    m["delta_cagr_vs_base_pct"] = m["cagr_pct"] - [base.get((c, k), np.nan) for c, k in zip(m.case, m.cost_bps)]
    hbase = m[m.strategy == "base"].set_index(["case", "cost_bps"])["holding_cagr_pct"]
    m["delta_holding_cagr_vs_base_pct"] = m["holding_cagr_pct"] - [hbase.get((c, k), np.nan) for c, k in zip(m.case, m.cost_bps)]
    m.to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(pred_rows).to_csv(OUT / "prediction_metrics.csv", index=False)
    pd.DataFrame(count_rows).to_csv(OUT / "sample_counts.csv", index=False)
    pd.DataFrame(yearly_rows).to_csv(OUT / "yearly_returns_5bps.csv", index=False)
    pd.concat(curve_rows).to_csv(OUT / "equity_curves_5bps.csv", index=False)
    pd.DataFrame(coef_rows).to_csv(OUT / "model_explanations.csv", index=False)
    (OUT / "selected_hyperparameters.json").write_text(
        json.dumps({k: {"model": v[0].label, "features": v[1]} for k, v in chosen.items()}, indent=2)
    )
    write_summary(m, pd.DataFrame(pred_rows), selection, pd.DataFrame(count_rows), chosen)
    print(f"[{time.time() - t0:5.1f}s] all done -> {OUT}")


LABELS = {
    "buy_hold": "Buy & hold", "base": "Base rule", "vol_filter": "Vol filter", "logit": "Logit",
    "rf": "Random forest", "gb": "Gradient boosting", "logit_trend": "Logit (trend only)", "logit_risk": "Logit (risk only)",
}


def plot_case(case, curves, names):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    for n in names:
        cur = curves[(n, MAIN_COST)]
        d = pd.to_datetime(cur["date"])
        a1.plot(d, cur["equity"], label=LABELS.get(n, n), lw=1.2)
        a2.plot(d, cur["equity"] / cur["equity"].cummax().clip(lower=1) - 1, lw=0.9)
    a1.set_yscale("log")
    a1.set_ylabel("Growth of $1 (log)")
    a1.set_title(f"{case.replace('_', ' ')} — test 2019–2025, {MAIN_COST} bps per side")
    a1.legend(fontsize=8, ncol=3)
    a1.grid(alpha=0.3)
    a2.set_ylabel("Drawdown")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / f"equity_{case}.png", dpi=150)
    plt.close(fig)


def md_table(df: pd.DataFrame, floatfmt="{:.2f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def write_summary(m, pred, selection, counts, chosen):
    lines = ["# Results summary (auto-generated by scripts/run_experiments.py)", ""]
    lines += ["Test period 2019–2025, weekly decisions, next-open execution, cash earns 0.", ""]
    lines += ["## Selected hyperparameters (validation 2014–2018, lowest Brier)", ""]
    lines += [md_table(selection[["strategy", "model", "val_brier", "val_const_brier", "n_val", "selected"]], "{:.5f}"), ""]
    lines += ["## Candidate-week counts", "", md_table(counts, "{:.1f}"), ""]
    cols = ["strategy", "cagr_pct", "holding_cagr_pct", "delta_holding_cagr_vs_base_pct", "delta_cagr_vs_base_pct", "max_dd_pct", "sharpe", "ann_vol_pct", "exposure_pct", "trades_per_year"]
    for case, *_ in CASES:
        sub = m[(m.case == case) & (m.cost_bps == MAIN_COST)][cols]
        lines += [f"## {case} — {MAIN_COST} bps", "", md_table(sub), ""]
        piv = m[(m.case == case) & ~m.strategy.str.startswith("exposure")].pivot(index="strategy", columns="cost_bps", values="delta_cagr_vs_base_pct")
        piv.columns = [f"dCAGR@{c}bps" for c in piv.columns]
        lines += ["Cost sensitivity (ΔCAGR vs base, pp):", "", md_table(piv.reset_index()), ""]
    lines += ["## Prediction and filtering diagnostics (test)", "", md_table(pred, "{:.3f}"), ""]
    (OUT / "summary.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
