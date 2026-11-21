"""Robustness extensions beyond the prespecified 2019-2025 test (added after the main run).

A. Purged blocked 5-fold CV over 2000-2025. Each block is the test set once; the
   hyperparameters are chosen by nested CV on the other four blocks (3 train /
   1 validation, rotating, lowest pooled Brier); the final model trains on all
   four. Training weeks whose information window [d - 65 sessions, e_next]
   overlaps a held-out block are purged/embargoed. Training on later years is
   allowed here, so this measures predictive ability across regimes, not a
   tradable history.

B. Anchored walk-forward 2005-2025. Every year is predicted by a model trained
   only on weeks whose labels were complete before that year's first decision;
   the hyperparameters for year Y are those with the lowest Brier over the
   previous five years of out-of-sample predictions (2003 onward). Fully causal.

Hyperparameters are selected on the main case (QQQ SMA20/60) and transferred to
the other cases, as in the proposal. Cost is 5 bps per side throughout.
Usage: .venv/bin/python scripts/run_robustness.py
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

from ml_filter.backtest import equity_curve, metrics, yearly_returns  # noqa: E402
from ml_filter.data import FEATURES, RISK_FEATURES, TREND_FEATURES, load_daily, weekly_table  # noqa: E402
from ml_filter.models import GB_GRID, LOGIT_GRID, RF_GRID, THRESHOLD, VOL_QUANTILE, brier, fit_predict, walk_forward  # noqa: E402
from run_experiments import CASES, LABELS, MAIN, MAIN_RULE, MAIN_SYM, candidates, md_table  # noqa: E402

COST = 5
LOOKBACK = 65  # longest feature look-back in sessions (SMA60 slope over 5 sessions)
BLOCKS = [(2000, 2004), (2005, 2009), (2010, 2014), (2015, 2019), (2020, 2025)]
WF_PRED_YEARS = list(range(2003, 2026))
WF_TEST_YEARS = list(range(2005, 2026))
WF_LOOKBACK_YEARS = 5
OUT = ROOT / "results" / "robustness" if MAIN == "QQQ_SMA20_60" else ROOT / "results" / "cases" / MAIN / "robustness"
GROUPS = {  # key -> (features, grid, main-case only)
    "logit": (FEATURES, LOGIT_GRID, False),
    "rf": (FEATURES, RF_GRID, False),
    "gb": (FEATURES, GB_GRID, True),
    "logit_trend": (TREND_FEATURES, LOGIT_GRID, True),
    "logit_risk": (RISK_FEATURES, LOGIT_GRID, True),
}


def span_mask(W: pd.DataFrame, a: int, b: int) -> np.ndarray:
    return (W["year"].between(a, b) & (W["e_next"] <= pd.Timestamp(f"{b}-12-31"))).to_numpy()


def clean_of(W: pd.DataFrame, held: np.ndarray) -> np.ndarray:
    """Weeks whose information window does not touch the held-out sessions."""
    lo, hi = W.loc[held, "i_e"].min(), W.loc[held, "i_e_next"].max()
    return ~(((W["i_d"] - LOOKBACK) <= hi) & (W["i_e_next"] >= lo)).to_numpy()


def strategy_targets(W, cand, test, train, probas):
    q = np.quantile(W.loc[train, "vol20"].to_numpy(), VOL_QUANTILE)
    t = {
        "buy_hold": np.ones(len(W)),
        "base": cand.astype(float),
        "vol_filter": (cand & (W["vol20"] <= q).to_numpy()).astype(float),
    }
    for k, p in probas.items():
        full = np.zeros(len(W))
        full[test & cand] = p >= THRESHOLD
        t[k] = full
    return t


def pred_stats(W, cand, test, train, p):
    y = W.loc[test & cand, "y"].to_numpy()
    c = np.full_like(p, W.loc[train, "y"].mean())
    return {
        "n": len(y), "brier": brier(p, y), "brier_const": brier(c, y),
        "brier_skill_pct": (1 - brier(p, y) / brier(c, y)) * 100,
        "auc": roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan,
        "retained_pct": (p >= THRESHOLD).mean() * 100,
    }


# ---------------------------------------------------------------- A. blocked CV
def run_cv(weeks, daily):
    rows, preds, sel_rows, chained = [], [], [], {}
    Wm = weeks[MAIN_SYM]
    cand_m = candidates(Wm, MAIN_RULE)
    base_ok = (Wm["year"] >= 2000).to_numpy()
    for (a, b) in BLOCKS:
        test_m = span_mask(Wm, a, b)
        outer_ok = base_ok & ~test_m & clean_of(Wm, test_m)
        chosen = {}
        for key, (feats, grid, _) in GROUPS.items():
            best, best_s = None, np.inf
            for spec in grid:
                ps, ys = [], []
                for (va, vb) in BLOCKS:
                    if (va, vb) == (a, b):
                        continue
                    val = span_mask(Wm, va, vb)
                    tr = outer_ok & ~val & clean_of(Wm, val) & cand_m
                    ps.append(fit_predict(Wm, tr, val & cand_m, feats, spec))
                    ys.append(Wm.loc[val & cand_m, "y"].to_numpy())
                s = brier(np.concatenate(ps), np.concatenate(ys))
                sel_rows.append({"test_block": f"{a}-{b}", "strategy": key, "model": spec.label, "inner_val_brier": s})
                if s < best_s - 1e-12:
                    best, best_s = spec, s
            chosen[key] = best
        for r in sel_rows:
            if r["test_block"] == f"{a}-{b}":
                r["selected"] = r["model"] == chosen[r["strategy"]].label

        for case, sym, rule in CASES:
            W, D = weeks[sym], daily[sym]
            cand = candidates(W, rule)
            test = span_mask(W, a, b)
            train = (W["year"] >= 2000).to_numpy() & ~test & clean_of(W, test) & cand
            probas = {}
            for key, (feats, _, main_only) in GROUPS.items():
                if main_only and case != MAIN:
                    continue
                probas[key] = fit_predict(W, train, test & cand, feats, chosen[key])
                preds.append({"test_block": f"{a}-{b}", "case": case, "strategy": key, "model": chosen[key].label,
                              **pred_stats(W, cand, test, train, probas[key])})
            targets = strategy_targets(W, cand, test, train, probas)
            ev = W[test]
            for name, w in targets.items():
                cur = equity_curve(D, ev, w[test], COST)
                rows.append({"test_block": f"{a}-{b}", "case": case, "strategy": name, **metrics(cur)})
                chained.setdefault((case, name), []).append(cur)

    for (case, name), parts in chained.items():
        scale, segs = 1.0, []
        for cur in parts:
            segs.append(cur.assign(equity=cur["equity"] * scale))
            scale *= cur["equity"].iloc[-1]
        full = pd.concat(segs, ignore_index=True)
        rows.append({"test_block": "2000-2025 chained", "case": case, "strategy": name, **metrics(full)})
    return pd.DataFrame(rows), pd.DataFrame(preds), pd.DataFrame(sel_rows)


# ---------------------------------------------------------- B. walk-forward
def run_walk_forward(weeks, daily):
    cache = {}

    def wf(case, sym, rule, key, spec):
        k = (case, key, spec.label)
        if k not in cache:
            W = weeks[sym]
            cache[k] = walk_forward(W, candidates(W, rule), WF_PRED_YEARS, GROUPS[key][0], spec).proba
        return cache[k]

    # rolling selection on the main case
    Wm = weeks[MAIN_SYM]
    sel_rows, choice = [], {}
    for key, (feats, grid, _) in GROUPS.items():
        scores = {}
        for spec in grid:
            p = wf(MAIN, MAIN_SYM, MAIN_RULE, key, spec)
            m = p.notna()
            for yr in WF_PRED_YEARS:
                mm = m & (Wm["year"] == yr)
                scores[(spec.label, yr)] = (((p[mm] - Wm.loc[mm, "y"]) ** 2).sum(), int(mm.sum()))
        for yr in WF_TEST_YEARS:
            window = range(max(WF_PRED_YEARS[0], yr - WF_LOOKBACK_YEARS), yr)
            best, best_s = None, np.inf
            for spec in grid:
                se = sum(scores[(spec.label, y)][0] for y in window)
                n = sum(scores[(spec.label, y)][1] for y in window)
                s = se / n
                if s < best_s - 1e-12:
                    best, best_s = spec, s
            choice[(key, yr)] = best
            sel_rows.append({"strategy": key, "year": yr, "model": best.label, "trailing_brier": best_s})

    rows, yearly, preds, curves_main = [], [], [], {}
    for case, sym, rule in CASES:
        W, D = weeks[sym], daily[sym]
        cand = candidates(W, rule)
        test = span_mask(W, WF_TEST_YEARS[0], WF_TEST_YEARS[-1])
        base_wf = walk_forward(W, cand, WF_TEST_YEARS, FEATURES, None)
        targets = {
            "buy_hold": np.ones(len(W)),
            "base": cand.astype(float),
            "vol_filter": (cand & (W["vol20"] <= base_wf.vol_q).to_numpy()).astype(float),
        }
        for key, (feats, grid, main_only) in GROUPS.items():
            if main_only and case != MAIN:
                continue
            proba = pd.Series(np.nan, index=W.index)
            for yr in WF_TEST_YEARS:
                p = wf(case, sym, rule, key, choice[(key, yr)])
                m = W["year"] == yr
                proba[m] = p[m]
            targets[key] = (cand & (proba >= THRESHOLD).to_numpy()).astype(float)
            mm = (test & cand) & proba.notna().to_numpy()
            y = W.loc[mm, "y"].to_numpy()
            pp = proba[mm].to_numpy()
            const = base_wf.base_rate[mm].to_numpy()
            preds.append({"case": case, "strategy": key, "n": len(y), "brier": brier(pp, y), "brier_const": brier(const, y),
                          "brier_skill_pct": (1 - brier(pp, y) / brier(const, y)) * 100, "auc": roc_auc_score(y, pp),
                          "retained_pct": (pp >= THRESHOLD).mean() * 100})
        ev = W[test]
        for name, w in targets.items():
            cur = equity_curve(D, ev, w[test], COST)
            rows.append({"case": case, "strategy": name, **metrics(cur)})
            for yr, r in yearly_returns(cur).items():
                yearly.append({"case": case, "strategy": name, "year": yr, "return_pct": r})
            if case == MAIN:
                curves_main[name] = cur
    return pd.DataFrame(rows), pd.DataFrame(yearly), pd.DataFrame(preds), pd.DataFrame(sel_rows), curves_main


def plot_cv(cv):
    sub = cv[(cv.case == MAIN) & (cv.test_block != "2000-2025 chained")]
    names = ["buy_hold", "base", "vol_filter", "logit", "rf", "gb"]
    blocks = sub["test_block"].unique()
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, col, title in [(axes[0], "cagr_pct", "CAGR %"), (axes[1], "holding_cagr_pct", "Holding-period CAGR %")]:
        x = np.arange(len(blocks))
        width = 0.13
        for j, n in enumerate(names):
            vals = [sub[(sub.test_block == bl) & (sub.strategy == n)][col].iloc[0] for bl in blocks]
            ax.bar(x + (j - 2.5) * width, vals, width, label=LABELS[n])
        ax.set_xticks(x, blocks)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(f"{MAIN} — blocked CV test blocks, {COST} bps — {title}")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / f"cv_blocks_{MAIN}.png", dpi=150)
    plt.close(fig)


def plot_wf(curves):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    for n in ["buy_hold", "base", "vol_filter", "logit", "rf", "gb"]:
        cur = curves[n]
        d = pd.to_datetime(cur["date"])
        a1.plot(d, cur["equity"], lw=1.1, label=LABELS[n])
        a2.plot(d, cur["equity"] / cur["equity"].cummax().clip(lower=1) - 1, lw=0.8)
    a1.set_yscale("log")
    a1.set_title(f"{MAIN} — anchored walk-forward 2005–2025 (causal), {COST} bps")
    a1.set_ylabel("Growth of $1 (log)")
    a1.legend(fontsize=8, ncol=3)
    a1.grid(alpha=0.3)
    a2.set_ylabel("Drawdown")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / f"wf_equity_{MAIN}.png", dpi=150)
    plt.close(fig)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    daily = {s: load_daily(s) for s in ("QQQ", "SPY")}
    weeks = {s: weekly_table(d) for s, d in daily.items()}

    cv, cv_pred, cv_sel = run_cv(weeks, daily)
    print(f"[{time.time() - t0:5.1f}s] blocked CV done")
    wf_m, wf_y, wf_pred, wf_sel, curves = run_walk_forward(weeks, daily)
    print(f"[{time.time() - t0:5.1f}s] walk-forward done")

    def add_deltas(df, keys):
        ref = df[df.strategy == "base"][keys + ["cagr_pct", "holding_cagr_pct"]].rename(
            columns={"cagr_pct": "_b_cagr", "holding_cagr_pct": "_b_hold"})
        df = df.merge(ref, on=keys, how="left")
        df["delta_cagr_vs_base"] = df["cagr_pct"] - df.pop("_b_cagr")
        df["delta_holding_cagr_vs_base"] = df["holding_cagr_pct"] - df.pop("_b_hold")
        return df

    cv = add_deltas(cv, ["test_block", "case"])
    wf_m = add_deltas(wf_m, ["case"])
    cv.to_csv(OUT / "cv_metrics.csv", index=False)
    cv_pred.to_csv(OUT / "cv_predictions.csv", index=False)
    cv_sel.to_csv(OUT / "cv_selection.csv", index=False)
    wf_m.to_csv(OUT / "wf_metrics.csv", index=False)
    wf_y.to_csv(OUT / "wf_yearly_returns.csv", index=False)
    wf_pred.to_csv(OUT / "wf_predictions.csv", index=False)
    wf_sel.to_csv(OUT / "wf_selection.csv", index=False)
    plot_cv(cv)
    plot_wf(curves)

    cols = ["strategy", "cagr_pct", "holding_cagr_pct", "delta_holding_cagr_vs_base", "delta_cagr_vs_base", "max_dd_pct", "sharpe", "exposure_pct", "trades_per_year"]
    L = ["# Robustness results (auto-generated by scripts/run_robustness.py)", "", __doc__.split("Usage")[0], ""]
    L += [f"## A. Blocked CV — {MAIN} by test block", ""]
    for bl in cv["test_block"].unique():
        L += [f"### {bl}", "", md_table(cv[(cv.case == MAIN) & (cv.test_block == bl)][cols]), ""]
    L += ["### Chained 2000–2025, all cases (Δ holding CAGR vs base, pp)", ""]
    ch = cv[cv.test_block == "2000-2025 chained"].pivot(index="strategy", columns="case", values="delta_holding_cagr_vs_base")
    L += [md_table(ch.reset_index()), ""]
    L += ["### Prediction quality by block", "", md_table(cv_pred, "{:.3f}"), ""]
    L += ["## B. Walk-forward 2005–2025", ""]
    for case, *_ in CASES:
        L += [f"### {case}", "", md_table(wf_m[wf_m.case == case][cols]), ""]
    L += ["### Prediction quality", "", md_table(wf_pred, "{:.3f}"), ""]
    yr = wf_y[wf_y.case == MAIN].pivot(index="year", columns="strategy", values="return_pct")
    L += [f"### {MAIN} calendar-year returns (%)", "", md_table(yr.reset_index(), "{:.1f}"), ""]
    (OUT / "summary_robustness.md").write_text("\n".join(L))
    print(f"[{time.time() - t0:5.1f}s] written -> {OUT}")


if __name__ == "__main__":
    main()
