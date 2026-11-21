"""Stock extension: dual-SMA candidates vs dual-SMA + ranking filters.

Usage: .venv/bin/python scripts/run_stocks.py [--sma 20_60] [--k 20]
Outputs: results/stocks/sma<F>_<S>/
"""

from __future__ import annotations

import argparse
import json
import sys
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

from ml_filter.data import FEATURES  # noqa: E402
from ml_filter.meta import sma_windows  # noqa: E402
from ml_filter.models import GB_GRID, LOGIT_GRID, RF_GRID, ModelSpec  # noqa: E402
from ml_filter.portfolio import capped_equal, daily_curve, portfolio_metrics, weekly_ledger, weekly_metrics  # noqa: E402
from ml_filter.universe import build_panel, load_qqq  # noqa: E402
from run_experiments import md_table  # noqa: E402

TRAIN_START = 2000
PRED_YEARS = list(range(2003, 2026))
EVAL_YEARS = (2005, 2025)
SLICE_YEARS = (2019, 2025)
LOOKBACK_YEARS = 5
COSTS = [0, 5, 10]
MAIN_COST = 5
N_RANDOM = 200
BUFFER_RANK = 30
RANK_FEATURES = [f + "_rk" for f in FEATURES]
X_COLS = FEATURES + RANK_FEATURES
GRIDS = {"logit": LOGIT_GRID, "rf": RF_GRID, "gb": GB_GRID}


def nw_t(x: np.ndarray, lags: int = 4) -> float:
    x = np.asarray(x, float)
    n, mu = len(x), x.mean()
    e = x - mu
    s = (e @ e) / n
    for l in range(1, lags + 1):
        s += 2 * (1 - l / (lags + 1)) * (e[l:] @ e[:-l]) / n
    return mu / np.sqrt(s / n)


def top_k(score: np.ndarray, k: int, buffer: int | None = None) -> np.ndarray:
    """score: (weeks x assets), NaN = not a candidate, higher is better. Ties -> lower asset index."""
    s = np.where(np.isnan(score), -np.inf, score)
    order = np.argsort(-s, axis=1, kind="stable")
    rank = np.empty_like(order)
    rows = np.arange(s.shape[0])[:, None]
    rank[rows, order] = np.arange(s.shape[1])[None, :]
    valid = ~np.isnan(score)
    if buffer is None:
        return valid & (rank < k)
    sel = np.zeros_like(valid)
    prev = np.zeros(s.shape[1], bool)
    for t in range(s.shape[0]):
        keep = prev & valid[t] & (rank[t] < buffer)
        need = k - keep.sum()
        fill = [j for j in order[t] if valid[t, j] and not keep[j]][: max(need, 0)]
        cur = keep.copy()
        cur[fill] = True
        sel[t], prev = cur, cur
    return sel


def fit_task(df, label, spec: ModelSpec, first_decision):
    out = pd.Series(np.nan, index=df.index)
    for yr in PRED_YEARS:
        tr = (df.year >= TRAIN_START) & (df.e_next <= first_decision[yr])
        te = df.year == yr
        model = spec.build()
        if spec.family == "rf":
            model.set_params(n_jobs=1)
        model.fit(df.loc[tr, X_COLS].to_numpy(), df.loc[tr, label].to_numpy())
        out[te] = model.predict_proba(df.loc[te, X_COLS].to_numpy())[:, 1]
    return label, spec.label, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sma", default="20_60")
    ap.add_argument("--k", type=int, default=20)
    a = ap.parse_args()
    fast, slow = (int(x) for x in a.sma.split("_"))
    K = a.k
    out = ROOT / "results" / "stocks" / f"sma{fast}_{slow}"
    (out / "figures").mkdir(parents=True, exist_ok=True)

    P = build_panel()
    weeks, F, Of, Cf, cal = P["weeks"], P["feat"], P["Of"], P["Cf"], P["cal"]
    nW, nJ = P["R"].shape
    for w in (fast, slow):
        if f"sma{w}" not in F:  # windows outside the panel are computed (and cached) by ml_filter.meta.sma_windows
            F[f"sma{w}"] = sma_windows(P, [w])[w]
    complete = np.all([~np.isnan(F[c]) for c in FEATURES + [f"sma{slow}"]], axis=0)
    cand = P["tradable"] & complete & (F[f"sma{fast}"] > F[f"sma{slow}"])
    in_study = (weeks.year >= TRAIN_START) & (weeks.e_next <= pd.Timestamp("2025-12-31"))
    cand &= in_study.to_numpy()[:, None]

    # ---- long panel of candidate rows ----
    kk, jj = np.nonzero(cand)
    df = pd.DataFrame({"k": kk, "j": jj})
    for c in FEATURES + ["ret20"]:
        df[c] = F[c][kk, jj]
    df = df.loc[:, ~df.columns.duplicated()]
    df["R"] = P["R"][kk, jj]
    df["year"] = weeks.year.to_numpy()[kk]
    df["e_next"] = weeks.e_next.to_numpy()[kk]
    g = df.groupby("k")
    for f in FEATURES:
        df[f + "_rk"] = g[f].rank(pct=True)
    df["y_abs"] = (df.R > 0).astype(int)
    df["y_rel"] = (df.R > g["R"].transform("median")).astype(int)
    first_decision = weeks.groupby("year")["d"].min().to_dict()
    print(f"candidate rows {len(df):,}; mean candidates/week {df.groupby('k').size().mean():.1f}")

    # ---- walk-forward ML (pooled panel, annual expanding refit) ----
    tasks = [(lab, spec) for lab in ("y_rel", "y_abs") for grid in GRIDS.values() for spec in grid]
    res = Parallel(n_jobs=min(len(tasks), 12), verbose=0)(delayed(fit_task)(df, lab, spec, first_decision) for lab, spec in tasks)
    proba = {(lab, s): p for lab, s, p in res}
    sel_rows, scores = [], {}
    for lab in ("y_rel", "y_abs"):
        base_rate = {yr: df.loc[(df.year >= TRAIN_START) & (df.e_next <= first_decision[yr]), lab].mean() for yr in PRED_YEARS}
        for fam, grid in GRIDS.items():
            se = {(s.label, yr): float(((proba[(lab, s.label)] - df[lab]) ** 2)[df.year == yr].sum()) for s in grid for yr in PRED_YEARS}
            n = {yr: int((df.year == yr).sum()) for yr in PRED_YEARS}
            chosen = pd.Series(np.nan, index=df.index)
            for yr in range(EVAL_YEARS[0], EVAL_YEARS[1] + 1):
                win = range(max(PRED_YEARS[0], yr - LOOKBACK_YEARS), yr)
                best = min(grid, key=lambda s: (sum(se[(s.label, y)] for y in win) / sum(n[y] for y in win), grid.index(s)))
                m = df.year == yr
                chosen[m] = proba[(lab, best.label)][m]
                sel_rows.append({"label": lab, "family": fam, "year": yr, "model": best.label})
            scores[f"{'M1' if lab == 'y_rel' else 'M2'}_{fam}"] = (lab, chosen)
        df[f"base_{lab}"] = df.year.map(base_rate)
    pd.DataFrame(sel_rows).to_csv(out / "wf_selection.csv", index=False)

    def to_matrix(v: pd.Series) -> np.ndarray:
        M = np.full((nW, nJ), np.nan)
        M[df.k, df.j] = v.to_numpy()
        return M

    score_mats = {
        "F1_momentum": to_matrix(df["ret20"]),
        "F2_lowvol": to_matrix(-df["vol20"]),
        **{name: to_matrix(v) for name, (lab, v) in scores.items()},
    }

    # ---- selections ----
    member_tr = P["tradable"] & in_study.to_numpy()[:, None]
    sels = {"EW_members": member_tr, "S0_sma_all": cand}
    for name, M in score_mats.items():
        sels[name] = top_k(M, K)
    rng = np.random.default_rng(628)
    rand_sels = [top_k(np.where(cand, rng.random(cand.shape), np.nan), K) for _ in range(N_RANDOM)]
    qo, qc = load_qqq(cal)

    def period_rows(y0, y1):
        m = (weeks.year >= y0) & (weeks.year <= y1) & (weeks.e_next <= pd.Timestamp(f"{y1}-12-31"))
        return np.flatnonzero(m.to_numpy())

    def run(sel, rows, cost, o=Of, c=Cf, weights=None, curve=True):
        wk = weeks.iloc[rows].reset_index(drop=True)
        W = capped_equal(sel[rows]) if weights is None else weights[rows]
        G = o[wk.i_e_next] / o[wk.i_e]
        led = weekly_ledger(W, G, cost)
        if not curve:
            return weekly_metrics(led, wk)
        cv = daily_curve(W, wk, o, c, led, cal)
        return portfolio_metrics(cv, W, led, wk), cv

    metric_rows, curves, yearly = [], {}, {}
    for pname, (y0, y1) in {"2005-2025": EVAL_YEARS, "2019-2025": SLICE_YEARS}.items():
        rows = period_rows(y0, y1)
        for cost in COSTS:
            m, cv = run(None, rows, cost, qo, qc, weights=np.ones((nW, 1)))
            metric_rows.append({"period": pname, "cost_bps": cost, "strategy": "QQQ_buy_hold", **m})
            if cost == MAIN_COST:
                curves[(pname, "QQQ_buy_hold")] = cv
            for name, sel in sels.items():
                m, cv = run(sel, rows, cost)
                metric_rows.append({"period": pname, "cost_bps": cost, "strategy": name, **m})
                if cost == MAIN_COST:
                    curves[(pname, name)] = cv
            rnd = pd.DataFrame([run(s, rows, cost, curve=False) for s in rand_sels])
            metric_rows.append({"period": pname, "cost_bps": cost, "strategy": f"R_random{K}_median",
                                "cagr_pct": rnd.cagr_pct.median(), "sharpe_weekly": rnd.sharpe_weekly.median()})
            rnd.to_csv(out / f"placebo_{pname}_{cost}bps.csv", index=False)
            for r in metric_rows:
                if r["period"] == pname and r["cost_bps"] == cost and (r["strategy"] in sels or r["strategy"] == "S0_sma_all") and r["strategy"] != "EW_members":
                    r["placebo_pctile_cagr"] = float((rnd.cagr_pct < r["cagr_pct"]).mean() * 100)
                    r["placebo_pctile_sharpe"] = float((rnd.sharpe_weekly < r["sharpe_weekly"]).mean() * 100)
    met = pd.DataFrame(metric_rows)
    s0 = met[met.strategy == "S0_sma_all"][["period", "cost_bps", "cagr_pct", "holding_cagr_pct"]].rename(columns={"cagr_pct": "_c", "holding_cagr_pct": "_h"})
    met = met.merge(s0, on=["period", "cost_bps"], how="left")
    met["delta_cagr_vs_S0"] = met.cagr_pct - met.pop("_c")
    met["delta_holding_cagr_vs_S0"] = met.holding_cagr_pct - met.pop("_h")
    met.to_csv(out / "metrics.csv", index=False)

    for name in ["QQQ_buy_hold", *sels]:
        cv = curves[("2005-2025", name)]
        s = cv.set_index(pd.to_datetime(cv.date))["equity"]
        ye = s.groupby(s.index.year).last()
        yearly[name] = (ye / ye.shift(1).fillna(1.0) - 1) * 100
    pd.DataFrame(yearly).round(2).to_csv(out / "yearly_returns_5bps.csv")

    # ---- selection skill: top-K vs rest of candidates (main period) ----
    rows = period_rows(*EVAL_YEARS)
    ev = df[df.k.isin(rows)].copy()
    ncand = ev.groupby("k").size()
    deep = ncand[ncand > K].index
    stat_rows, quint_rows = [], []
    for name, M in score_mats.items():
        ev["score"] = M[ev.k, ev.j]
        ev["sel"] = sels[name][ev.k, ev.j]
        d = ev[ev.k.isin(deep)]
        spread = d.groupby("k").apply(lambda x: x.R[x.sel].mean() - x.R.mean(), include_groups=False)  # top-K minus all candidates
        ic = d.groupby("k").apply(lambda x: x.score.corr(x.R, method="spearman"), include_groups=False).dropna()
        d = d.assign(q=d.groupby("k")["score"].rank(pct=True, method="first").mul(5).apply(np.ceil).clip(1, 5))
        qm = d.groupby(["k", "q"]).R.mean().unstack().mean() * 100
        quint_rows.append({"strategy": name, **{f"Q{int(q)}_pct": v for q, v in qm.items()}})
        stat_rows.append({"strategy": name, "weeks": len(spread), "spread_bps_per_week": spread.mean() * 1e4,
                          "spread_nw_t": nw_t(spread.to_numpy()), "spread_pos_pct": (spread > 0).mean() * 100,
                          "rank_ic_mean": ic.mean(), "rank_ic_t": ic.mean() / ic.std(ddof=1) * np.sqrt(len(ic))})
    rs = []
    for s in rand_sels[:50]:
        ev["sel"] = s[ev.k, ev.j]
        d = ev[ev.k.isin(deep)]
        rs.append(d.groupby("k").apply(lambda x: x.R[x.sel].mean() - x.R.mean(), include_groups=False).mean() * 1e4)
    stat_rows.append({"strategy": f"R_random{K} (50 seeds)", "weeks": len(deep), "spread_bps_per_week": float(np.mean(rs)),
                      "spread_nw_t": float(np.mean(rs) / np.std(rs, ddof=1)) if len(rs) > 1 else np.nan})
    stats = pd.DataFrame(stat_rows)
    stats.to_csv(out / "selection_stats.csv", index=False)
    quint = pd.DataFrame(quint_rows)
    quint.to_csv(out / "quintiles.csv", index=False)

    pred_rows = []
    for name, (lab, v) in scores.items():
        m = df.k.isin(rows)
        y, p, b = df.loc[m, lab], v[m], df.loc[m, f"base_{lab}"]
        pred_rows.append({"strategy": name, "label": lab, "n": int(m.sum()), "auc": roc_auc_score(y, p),
                          "brier": float(((p - y) ** 2).mean()), "const_brier": float(((b - y) ** 2).mean()),
                          "brier_skill": 1 - float(((p - y) ** 2).mean()) / float(((b - y) ** 2).mean())})
    pred = pd.DataFrame(pred_rows)
    pred.to_csv(out / "prediction_metrics.csv", index=False)

    # ---- sensitivity: K and buffer (main period, 5 bps) ----
    sens = []
    rows = period_rows(*EVAL_YEARS)
    for k2 in sorted({10, 20, 30} | {K}):
        rnd = pd.DataFrame([run(top_k(np.where(cand, np.random.default_rng(1000 + i).random(cand.shape), np.nan), k2), rows, MAIN_COST, curve=False)
                            for i in range(N_RANDOM)])
        sens.append({"K": k2, "buffer": "none", "strategy": "R_random_median", "cagr_pct": rnd.cagr_pct.median(), "sharpe_weekly": rnd.sharpe_weekly.median()})
        for name, M in score_mats.items():
            for buf in ([None, BUFFER_RANK] if k2 == K else [None]):
                m, _ = run(top_k(M, k2, buf), rows, MAIN_COST)
                sens.append({"K": k2, "buffer": buf or "none", "strategy": name, "cagr_pct": m["cagr_pct"],
                             "holding_cagr_pct": m["holding_cagr_pct"], "sharpe_weekly": m["sharpe_weekly"],
                             "max_dd_pct": m["max_dd_pct"], "annual_turnover": m["annual_turnover"],
                             "placebo_pctile_cagr": float((rnd.cagr_pct < m["cagr_pct"]).mean() * 100)})
    sens = pd.DataFrame(sens)
    sens.to_csv(out / "sensitivity_5bps.csv", index=False)

    # ---- figures ----
    fig, ax = plt.subplots(figsize=(11, 5.5))
    show = ["QQQ_buy_hold", "EW_members", "S0_sma_all", "F1_momentum", "F2_lowvol", "M1_logit", "M1_rf", "M1_gb"]
    for name in show:
        cv = curves[("2005-2025", name)]
        ax.plot(pd.to_datetime(cv.date), cv.equity, label=name, lw=1.6 if name.startswith(("S0", "QQQ")) else 1.0)
    ax.set_yscale("log"); ax.legend(ncol=4, fontsize=8); ax.grid(alpha=0.3)
    ax.set_title(f"Nasdaq-100 PIT stocks, SMA{fast}/{slow} candidates, top-{K}, 5 bps, 2005–2025")
    fig.tight_layout(); fig.savefig(out / "figures" / "equity_2005_2025.png", dpi=150); plt.close(fig)

    rnd = pd.read_csv(out / f"placebo_2005-2025_{MAIN_COST}bps.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(rnd.cagr_pct, bins=30, color="0.75", label=f"random {K} ({N_RANDOM} seeds)")
    mm = met[(met.period == "2005-2025") & (met.cost_bps == MAIN_COST)].set_index("strategy")
    for i, name in enumerate([n for n in sels if n not in ("EW_members",)]):
        ax.axvline(mm.loc[name, "cagr_pct"], color=f"C{i}", lw=1.5, label=name)
    ax.set_xlabel("CAGR % (2005–2025, 5 bps)"); ax.legend(fontsize=7); fig.tight_layout()
    fig.savefig(out / "figures" / "placebo_cagr.png", dpi=150); plt.close(fig)

    # ---- summary ----
    cols = ["strategy", "cagr_pct", "holding_cagr_pct", "sharpe", "max_dd_pct", "exposure_pct", "avg_names", "annual_turnover", "placebo_pctile_cagr"]
    lines = [f"# Stock extension — SMA{fast}/{slow}, top-{K}\n", f"Candidate rows {len(df):,}; mean candidates/week {df.groupby('k').size().mean():.1f}\n"]
    for pname in ("2005-2025", "2019-2025"):
        for cost in (MAIN_COST, 0):
            t = met[(met.period == pname) & (met.cost_bps == cost)][cols]
            lines += [f"\n## Portfolio, {pname}, {cost} bps\n", md_table(t)]
    lines += ["\n## Selection skill (2005–2025, weeks with > K candidates)\n", md_table(stats, '{:.3f}'),
              "\n## Quintile mean weekly return % (Q5 = highest score)\n", md_table(quint, '{:.3f}'),
              "\n## Prediction metrics\n", md_table(pred, '{:.4f}'),
              "\n## Sensitivity (5 bps, 2005–2025)\n", md_table(sens)]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
