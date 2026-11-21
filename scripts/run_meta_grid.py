"""Meta-labeling across a dual-SMA grid: does ML add more when the base rule is weaker? (exploratory)

For every pair short SMA 1..51 step 5 (SMA1 = close) x long SMA 20..220 step 10 (short < long):
build the stock trades (src/ml_filter/meta.py), predict each trade at entry with a rolling annual
refit (label `win`: whole-trade return > 0; also `cap26`), keep trades with p >= training win rate
(`<model>`) or exactly the rejected ones (`<model>_inv`), plus a no-model `hivol` rule (entry vol20 in the
top half of that week's candidates), and compare against
  S0       all trades of that pair
  placebo  each live trade kept at random with the same keep rate (50 seeds)
Fixed specs (no grid search, to keep 220+ pairs cheap): logit(C=0.1) and gb(n_estimators=50, max_depth=1).
Weekly timing, capped equal weights min(1/n, 10%), 5 bps. Periods 2005-2025 and 2019-2025.
Usage: .venv/bin/python scripts/run_meta_grid.py
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
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ml_filter.data import load_daily, weekly_table  # noqa: E402
from ml_filter.meta import Holdings, build_trades, label_known, ledger_stats, rolling_predict, sma_windows  # noqa: E402
from ml_filter.models import ModelSpec  # noqa: E402
from ml_filter.universe import build_panel  # noqa: E402

FAST = np.arange(1, 52, 5)
SLOW = np.arange(20, 221, 10)
PRED_YEARS = list(range(2003, 2026))
PERIODS = {"2005-2025": (2005, 2025), "2019-2025": (2019, 2025)}
MODELS = {"logit": ModelSpec("logit", (("C", 0.1),)), "gb": ModelSpec("gb", (("n_estimators", 50), ("max_depth", 1)))}
LABELS = ["win", "cap26"]
COST = 5
N_RANDOM = 50
OUT = ROOT / "results" / "meta_label" / "grid"
_G: dict = {}


def _globals():
    if not _G:
        P = build_panel()
        _G["P"] = P
        _G["S"] = sma_windows(P, np.r_[FAST, SLOW])
        _G["qw"] = weekly_table(load_daily("QQQ")).set_index("d")
        _G["fd"] = P["weeks"].groupby("year")["d"].min().to_dict()
    return _G


def one_pair(f: int, s: int) -> list[dict]:
    g = _globals()
    P, S, fd = g["P"], g["S"], g["fd"]
    weeks, Of = P["weeks"], P["Of"]
    nW, nJ = P["R"].shape
    T, _ = build_trades(P, S[f], S[s], g["qw"])
    H = Holdings(T, (nW, nJ))
    s0 = H.weights(np.ones(len(T), bool))
    rng = np.random.default_rng(1000 * f + s)
    preds = {(lab, mk): rolling_predict(T, lab, spec, fd, PRED_YEARS) for lab in LABELS for mk, spec in MODELS.items()}
    out = []
    for pname, (y0, y1) in PERIODS.items():
        rows = np.flatnonzero(((weeks.year >= y0) & (weeks.year <= y1) & (weeks.e_next <= pd.Timestamp(f"{y1}-12-31"))).to_numpy())
        live = ((T.k1 >= rows[0]) & (T.k0 <= rows[-1])).to_numpy()
        base = ledger_stats(s0, rows, weeks, Of, COST)
        ent_all = T.entry_year.between(y0, y1) & T.completed
        common = {"fast": f, "slow": s, "period": pname, "trades": int(ent_all.sum()),
                  "win_rate": float((T.ret[ent_all] > 0).mean()), "mean_trade_ret_pct": float(T.ret[ent_all].mean() * 100),
                  "median_weeks": float(T.weeks_held[ent_all].median()),
                  "S0_cagr": base["cagr_pct"], "S0_sharpe": base["sharpe_weekly"], "S0_maxdd": base["max_dd_pct"], "S0_exposure": base["exposure_pct"]}
        rules = [("-", "hivol", (T.vol20_rk >= 0.5).to_numpy() & live, None)]
        for (lab, mk), pr in preds.items():
            rules += [(lab, mk, (pr.p >= pr.base).to_numpy() & live, pr), (lab, mk + "_inv", (pr.p < pr.base).to_numpy() & live, pr)]
        for lab, mk, keep, pr in rules:
            q = keep.sum() / max(live.sum(), 1)
            m = ledger_stats(H.weights(keep), rows, weeks, Of, COST)
            rnd = np.array([ledger_stats(H.weights((rng.random(len(T)) < q) & live), rows, weeks, Of, COST)["cagr_pct"] for _ in range(N_RANDOM)])
            if pr is None:
                ent = ent_all.to_numpy()
                auc = np.nan
            else:
                ent = (T.entry_year.between(y0, y1) & label_known(T, lab) & pr.p.notna()).to_numpy()
                yy, pp = pr.y[ent].to_numpy(), pr.p[ent].to_numpy()
                auc = roc_auc_score(yy, pp) if 0 < yy.mean() < 1 else np.nan
            ret, k = T.ret[ent].to_numpy(), keep[ent]
            out.append({**common, "label": lab, "model": mk, "keep_pct": float(k.mean() * 100), "auc": auc,
                        "ret_kept_minus_dropped_pct": float((ret[k].mean() - ret[~k].mean()) * 100) if 0 < k.mean() < 1 else np.nan,
                        "ml_cagr": m["cagr_pct"], "ml_sharpe": m["sharpe_weekly"], "ml_maxdd": m["max_dd_pct"], "ml_exposure": m["exposure_pct"],
                        "placebo_median_cagr": float(np.median(rnd)), "placebo_pctile": float((rnd < m["cagr_pct"]).mean() * 100)})
    return out


def heat(ax, df, val, title, cmap="RdBu", center=True):
    piv = df.pivot(index="slow", columns="fast", values=val).reindex(index=SLOW, columns=FAST)
    a = piv.to_numpy()
    lim = np.nanmax(np.abs(a)) if center else None
    im = ax.imshow(a, origin="lower", aspect="auto", cmap=cmap, vmin=-lim if center else None, vmax=lim if center else None)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if np.isfinite(a[i, j]):
                ax.text(j, i, f"{a[i, j]:.0f}" if abs(a[i, j]) >= 10 else f"{a[i, j]:.1f}", ha="center", va="center", fontsize=5)
    ax.set_xticks(range(len(FAST)), FAST); ax.set_yticks(range(len(SLOW)), SLOW)
    ax.set_xlabel("short SMA"); ax.set_ylabel("long SMA"); ax.set_title(title, fontsize=9)
    plt.colorbar(im, ax=ax, shrink=0.8)


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    _globals()  # build the SMA cache once before forking workers
    _G.clear()
    pairs = [(int(f), int(s)) for f in FAST for s in SLOW if f < s]
    res = Parallel(n_jobs=10, verbose=5)(delayed(one_pair)(f, s) for f, s in pairs)
    df = pd.DataFrame([r for rr in res for r in rr])
    df["ml_minus_S0"] = df.ml_cagr - df.S0_cagr
    df["ml_minus_placebo"] = df.ml_cagr - df.placebo_median_cagr
    df.to_csv(OUT / "meta_grid.csv", index=False)
    print(f"[{time.time() - t0:5.1f}s] {len(pairs)} pairs done")

    lines = []
    for pname in PERIODS:
        d0 = df[(df.period == pname) & (df.label == "win") & (df.model == "logit")]
        fig, axes = plt.subplots(1, 1, figsize=(8, 7))
        heat(axes, d0, "S0_cagr", f"S0 (all trades) CAGR %, {pname}, {COST} bps", cmap="RdYlGn", center=False)
        fig.tight_layout(); fig.savefig(OUT / f"S0_cagr_{pname}.png", dpi=150); plt.close(fig)
        for lab in LABELS:
            fig, axes = plt.subplots(2, 3, figsize=(22, 13))
            for r, mk in enumerate(MODELS):
                d = df[(df.period == pname) & (df.label == lab) & (df.model == mk)]
                heat(axes[r, 0], d, "ml_minus_S0", f"{lab}/{mk}: ML − S0 CAGR (pp)")
                heat(axes[r, 1], d, "ml_minus_placebo", f"{lab}/{mk}: ML − random-keep placebo CAGR (pp)")
                heat(axes[r, 2], d.assign(a=(d.auc - 0.5) * 100), "a", f"{lab}/{mk}: (AUC − 0.5) × 100")
            fig.suptitle(f"Meta-labeling across SMA pairs — {pname}, {COST} bps")
            fig.tight_layout(); fig.savefig(OUT / f"heat_{lab}_{pname}.png", dpi=130); plt.close(fig)
        for (lab, mk), d in df[df.period == pname].groupby(["label", "model"]):
            weak = d.S0_cagr <= d.S0_cagr.median()
            lines.append(
                f"{pname} {lab:5s} {mk:5s}: ML>S0 in {(d.ml_minus_S0 > 0).mean() * 100:.0f}% of pairs, ML>placebo median in "
                f"{(d.ml_minus_placebo > 0).mean() * 100:.0f}%; mean ML−placebo {d.ml_minus_placebo.mean():+.2f} pp "
                f"(weak half of S0 {d.ml_minus_placebo[weak].mean():+.2f}, strong half {d.ml_minus_placebo[~weak].mean():+.2f}); "
                f"median AUC {d.auc.median():.3f}; corr(S0 CAGR, ML−placebo) {d.S0_cagr.corr(d.ml_minus_placebo):+.2f}")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
