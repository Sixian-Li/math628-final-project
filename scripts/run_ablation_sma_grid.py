"""SMA-pair surface of the three-component strategy, measured against the plain SMA rule on QQQ (exploratory, descriptive).

scripts/run_ablation.py fixes the base rule at SMA20/140. Here the whole pipeline is rebuilt for every pair SMA_fast > SMA_slow
with fast in 1..51 step 5 (SMA1 = close) and slow in 30..230 step 10 (pairs with fast >= slow are skipped):
  SMA (QQQ)                  QQQ held in the weeks with SMA_fast > SMA_slow. This is the baseline of every cell.
  SMA + Ridge (QQQ)          the same weeks, kept only when the Ridge forecast is > 0. The Ridge model is refitted on this
                             pair's uptrend weeks; its 16 market-state features do not change with the pair.
  SMA + Ridge + momentum k   the Ridge-kept weeks, holding the top k (10, 20) members by 12-month return among the members
                             whose own SMA_fast > SMA_slow, re-ranked every 4 weeks.
A heatmap cell is strategy minus `SMA (QQQ)` on the same pair and period: Sharpe, CAGR and holding CAGR (pp per year).
Periods, ledger, costs and metrics are those of scripts/run_ablation.py (weekly open-to-open, 5 bps per side, Sharpe from weekly
returns, holding CAGR over invested weeks only). The pair 20/140 is also run and checked against results/ablation/grid.csv.

Caveats: this is a surface over many pairs, not a parameter selection, and the 2019-2025 panel must not be used to pick a pair.
Holding CAGR is inflated when few weeks are invested, so read it with `invested` in grid.csv.
Usage: .venv/bin/python scripts/run_ablation_sma_grid.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import run_ablation as AB  # noqa: E402
import run_qqq_reverse as RV  # noqa: E402
import run_qqq_state as QS  # noqa: E402
from ml_filter.data import load_daily, weekly_table  # noqa: E402
from ml_filter.meta import sma_windows  # noqa: E402
from ml_filter.portfolio import capped_equal  # noqa: E402
from ml_filter.universe import build_panel  # noqa: E402
from ml_filter.xfeat import market_features, stock_features  # noqa: E402
from run_experiments import md_table  # noqa: E402
from run_stock_rank import select  # noqa: E402

FAST = list(range(1, 52, 5))
SLOW = list(range(30, 231, 10))
REF = (20, 140)  # the pair of run_ablation.py; run as a check, not part of the grid
MARK = (21, 140)  # grid cell closest to REF, outlined in the figures
PERIODS = {k: v for k, v in AB.PERIODS.items() if not k.startswith("rev_b")}
PLABEL = {"rev 2001-2008": "2001–2008 (one model fitted on 2009–2025)", "dev 2005-2018": "2005–2018 (walk-forward)",
          "holdout 2019-2025": "2019–2025 (walk-forward, holdout)", "full 2005-2025": "2005–2025 (walk-forward)"}
METRICS = {"sharpe": ("Sharpe", "Sharpe difference", "{:+.2f}"), "cagr": ("CAGR", "pp per year", "{:+.1f}"),
           "hold_cagr": ("Holding CAGR", "pp per year", "{:+.1f}")}

INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"
# diverging: red (worse) <- neutral gray -> blue (better), equal steps per arm
CMAP = LinearSegmentedColormap.from_list("red_gray_blue", ["#7a1616", "#b02a2a", "#e34948", "#ee8f88", "#f7c4bf", "#f0efec",
                                                           "#b7d3f6", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])


def heat(d, strat, m, path, sym):
    label, unit, fmt = METRICS[m]
    sub = d[d.strategy == strat]
    lim = float(np.nanpercentile(np.abs(sub["d_" + m]), 95))
    norm = Normalize(-lim, lim)
    fig, axes = plt.subplots(2, 2, figsize=(21, 11.5), facecolor=SURF, layout="constrained")
    for ax, pn in zip(axes.ravel(), PERIODS):
        M = sub[sub.period == pn].pivot(index="fast", columns="slow", values="d_" + m).reindex(index=FAST, columns=SLOW).to_numpy()
        im = ax.pcolormesh(np.ma.masked_invalid(M), cmap=CMAP, norm=norm, edgecolors=SURF, linewidth=1.6)
        ax.invert_yaxis()
        ax.set_facecolor(SURF)
        for i in range(len(FAST)):
            for j in range(len(SLOW)):
                if np.isfinite(M[i, j]):
                    r, g, b, _ = CMAP(norm(M[i, j]))
                    ax.text(j + 0.5, i + 0.5, fmt.format(M[i, j]), ha="center", va="center", fontsize=8.5,
                            color="white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.42 else INK)
        ax.add_patch(Rectangle((SLOW.index(MARK[1]), FAST.index(MARK[0])), 1, 1, fill=False, edgecolor=INK, linewidth=2.2))
        ax.set_xticks(np.arange(len(SLOW)) + 0.5, SLOW)
        ax.set_yticks(np.arange(len(FAST)) + 0.5, FAST)
        ax.tick_params(length=0, labelsize=10, colors=INK2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_xlabel("long SMA (sessions)", fontsize=10.5, color=INK2)
        ax.set_ylabel("short SMA (sessions; 1 = close)", fontsize=10.5, color=INK2)
        n, win = int(np.isfinite(M).sum()), int((M > 0).sum())
        ax.set_title(f"{PLABEL[pn]}", loc="left", fontsize=13, color=INK, fontweight="bold", pad=22)
        ax.text(0, 1.012, f"better than the plain rule in {win} of {n} pairs; median {fmt.format(np.nanmedian(M))}",
                transform=ax.transAxes, fontsize=11, color=INK2, va="bottom")
    cb = fig.colorbar(im, ax=axes, shrink=0.55, aspect=30, pad=0.012, extend="both")
    cb.set_label(f"{unit}   (blue = better than SMA on {sym}, red = worse)", fontsize=11, color=INK2)
    cb.ax.tick_params(labelsize=10, colors=INK2, length=0)
    cb.outline.set_visible(False)
    fig.suptitle(f"{label} gain over the plain SMA rule on {sym}, by SMA pair:  {strat}", x=0.012, ha="left", fontsize=17, color=INK, fontweight="bold")
    note = (f"Each cell = strategy minus SMA ({sym}) on the same short/long pair and period; weekly ledger, 5 bps per side. "
            f"Outlined cell: {MARK[0]}/{MARK[1]}, the grid point closest to the reference {REF[0]}/{REF[1]}. Colour is clipped at the 95th percentile of |gain|.")
    fig.text(0.012, -0.022, note, fontsize=10.5, color=INK2, va="top")
    fig.savefig(path, dpi=130, facecolor=SURF, bbox_inches="tight")
    plt.close(fig)


def main():
    sym = "QQQ"  # the asset that carries the SMA rule, the Ridge target and the baseline
    OUT = ROOT / "results" / "ablation" / "sma_grid"
    BASE, RIDGE = f"SMA ({sym})", f"SMA + Ridge ({sym})"
    STRATS = {"SMA + Ridge + momentum top 20": "top20", "SMA + Ridge + momentum top 10": "top10", RIDGE: sym.lower()}
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    P = build_panel()
    weeks, Of = P["weeks"], P["Of"]
    S = sma_windows(P, sorted({1, 50, 140, 200, *REF, *FAST, *SLOW}))
    ret252 = stock_features(P)["ret252"]
    D = load_daily(sym)
    W = weekly_table(D)
    # market-state features are common; the trend state, the target and the baseline belong to the traded asset
    X = market_features(P, S).set_index("d").reindex(W.d).reset_index()
    X["R1"], X["t1"], X["year"] = W.R.to_numpy(), W.e_next.to_numpy(), W.year.to_numpy()
    fd = W.groupby("year")["d"].min().to_dict()
    c = D.set_index("date")["close"]
    y2000 = (weeks.year >= 2000).to_numpy()[:, None]
    G = Of[weeks.i_e_next.to_numpy()] / Of[weeks.i_e.to_numpy()]
    Gq = (1 + W.R.to_numpy())[:, None]
    print(f"[{time.time() - t0:5.1f}s] data ready")

    def run_pair(f, s):
        ms = c.rolling(s).mean()
        up = (c.rolling(f).mean() > ms).astype(float).where(ms.notna()).reindex(W.d).to_numpy() == 1  # uptrend weeks of the traded asset
        pred = {"fwd": QS.walk(X, up, AB.FEATS, 1, AB.MODEL, fd)[0],
                "rev": RV.fit_apply(X, up, AB.FEATS, AB.MODEL, (2009, 2025), (2001, 2008))[0]}
        cand = P["tradable"] & ~np.isnan(S[s]) & (S[f] > S[s]) & y2000  # members in their own uptrend
        score = np.where(cand, ret252, np.nan)
        books = {k: select(score, cand, k=k, every=4) for k in (10, 20)}
        rows = []
        for pn, (sk, y0, y1) in PERIODS.items():
            rq, rs = AB.rows_of(W, y0, y1), AB.rows_of(weeks, y0, y1)
            keep = up & (pred[sk] > 0)
            tag = {"fast": f, "slow": s, "period": pn, "no_forecast_weeks": int((up & ~np.isfinite(pred[sk]))[rq].sum())}
            for name, sig in ((BASE, up), (RIDGE, keep)):
                rows.append({**tag, "strategy": name, **AB.stats(sig[rq][:, None].astype(float), Gq[rq], W.iloc[rq])})
            on = pd.Series(keep, index=X.d).reindex(weeks.d).fillna(False).to_numpy(bool)
            for k, sel in books.items():
                rows.append({**tag, "strategy": f"SMA + Ridge + momentum top {k}",
                             **AB.stats(capped_equal(sel[rs] & on[rs][:, None]), G[rs], weeks.iloc[rs])})
        return rows

    pairs = [(f, s) for f in FAST for s in SLOW if f < s]
    rows = []
    for n, (f, s) in enumerate([REF] + pairs):
        rows += run_pair(f, s)
        if n % 40 == 0:
            print(f"[{time.time() - t0:5.1f}s] {n + 1}/{len(pairs) + 1} pairs")
    df = pd.DataFrame(rows)
    ref = df[(df.fast == REF[0]) & (df.slow == REF[1])]
    df = df[df.fast.isin(FAST) & df.slow.isin(SLOW)]
    df.to_csv(OUT / "grid.csv", index=False)

    ref.to_csv(OUT / "reference_pair.csv", index=False)
    # the 20/140 pair must reproduce results/ablation/grid.csv
    ab = pd.read_csv(ROOT / "results" / "ablation" / "grid.csv").set_index(["universe", "book", "timing", "period"])
    amap = {BASE: ("-", "QQQ ETF", "sma"), RIDGE: ("-", "QQQ ETF", "sma+ridge"),
            "SMA + Ridge + momentum top 20": ("trend", "momentum top20", "sma+ridge"), "SMA + Ridge + momentum top 10": ("trend", "momentum top10", "sma+ridge")}
    gap = max(abs(r[m] - ab.loc[(*amap[r["strategy"]], r["period"]), m]) for r in ref.to_dict("records") for m in ("cagr", "hold_cagr", "sharpe"))
    check = f"Check: the pair {REF[0]}/{REF[1]} reproduces results/ablation/grid.csv with a maximum absolute difference of {gap:.1e}."
    print(check)

    key = ["fast", "slow", "period"]
    base = df[df.strategy == BASE].set_index(key)[[*METRICS, "invested"]]
    d = df[df.strategy != BASE].join(base, on=key, rsuffix="_base")
    for m in METRICS:
        d["d_" + m] = d[m] - d[m + "_base"]
    d.to_csv(OUT / "gain_vs_sma.csv", index=False)

    for strat, tag in STRATS.items():
        for m in METRICS:
            heat(d, strat, m, OUT / f"heat_{m}_{tag}.png", sym)

    summ = []
    for strat in STRATS:
        for pn in PERIODS:
            g = d[(d.strategy == strat) & (d.period == pn)]
            r = {"strategy": strat, "period": pn, "pairs": len(g), "invested % (median)": g.invested.median(), "base invested % (median)": g.invested_base.median()}
            for m, (label, _, _) in METRICS.items():
                v = g["d_" + m]
                r[f"{label}: better %"], r[f"{label}: median"], r[f"{label}: min"], r[f"{label}: max"] = (v > 0).mean() * 100, v.median(), v.min(), v.max()
            summ.append(r)
    summ = pd.DataFrame(summ)
    summ.to_csv(OUT / "summary.csv", index=False)
    skipped = df[(df.strategy == BASE) & (df.no_forecast_weeks > 0)][["fast", "slow", "period", "no_forecast_weeks"]]
    L = [f"# SMA-pair surface: gain over the plain SMA rule on {sym} (auto-generated by scripts/run_ablation_sma_grid.py)", "", __doc__.split("Usage")[0],
         check, "",
         f"## Gain over `{BASE}` on the same pair: share of pairs that improve, and median / min / max gain (Sharpe difference; CAGR and holding CAGR in pp per year)", "",
         md_table(summ, "{:.2f}"), "",
         "## Uptrend weeks without a forecast (the walk-forward skips a year with fewer than 60 training weeks; such weeks are held in cash)", "",
         md_table(skipped) if len(skipped) else "none", ""]
    (OUT / "summary.md").write_text("\n".join(L))
    print(md_table(summ, "{:.2f}"))
    print(f"[{time.time() - t0:5.1f}s] done; uptrend weeks without a forecast: {len(skipped)} pair-periods")


if __name__ == "__main__":
    main()
