"""Figures 1 to 6 of the report, written to reports/latex/figures/.

The figures only read saved results:
  Figure 1  reports/exhibits/qqq_daily.csv, equity_weekly.csv     QQQ, its two averages and the SMA20/140 rule
  Figure 2  reports/exhibits/example_daily.csv                    the PLTR episode under the dual-SMA rule
  Figure 3  reports/exhibits/equity_weekly.csv                    QQQ: buy-and-hold, the rule, the rule with the Ridge filter
  Figure 4  reports/exhibits/equity_weekly.csv                    the three-component strategy
  Figure 5  reports/exhibits/example_daily.csv, example_weeks.csv the PLTR episode with the three-component portfolio
  Figure 6  results/ablation/sma_grid/gain_vs_sma.csv             222 moving-average pairs, 2019-2025
Run scripts/make_report_exhibits.py and scripts/run_ablation_sma_grid.py first. Figures 7 and A1 to A3 are drawn by
scripts/run_ablation_blocks.py and scripts/run_ablation_sma_grid.py; scripts/make_report_tables.py copies them next to these.
Usage: .venv/bin/python scripts/make_report_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

ROOT = Path(__file__).resolve().parents[1]
RES, EXH = ROOT / "results", ROOT / "reports" / "exhibits"
FIG = ROOT / "reports" / "latex" / "figures"

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e3e2dd", "#fcfcfb"
# one colour per strategy, the same in every figure
COL = {"three": "#2a78d6", "bh": "#eb6834", "sma": "#1baf7a", "mom": "#eda100", "ridge": "#6250d6", "fast": "#6da7ec", "slow": "#184f95"}
STYLE = {
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "figure.dpi": 110, "savefig.dpi": 160,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "xtick.major.size": 0, "ytick.major.size": 0, "font.size": 10, "axes.titlesize": 11.5,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2, "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round", "legend.frameon": False, "legend.fontsize": 9.5}
K = 10  # number of stocks in the momentum book shown in Figure 4


def log_axis(ax, ticks, fmt):
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt.format(v)))


def end_labels(ax, items, x_text, gap=0.075):
    # direct labels to the right of each line end; labels that would overlap are spread out and joined by a thin leader
    lo, hi = np.log10(ax.get_ylim())
    items = sorted(items, key=lambda i: i[1])
    pos = [np.log10(i[1]) for i in items]
    for k in range(1, len(pos)):
        pos[k] = max(pos[k], pos[k - 1] + gap * (hi - lo))
    for (text, y, x_end, color), p in zip(items, pos):
        ax.plot([x_end, x_text], [y, 10 ** p], color=INK2, lw=0.6, zorder=4)
        ax.plot([x_text], [10 ** p], "o", ms=8, color=color, mec=SURF, mew=2, zorder=5)   # colour key beside its label
        ax.text(x_text, 10 ** p, "   " + text, va="center", ha="left", color=INK, fontsize=9.5)


def year_axis(ax, y0, y1, step=5, pad_years=0.0):
    ax.set_xlim(pd.Timestamp(f"{y0}-01-01"), pd.Timestamp(f"{y1}-12-31") + pd.Timedelta(days=365.25 * pad_years))
    ax.set_xticks([pd.Timestamp(f"{y}-01-01") for y in range(y0, y1 + 1, step)], [str(y) for y in range(y0, y1 + 1, step)])


def day(d):
    return f"{d.day} {d:%B}"


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)


def fig1_qqq_rule(E):
    q = pd.read_csv(EXH / "qqq_daily.csv", parse_dates=["date"], index_col="date").loc["2005":]

    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 6.8), sharex=True, height_ratios=[1.2, 1], layout="constrained")
    a.fill_between(q.index, 0, 1, where=q.sma20 > q.sma140, transform=a.get_xaxis_transform(), color=COL["sma"], alpha=0.12, lw=0)
    a.plot(q.index, q.close, color=INK2, lw=1.0)
    a.plot(q.index, q.sma20, color=COL["fast"], lw=1.5)
    a.plot(q.index, q.sma140, color=COL["slow"], lw=1.5)
    log_axis(a, [25, 50, 100, 200, 400], "{:.0f}")
    a.set_title("QQQ price with its 20-day and 140-day averages")
    a.legend(handles=[plt.Line2D([], [], color=INK2, lw=1.0, label="QQQ close (adjusted, log scale)"),
                      plt.Line2D([], [], color=COL["fast"], lw=1.5, label="20-day average"),
                      plt.Line2D([], [], color=COL["slow"], lw=1.5, label="140-day average"),
                      Patch(color=COL["sma"], alpha=0.25, label="rule is invested (20-day above 140-day)")], loc="upper left", ncols=2)

    for key, name in (("bh", "QQQ buy-and-hold"), ("sma", "SMA (QQQ)")):
        b.plot(E.index, E[name], color=COL[key], label={"bh": "Buy and hold", "sma": "SMA20/140 rule"}[key])
    log_axis(b, [1, 2, 5, 10, 20], "{:.0f}×")
    b.set_ylim(0.55, 30)
    b.set_title("Value of 1 invested at the start of 2005 (after costs)")
    b.legend(loc="upper left")
    year_axis(b, 2005, 2025, pad_years=4.4)
    end_labels(b, [(f"Buy and hold  {E['QQQ buy-and-hold'].iloc[-1]:.1f}×", E["QQQ buy-and-hold"].iloc[-1], E.index[-1], COL["bh"]),
                   (f"SMA20/140 rule  {E['SMA (QQQ)'].iloc[-1]:.1f}×", E["SMA (QQQ)"].iloc[-1], E.index[-1], COL["sma"])],
               E.index[-1] + pd.Timedelta(days=150))
    save(fig, "fig1_qqq_rule.png")


def example_chart(fig, ax, x, xw, with_filter):
    """The PLTR episode, late May to September 2025; with_filter adds the trades of the three-component portfolio."""
    decline = x.loc["2025-08-11":"2025-09-02"]
    peak, trough = decline.close.idxmax(), decline.close.idxmin()
    ax.axvspan(x.index[0], x.index[-1], color=COL["sma"], alpha=0.10, lw=0)
    handles = [plt.Line2D([], [], color=INK2, lw=1.3, label="PLTR close (adjusted)"),
               plt.Line2D([], [], color=COL["fast"], lw=2, label="20-day average"),
               plt.Line2D([], [], color=COL["slow"], lw=2, label="140-day average"),
               Patch(color=COL["sma"], alpha=0.22, label="dual-SMA rule on PLTR is invested (every day shown)")]
    if with_filter:
        w = xw[(xw.e_next > x.index[0]) & (xw.e <= x.index[-1])]
        for r in w[~w.keep].itertuples():
            ax.axvspan(max(r.e, x.index[0]), min(r.e_next, x.index[-1]), color=INK2, alpha=0.17, lw=0)
    ax.plot(x.index, x.close, color=INK2, lw=1.3)
    ax.plot(x.index, x.sma20, color=COL["fast"])
    ax.plot(x.index, x.sma140, color=COL["slow"])
    if with_filter:
        trades = w[w.keep != w.keep.shift(1, fill_value=False)]
        spots = [dict(xytext=(16, -58), ha="left"), dict(xytext=(-9, 11), ha="right"), dict(xytext=(9, -19), ha="left"), dict(xytext=(-10, 28), ha="right")]
        for n, r in enumerate(trades.itertuples()):
            buy = bool(r.keep)
            ax.plot([r.e], [r.open_e], "^" if buy else "v", ms=11, color=COL["three"], mec=SURF, mew=1.6, zorder=6)
            word = ("Buys" if n == 0 else "Buys back") if buy else "Sells"
            ax.annotate(f"{word} at {r.open_e:.2f}\n{day(r.e)}", (r.e, r.open_e), textcoords="offset points", color=INK, fontsize=9.5, va="center",
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6, shrinkA=3, shrinkB=8) if n == 0 else None, **spots[n % len(spots)])
        handles += [Patch(color=INK2, alpha=0.30, label="three-component portfolio is in cash"),
                    plt.Line2D([], [], marker="^", ls="", ms=9, color=COL["three"], label="portfolio buys PLTR at the open"),
                    plt.Line2D([], [], marker="v", ls="", ms=9, color=COL["three"], label="portfolio sells PLTR at the open")]
    else:
        for d in (peak, trough):
            ax.plot([d], [x.close[d]], "o", ms=8, color=INK, mec=SURF, mew=2, zorder=5)
        ax.annotate(f"{x.close[peak]:.2f} on {day(peak)}", (peak, x.close[peak]), xytext=(-9, 9), textcoords="offset points", ha="right", color=INK, fontsize=9.5)
        ax.annotate(f"{x.close[trough]:.2f} on {day(trough)}\n−{(1 - x.close[trough] / x.close[peak]) * 100:.1f}% in six trading days", (trough, x.close[trough]),
                    xytext=(10, -22), textcoords="offset points", color=INK, fontsize=9.5)
    ax.set_ylim(90, 200)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax.set_xlim(x.index[0], x.index[-1])
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.legend(handles=handles, loc="outside lower center", ncols=2 if with_filter else 4)


def fig2_example_rule(x, xw):
    fig, ax = plt.subplots(figsize=(10, 4.9), layout="constrained")
    example_chart(fig, ax, x, xw, with_filter=False)
    ax.set_title("PLTR, summer 2025: the averages stay far apart, so the rule holds the stock through the August decline")
    save(fig, "fig2_example_rule.png")


def fig3_qqq_filter(E):
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 6.6), sharex=True, height_ratios=[1.5, 1], layout="constrained")
    series = (("bh", "QQQ buy-and-hold", "Buy and hold"), ("sma", "SMA (QQQ)", "SMA20/140 rule"), ("ridge", "SMA + Ridge (QQQ)", "Rule + Ridge filter"))
    for key, name, label in series:
        a.plot(E.index, E[name], color=COL[key], label=label)
        b.plot(E.index, (E[name] / E[name].cummax() - 1) * 100, color=COL[key], lw=1.6, label=label)
    log_axis(a, [1, 2, 5, 10, 20], "{:.0f}×")
    a.set_ylim(0.55, 30)
    a.set_title("Value of 1 invested at the start of 2005 (after costs)")
    a.legend(loc="upper left")
    end_labels(a, [(f"{label}  {E[name].iloc[-1]:.1f}×", E[name].iloc[-1], E.index[-1], COL[key]) for key, name, label in series], E.index[-1] + pd.Timedelta(days=150))
    b.set_title("Drawdown from the previous peak")
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%".replace("-", "−")))
    b.legend(loc="center right")
    year_axis(b, 2005, 2025, pad_years=5.8)
    save(fig, "fig3_qqq_filter.png")


def fig4_three_component(E):
    fig, (a, b) = plt.subplots(2, 1, figsize=(10, 6.9), sharex=True, height_ratios=[1.55, 1], layout="constrained")
    series = (("three", f"SMA + Ridge + momentum top {K}", "SMA + Ridge + momentum"), ("mom", f"Momentum top {K}", "Momentum, always invested"),
              ("bh", "QQQ buy-and-hold", "QQQ buy and hold"), ("sma", "SMA (QQQ)", "SMA20/140 rule on QQQ"))
    for key, name, label in series:
        a.plot(E.index, E[name], color=COL[key], label=label, zorder=3 if key == "three" else 2)
    log_axis(a, [1, 2, 5, 10, 20, 40], "{:.0f}×")
    a.set_ylim(0.5, 75)
    a.set_title("Value of 1 invested at the start of 2005 (after costs)")
    a.legend(loc="upper left")
    end_labels(a, [(f"{label}  {E[name].iloc[-1]:.1f}×", E[name].iloc[-1], E.index[-1], COL[key]) for key, name, label in series], E.index[-1] + pd.Timedelta(days=150))
    for key, name, label in series[:2]:
        dd = (E[name] / E[name].cummax() - 1) * 100
        b.fill_between(E.index, dd, 0, color=COL[key], alpha=0.10, lw=0)
        b.plot(E.index, dd, color=COL[key], lw=1.6, label=label)
    b.set_title("Drawdown from the previous peak")
    b.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%".replace("-", "−")))
    b.legend(loc="center right")
    year_axis(b, 2005, 2025, pad_years=7.2)
    save(fig, "fig4_three_component.png")


def fig5_example_filter(x, xw):
    fig, ax = plt.subplots(figsize=(10, 5.3), layout="constrained")
    example_chart(fig, ax, x, xw, with_filter=True)
    ax.set_title("The same episode with the three-component portfolio: sold on 11 August, bought back on 2 September")
    save(fig, "fig5_example_filter.png")


def fig6_sma_pairs():
    gv = pd.read_csv(RES / "ablation" / "sma_grid" / "gain_vs_sma.csv")
    sub = gv[(gv.strategy == "SMA + Ridge + momentum top 10") & (gv.period == "holdout 2019-2025")]
    FAST, SLOW = sorted(sub.fast.unique()), sorted(sub.slow.unique())
    CMAP = LinearSegmentedColormap.from_list("red_gray_blue", ["#7a1616", "#b02a2a", "#e34948", "#ee8f88", "#f7c4bf", "#f0efec", "#b7d3f6", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
    fig, axes = plt.subplots(3, 1, figsize=(11, 14.2), layout="constrained")
    for ax, (col, label, unit, fmt) in zip(axes, (("d_cagr", "CAGR", "pp per year", "{:+.1f}"), ("d_hold_cagr", "Holding CAGR", "pp per year", "{:+.0f}"), ("d_sharpe", "Sharpe ratio", "difference", "{:+.2f}"))):
        M = sub.pivot(index="fast", columns="slow", values=col).reindex(index=FAST, columns=SLOW).to_numpy()
        norm = Normalize(-np.nanpercentile(np.abs(M), 95), np.nanpercentile(np.abs(M), 95))
        im = ax.pcolormesh(np.ma.masked_invalid(M), cmap=CMAP, norm=norm, edgecolors=SURF, linewidth=1.6)
        ax.invert_yaxis()
        ax.grid(False)
        for i in range(len(FAST)):
            for j in range(len(SLOW)):
                if np.isfinite(M[i, j]):
                    r_, g_, b_, _ = CMAP(norm(M[i, j]))
                    ax.text(j + 0.5, i + 0.5, fmt.format(M[i, j]), ha="center", va="center", fontsize=8, color="white" if 0.2126 * r_ + 0.7152 * g_ + 0.0722 * b_ < 0.42 else INK)
        ax.set_xticks(np.arange(len(SLOW)) + 0.5, SLOW)
        ax.set_yticks(np.arange(len(FAST)) + 0.5, FAST)
        ax.set_xlabel("long average (days)")
        ax.set_ylabel("short average (days; 1 = close)")
        ax.spines["bottom"].set_visible(False)
        ok = np.isfinite(M)
        ax.set_title(f"{label}: better than the plain rule in {int((M[ok] > 0).sum())} of {int(ok.sum())} pairs; median {fmt.format(np.nanmedian(M))} ({unit})")
        cb = fig.colorbar(im, ax=ax, shrink=0.8, aspect=22, pad=0.012, extend="both")
        cb.set_label(f"{unit}  (blue = better than the plain rule)", color=INK2)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0)
    save(fig, "fig6_sma_pairs_2019_2025.png")


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(STYLE)
    E = pd.read_csv(EXH / "equity_weekly.csv", parse_dates=["date"], index_col="date")   # weekly equity, 2005-2025
    x = pd.read_csv(EXH / "example_daily.csv", parse_dates=["date"], index_col="date").loc["2025-05-27":"2025-09-30"]
    xw = pd.read_csv(EXH / "example_weeks.csv", parse_dates=["d", "e", "e_next"])
    fig1_qqq_rule(E)
    fig2_example_rule(x, xw)
    fig3_qqq_filter(E)
    fig4_three_component(E)
    fig5_example_filter(x, xw)
    fig6_sma_pairs()
    print("figures:", ", ".join(sorted(p.name for p in FIG.glob("fig[1-6]_*.png"))))


if __name__ == "__main__":
    main()
