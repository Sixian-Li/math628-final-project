"""LaTeX tables and figure copies for reports/latex/main.tex.

Every computed table of the report is written here as a booktabs `tabular` fragment under reports/latex/tables/, read straight
from the saved result files, so the numbers in the PDF cannot drift from the results. Figures 7 and A1 to A3 are copied from
results/ to reports/latex/figures/, next to Figures 1 to 6 (scripts/make_report_figures.py), so that the folder reports/latex/
compiles on its own. The script also prints the few numbers that the text quotes next to Table 9 and in the single-stock example.
Usage: .venv/bin/python scripts/make_report_tables.py
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, EXH = ROOT / "results", ROOT / "reports" / "exhibits"
TEX = ROOT / "reports" / "latex"
TAB, FIG = TEX / "tables", TEX / "figures"
FEATURES = {"vol20": ("Realized volatility", "Volatility"), "vix": ("VIX level", "Volatility"), "vix_rel_1y": ("VIX relative to its average", "Volatility"),
            "vix_chg20": ("VIX change", "Volatility"), "vrp": ("Implied minus realized variance", "Volatility"),
            "breadth50": ("Short-term breadth", "Breadth"), "breadth140": ("Trend breadth", "Breadth"), "breadth200": ("Long-term breadth", "Breadth"),
            "newhigh_share": ("New highs", "Breadth"), "dispersion": ("Dispersion", "Breadth"), "corr_proxy": ("Correlation proxy", "Breadth"),
            "breadth50_chg4w": ("Breadth change", "Breadth"), "spy_rel60": ("Nasdaq against the market", "Cross-asset"),
            "iwm_rel60": ("Small caps against the market", "Cross-asset"), "def_cyc60": ("Defensive against cyclical", "Cross-asset"),
            "sector_breadth": ("Sector breadth", "Cross-asset")}


def esc(v) -> str:
    s = str(v)
    s = re.sub(r"(?<![\w–])-(?=\d)", "$-$", s).replace("−", "$-$")   # minus signs
    s = re.sub(r"\+(?=\d)", "$+$", s)
    for a, b in (("%", r"\%"), ("&", r"\&"), ("_", r"\_"), ("#", r"\#"), ("–", "--"), ("α", r"$\alpha$"), ("×", r"$\times$")):
        s = s.replace(a, b)
    return s


def write(name: str, colspec: str, header: list[str], rows: list[list], midrules: tuple[int, ...] = ()):
    lines = [rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", *header, r"\midrule"]
    for i, r in enumerate(rows):
        if i in midrules:
            lines.append(r"\addlinespace")
        lines.append(" & ".join(esc(v) for v in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (TAB / f"{name}.tex").write_text("\n".join(lines))


def trio(r) -> str:
    return f"{r.cagr:.1f} / {r.hold_cagr:.1f} / {r.sharpe:.2f}".replace("-0.0 ", "0.0 ")


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    AB = pd.read_csv(RES / "ablation" / "grid.csv").set_index(["universe", "book", "timing", "period"])
    P3 = {"dev 2005-2018": "2005–2018", "holdout 2019-2025": "2019–2025", "full 2005-2025": "2005–2025"}
    cgs = r"\footnotesize CAGR / holding CAGR / Sharpe"

    # Table 2: first design on SMA20/140
    c = RES / "cases" / "QQQ_SMA20_140"
    m = pd.read_csv(c / "metrics.csv").query("cost_bps == 5").set_index("strategy")
    p = pd.read_csv(c / "prediction_metrics.csv").set_index("strategy")
    wf = pd.read_csv(c / "robustness" / "wf_metrics.csv").set_index("strategy")
    names = {"buy_hold": "QQQ buy-and-hold", "base": "SMA20/140 rule", "vol_filter": "Rule + volatility filter", "logit": "Rule + logistic regression",
             "rf": "Rule + random forest", "gb": "Rule + gradient boosting"}
    rows = []
    for k, name in names.items():
        ml = k in ("logit", "rf", "gb")
        rows.append([name, f"{m.loc[k, 'cagr_pct']:.1f}", f"{m.loc[k, 'holding_cagr_pct']:.1f}", f"{m.loc[k, 'exposure_pct']:.0f}",
                     f"{p.loc[k, 'retained_pct']:.1f}" if k in p.index else ("100.0" if k == "base" else ""),
                     f"{p.loc[k, 'auc']:.2f}" if ml else "", f"{p.loc[k, 'brier_skill_pct']:+.2f}" if ml else "",
                     f"{wf.loc[k, 'cagr_pct']:.1f}", f"{wf.loc[k, 'holding_cagr_pct']:.1f}"])
    write("tab02_first_design", "lrrrrrrrr",
          [r" & \multicolumn{6}{c}{Test period 2019--2025} & \multicolumn{2}{c}{Walk-forward 2005--2025} \\", r"\cmidrule(lr){2-7}\cmidrule(lr){8-9}",
           r"Strategy & CAGR & Holding & Exposure & Rule weeks & AUC & Brier & CAGR & Holding \\",
           r" & (\%) & CAGR (\%) & (\%) & kept (\%) & & skill (\%) & (\%) & CAGR (\%) \\"], rows, midrules=(3,))

    # Table 6: Ridge filter on QQQ
    rows = []
    for name, timing in (("QQQ buy-and-hold", "always"), ("SMA20/140 rule", "sma"), ("SMA20/140 rule + Ridge filter", "sma+ridge")):
        f = AB.loc[("-", "QQQ ETF", timing, "full 2005-2025")]
        rows.append([name, *[trio(AB.loc[("-", "QQQ ETF", timing, pk)]) for pk in P3], f"{f.invested:.0f}", f"{f.maxdd:.0f}"])
    write("tab06_qqq_filter", "lcccrr",
          [r"Strategy & 2005--2018 & 2019--2025 & 2005--2025 & Exposure & Max drawdown \\", rf" & \multicolumn{{3}}{{c}}{{{cgs}}} & (\%) & (\%) \\"], rows)

    # Tables 7 and 8: features x target, and models
    g = pd.read_csv(RES / "qqq_state" / "QQQ_sma20_140" / "grid.csv")
    base = g[g.model == "base"].set_index("period").hold_cagr
    w = g[~g.model.isin(["buy_hold", "base", "union"])].pivot_table(index=["cand", "feats", "h", "model", "thr"], columns="period", values="hold_cagr").reset_index()
    for per in ("dev", "holdout"):
        w["d_" + per] = w[per] - base[per]
    new = ["breadth", "breadth+st", "new", "new+cross", "new+cross+st"]
    w = w[(w.cand == "base") & (w.h == 1)]
    rows = []
    for label, sel in (("Original 7 price features", w.feats == "price7"), ("Market-state features", w.feats.isin(new))):
        r = [label]
        for mk in ("logit", "ridge10"):
            d = w[sel & (w.model == mk)]
            r.append(f"{d.d_dev.mean():+.1f} / {d.d_holdout.mean():+.1f}")
        rows.append(r)
    write("tab07_features_target", "lcc", [r"Features & Target = direction (logistic) & Target = return (Ridge) \\"], rows)

    models = {"ridge100": "Ridge, α = 100", "lasso": "Lasso", "ridge10": "Ridge, α = 10 (reference)", "logit_w": "Logistic, weighted by the size of the return",
              "rf": "Random forest", "gb_d1": "Gradient boosting, depth 1", "logit": "Logistic (direction)", "svr": "Support vector regression",
              "gb_d2": "Gradient boosting, depth 2", "knn": "K nearest neighbours"}
    rows = []
    for mk, name in models.items():
        d = w[w.feats.isin(new) & (w.model == mk)]
        rows.append([name, len(d), f"{d.d_dev.mean():+.1f}", f"{(d.d_dev > 0).mean() * 100:.0f}", f"{d.d_holdout.mean():+.1f}", f"{(d.d_holdout > 0).mean() * 100:.0f}",
                     f"{((d.d_dev > 0) & (d.d_holdout > 0)).mean() * 100:.0f}"])
    write("tab08_models", "lrrrrrr",
          [r" & & \multicolumn{2}{c}{2005--2018} & \multicolumn{2}{c}{2019--2025} & Both periods \\", r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-7}",
           r"Model & Configurations & Mean change & Beats the & Mean change & Beats the & Beats the \\", r" & & (pp) & rule (\%) & (pp) & rule (\%) & rule (\%) \\"], rows, midrules=(3,))

    # Table 9: fitted Ridge coefficients, last annual fit and stability over all annual fits
    co = pd.read_csv(EXH / "ridge_coefficients.csv").set_index("year")
    last = co.iloc[-1]
    rows = []
    for f in sorted(FEATURES, key=lambda f: last[f]):
        rows.append([FEATURES[f][0], FEATURES[f][1], f"{last[f]:+.3f}", f"{last[f] * last.sd_target * 1e4:+.0f}", f"{co[f].min():+.2f} to {co[f].max():+.2f}",
                     f"{(np.sign(co[f]) == np.sign(last[f])).mean() * 100:.0f}"])
    write("tab09_coefficients", "llrrcr", [rf" & & \multicolumn{{2}}{{c}}{{Last fit}} & \multicolumn{{2}}{{c}}{{All {len(co)} annual fits}} \\", r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
                                           r"Feature & Block & Coefficient & Effect of $+1$ s.d. (bp) & Range & Same sign (\%) \\"], rows)
    print(f"Table 9, last fit: {int(last.n_train)} rule weeks, 2000-{co.index[-1] - 1}; standard deviation of the weekly return {last.sd_target * 100:.2f}%; "
          f"intercept {last.intercept:.3f} ({last.intercept * last.sd_target * 1e4:.1f} bp); in-sample R2 {last.r2_in_sample * 100:.1f}%; "
          f"correlation between the coefficients of the first and the last fit {np.corrcoef(co[list(FEATURES)].iloc[0], co[list(FEATURES)].iloc[-1])[0, 1]:.2f}")

    # Table 10: ablation with a ten-stock book; Table 11: number of stocks
    strats = [("SMA + Ridge + momentum", ("trend", "momentum top10", "sma+ridge")), ("SMA + momentum", ("trend", "momentum top10", "sma")),
              ("Ridge + momentum", ("all", "momentum top10", "ridge_only")), ("Momentum, always invested", ("all", "momentum top10", "always")),
              ("SMA + Ridge (QQQ)", ("-", "QQQ ETF", "sma+ridge")), ("SMA (QQQ)", ("-", "QQQ ETF", "sma")), ("Ridge (QQQ)", ("-", "QQQ ETF", "ridge_only")),
              ("QQQ buy-and-hold", ("-", "QQQ ETF", "always"))]
    rows = []
    for name, key in strats:
        f = AB.loc[(*key, "full 2005-2025")]
        rows.append([name, *[trio(AB.loc[(*key, pk)]) for pk in (*P3, "rev 2001-2008")], f"{f.invested:.0f}", f"{f.maxdd:.0f}"])
    write("tab10_ablation", "lccccrr",
          [r" & \multicolumn{3}{c}{Annual walk-forward} & Reverse-time & \multicolumn{2}{c}{2005--2025} \\", r"\cmidrule(lr){2-4}\cmidrule(lr){5-5}\cmidrule(lr){6-7}",
           r"Strategy & 2005--2018 & 2019--2025 & 2005--2025 & 2001--2008 & Exposure & Max drawdown \\", rf" & \multicolumn{{4}}{{c}}{{{cgs}}} & (\%) & (\%) \\"], rows, midrules=(4,))

    rows = []
    for k in (10, 20, 30):
        for name, key in ((f"SMA + Ridge + momentum, top {k}", ("trend", f"momentum top{k}", "sma+ridge")), (f"Momentum, always invested, top {k}", ("all", f"momentum top{k}", "always"))):
            rows.append([name, *[trio(AB.loc[(*key, pk)]) for pk in P3]])
    rows.append(["SMA + Ridge (QQQ), for comparison", *[trio(AB.loc[("-", "QQQ ETF", "sma+ridge", pk)]) for pk in P3]])
    write("tab11_book_size", "lccc", [r"Strategy & 2005--2018 & 2019--2025 & 2005--2025 \\", rf" & \multicolumn{{3}}{{c}}{{{cgs}}} \\"], rows, midrules=(2, 4, 6))

    # the single-stock example in numbers: the portfolio's trades in PLTR, late May to September 2025
    xw = pd.read_csv(EXH / "example_weeks.csv", parse_dates=["d", "e", "e_next"])
    w = xw[(xw.e_next > "2025-05-27") & (xw.e <= "2025-09-30")].reset_index(drop=True)
    rows, last = [], None
    for n, r in enumerate(w[w.keep != w.keep.shift(1, fill_value=False)].itertuples()):
        buy = bool(r.keep)
        rows.append([("Buy" if n == 0 else "Buy back") if buy else "Sell", f"{r.e.day} {r.e:%B} 2025", f"{r.open_e:.2f}",
                     "" if last is None else f"{(r.open_e / last - 1) * 100:+.1f}% while " + ("in cash" if buy else "invested")])
        last = r.open_e
    write("tab_example", "llrl", [r"Trade & Date & PLTR open & Change since the previous trade \\"], rows)
    y = xw[(xw.e >= "2025-01-01") & (xw.e <= "2025-12-31")]
    print(f"PLTR example, whole of 2025: portfolio invested in {int(y.keep.sum())} of {len(y)} weeks; PLTR {((1 + y.ret[y.keep]).prod() - 1) * 100:+.1f}% over those weeks "
          f"and {((1 + y.ret[~y.keep]).prod() - 1) * 100:+.1f}% over the others; PLTR in the momentum top ten in {int(y.in_top10.sum())} weeks")

    # Table 12: 222 SMA pairs, walk-forward and reverse-time
    def pair_row(s):
        return [f"{s['CAGR: better %']:.0f}", f"{s['CAGR: median']:+.1f}", f"{s['Holding CAGR: better %']:.0f}", f"{s['Holding CAGR: median']:+.1f}",
                f"{s['Sharpe: better %']:.0f}", f"{s['Sharpe: median']:+.2f}"]

    pair_head = [r" & \multicolumn{2}{c}{CAGR} & \multicolumn{2}{c}{Holding CAGR} & \multicolumn{2}{c}{Sharpe ratio} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
                 r"Period & Pairs & Median & Pairs & Median & Pairs & Median \\", r" & better (\%) & gain (pp) & better (\%) & gain (pp) & better (\%) & gain \\"]
    sg = pd.read_csv(RES / "ablation" / "sma_grid" / "summary.csv").set_index(["strategy", "period"])
    pw = {"dev 2005-2018": "2005–2018 (walk-forward)", "holdout 2019-2025": "2019–2025 (walk-forward)", "full 2005-2025": "2005–2025 (walk-forward)", "rev 2001-2008": "2001–2008 (reverse-time)"}
    rows = [[r"@SMA + Ridge + momentum, top 10"]] + [[pl, *pair_row(sg.loc[("SMA + Ridge + momentum top 10", pk)])] for pk, pl in pw.items()]
    rows += [[r"@SMA + Ridge (QQQ)"]] + [[pl, *pair_row(sg.loc[("SMA + Ridge (QQQ)", pk)])] for pk, pl in pw.items()]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule", *pair_head, r"\midrule"]
    for r in rows:
        if len(r) == 1:
            lines += ([r"\addlinespace"] if r[0] != rows[0][0] else []) + [rf"\multicolumn{{7}}{{l}}{{\textit{{{esc(r[0][1:])}}}}} \\"]
        else:
            lines.append(r"\quad " + " & ".join(esc(v) for v in r) + r" \\")
    (TAB / "tab12_sma_pairs.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}", ""]))

    # Tables 13-15: leave-one-period-out test
    h = pd.read_csv(RES / "ablation" / "blocks" / "headline_top10.csv")
    h["strategy"] = h.strategy.str.replace(" top 10", "", regex=False).str.replace("Momentum", "Momentum, always invested")
    write("tab13_blocks", "lccccc", [r"Strategy & 2001--2006 & 2007--2012 & 2013--2018 & 2019--2025 & 2001--2025 (chained) \\", rf" & \multicolumn{{5}}{{c}}{{{cgs}}} \\"],
          h.values.tolist(), midrules=(4,))
    bs = pd.read_csv(RES / "ablation" / "blocks" / "summary.csv").set_index(["strategy", "period"])
    rows = [[pk.replace("-", "–") + (" (chained)" if pk == "2001-2025" else ""), *pair_row(bs.loc[("SMA + Ridge + momentum top 10", pk)])]
            for pk in ("2001-2006", "2007-2012", "2013-2018", "2019-2025", "2001-2025")]
    write("tab14_blocks_pairs", "lrrrrrr", pair_head, rows, midrules=(4,))
    vs = pd.read_csv(RES / "ablation" / "blocks" / "versus.csv")
    vs = vs[vs.strategy == "SMA + Ridge + momentum top 10"]
    rows = []
    for against in vs.against.unique():
        label = against.replace(" top 10", "")
        r = ["Momentum, always invested" if label == "Momentum" else label]
        for pk in ("2013-2018", "2019-2025", "2001-2025"):
            s = vs[(vs.against == against) & (vs.period == pk)].iloc[0]
            r.append(f"{s['CAGR: better %']:.0f} / {s['Holding CAGR: better %']:.0f} / {s['Sharpe: better %']:.0f}")
        rows.append(r)
    write("tab15_blocks_versus", "lccc", [r"Compared with & 2013--2018 & 2019--2025 & 2001--2025 (chained) \\",
                                          r" & \multicolumn{3}{c}{\footnotesize pairs better (\%) on CAGR / holding CAGR / Sharpe} \\"], rows, midrules=(3,))

    # Figures 7 and A1 to A3 are drawn by the two SMA-pair scripts; Figures 1 to 6 are written by scripts/make_report_figures.py
    for src, dst in ((RES / "ablation" / "blocks" / "heat_cagr_top10.png", "fig7_blocks_cagr.png"), (RES / "ablation" / "blocks" / "heat_hold_cagr_top10.png", "figA1_blocks_hold_cagr.png"),
                     (RES / "ablation" / "blocks" / "heat_sharpe_top10.png", "figA2_blocks_sharpe.png"), (RES / "ablation" / "sma_grid" / "heat_cagr_top10.png", "figA3_walkforward_cagr.png")):
        shutil.copyfile(src, FIG / dst)
    print("tables:", ", ".join(sorted(q.name for q in TAB.iterdir())))
    print("figures:", ", ".join(sorted(q.name for q in FIG.iterdir())))


if __name__ == "__main__":
    main()
