# Does Machine Learning Add Selection Skill to a Trend Strategy?

MATH 628, Machine Learning for Finance I — final project, Fall 2026.

Authors: Sixian Li, Tai Karir, Zizhao Ding.

This repository holds the final report and the code, data and results behind every table, figure and number in it. The report asks whether a machine-learning filter can tell a moving-average trend rule on QQQ which of its invested weeks to skip, and what a momentum ranking of Nasdaq-100 stocks adds on top.

The report is [reports/final_report.pdf](reports/final_report.pdf). Its LaTeX source is [reports/latex/main.tex](reports/latex/main.tex).

## Layout

| Path | Content |
| --- | --- |
| `reports/final_report.pdf` | The report |
| `reports/latex/` | `main.tex`, the table fragments in `tables/` and the figures in `figures/` |
| `reports/exhibits/` | Small series behind Figures 1 to 5, Table 9 and the PLTR example |
| `reports/slides/math628_presentation.pptx` | The presentation slides |
| `proposal/proposal_en_1009.pdf` | The project proposal. It specifies the first design (Section 5 of the report); Appendix D of the report lists what changed since |
| `scripts/` | One script per experiment, plus three `make_report_*.py` scripts that build the exhibits, figures and tables |
| `src/ml_filter/` | Shared code (see below) |
| `results/` | The outputs the report uses |
| `data/` | The data snapshot; sources and licence in [data/README.md](data/README.md) |

Modules in `src/ml_filter/`:

| Module | Content |
| --- | --- |
| `data.py` | Loads the daily bars; weekly decision table with timing, the seven price features and the label |
| `models.py` | Classifiers of the first design, their grids, the annual walk-forward |
| `backtest.py` | Single-asset account marked daily, and its metrics |
| `universe.py` | Point-in-time Nasdaq-100 stock panel on the weekly grid |
| `portfolio.py` | Weekly multi-asset account with costs on every weight change |
| `meta.py` | Moving averages of every stock; trade-level labels used in Appendix A |
| `xfeat.py` | Stock characteristics and the market-state features (volatility, breadth, cross-asset) |

## Setup

Python 3.13 is required. From the repository root:

```sh
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.lock.txt
```

`requirements.txt` lists the six packages the code needs; `requirements.lock.txt` pins every installed package. Compiling the report needs a LaTeX distribution with `latexmk`.

## Reproducing the results

Run the commands in this order from the repository root. Times are from one run on a 14-core laptop (Apple M3 Max); several scripts start up to 12 worker processes. All random seeds are fixed in the code.

| # | Command | What it produces | Time |
| --- | --- | --- | --- |
| 1 | `.venv/bin/python scripts/run_experiments.py` | First design on the proposal's five cases (Appendix A) | 40 s |
| 2 | `ML_CASE=QQQ:sma20_140 .venv/bin/python scripts/run_experiments.py` | First design on SMA20/140 (Table 2, test period) | 10 s |
| 3 | `.venv/bin/python scripts/run_robustness.py` | Walk-forward and blocked cross-validation of the five cases (Appendix A) | 2 min |
| 4 | `ML_CASE=QQQ:sma20_140 .venv/bin/python scripts/run_robustness.py` | The same on SMA20/140 (Table 2, walk-forward columns) | 45 s |
| 5 | `.venv/bin/python scripts/run_ml_only.py` | Classifier without the SMA rule (Appendix A) | 25 s |
| 6 | `.venv/bin/python scripts/run_stocks.py --sma 20_140` | Stocks ranked by classifier score (Appendix A). The first stock script to run also builds the panel cache | 5 min |
| 7 | the same with `--sma 20_60`, `--sma 10_30`, `--sma 40_120` | The other three moving-average pairs behind the same Appendix A item | 4.5 min each |
| 8 | `.venv/bin/python scripts/run_meta_label.py` | Trade-level labels, and buying the rejected trades (Appendix A) | 2.5 min |
| 9 | `.venv/bin/python scripts/run_meta_grid.py` | Trade-level labels over 215 moving-average pairs (Appendix A) | 6.5 min |
| 10 | `.venv/bin/python scripts/prescreen_dev.py > results/prescreen_dev.txt` | Pre-screen on data through 2018 (Table 3, Section 5.3) | 10 s |
| 11 | `.venv/bin/python scripts/run_qqq_state.py` | Ridge filter and the grid of 2,160 configurations (Tables 7 and 8, placebo) | 1 min |
| 12 | `.venv/bin/python scripts/diagnose_qqq_state.py` | Kept and dropped weeks, yearly comparison, costs, feature blocks, feature lag | 5 s |
| 13 | `.venv/bin/python scripts/run_qqq_reverse.py` | Reverse-time test | 15 s |
| 14 | `.venv/bin/python scripts/run_stock_rank.py` | Machine-learning rankings of stocks against the 12-month return (Section 6.4) | 2.5 min |
| 15 | `.venv/bin/python scripts/run_ablation.py` | SMA × Ridge × momentum ablation (Tables 6, 10 and 11) | 5 s |
| 16 | `.venv/bin/python scripts/run_ablation_sma_grid.py` | 222 moving-average pairs, walk-forward and reverse-time (Table 12, Figures 6 and A3) | 30 s |
| 17 | `.venv/bin/python scripts/run_ablation_blocks.py` | Leave-one-period-out test (Tables 13 to 15, Figures 7, A1 and A2) | 30 s |
| 18 | `.venv/bin/python scripts/make_report_exhibits.py` | `reports/exhibits/`: series behind Figures 1 to 5, coefficients of Table 9, the PLTR example | 5 s |
| 19 | `.venv/bin/python scripts/make_report_figures.py` | Figures 1 to 6 in `reports/latex/figures/` | 5 s |
| 20 | `.venv/bin/python scripts/make_report_tables.py` | Every computed table in `reports/latex/tables/`; copies Figures 7 and A1 to A3 | 5 s |

The whole sequence takes about 36 minutes. Steps 16, 17 and 18 compare their own numbers with `results/ablation/grid.csv`, so step 15 has to run before them. Steps 19 and 20 only read saved files; with the files in this repository they run on their own.

To compile the report:

```sh
cd reports/latex
latexmk -pdf main.tex
cp main.pdf ../final_report.pdf
```

## Where each table and figure comes from

| In the report | Result file | Script |
| --- | --- | --- |
| Table 2, Section 5.2 | `results/cases/QQQ_SMA20_140/metrics.csv`, `prediction_metrics.csv`, `robustness/wf_metrics.csv`; readable in `summary.md` and `robustness/summary_robustness.md` | `run_experiments.py`, `run_robustness.py` with `ML_CASE=QQQ:sma20_140` |
| Table 3 and the diagnoses of Section 5.3 | `results/prescreen_dev.txt` | `prescreen_dev.py` |
| Tables 1, 4 and 5 | written in `main.tex` (data, targets, feature definitions) | `src/ml_filter/xfeat.py` implements Table 5 |
| Tables 6, 10 and 11 | `results/ablation/grid.csv`; readable in `results/ablation/summary.md` | `run_ablation.py` |
| Tables 7 and 8 | `results/qqq_state/QQQ_sma20_140/grid.csv` | `run_qqq_state.py` |
| Section 7.1: placebo | `results/qqq_state/QQQ_sma20_140/summary.md` | `run_qqq_state.py` |
| Section 7.1: kept and dropped weeks, yearly comparison, costs. Section 7.2: feature blocks, Lasso reading, feature lag | `results/qqq_state/QQQ_sma20_140/diagnostics.md` | `diagnose_qqq_state.py` |
| Table 9 and the stability of the coefficients | `reports/exhibits/ridge_coefficients.csv` | `make_report_exhibits.py` |
| Figures 1, 3 and 4 | `reports/exhibits/qqq_daily.csv`, `equity_weekly.csv` | `make_report_exhibits.py`, `make_report_figures.py` |
| Figures 2 and 5, the PLTR table | `reports/exhibits/example_daily.csv`, `example_weeks.csv` | `make_report_exhibits.py`, `make_report_figures.py` |
| Section 7.3: the weeks in 2016 and 2020 in which the filter was wrong | `reports/exhibits/timing_weeks.csv` | `make_report_exhibits.py` |
| Section 6.4: machine-learning rankings are not better than the 12-month return | `results/stock_rank/sma20_140/summary.md` | `run_stock_rank.py` |
| Table 12, Figure 6, Figure A3 | `results/ablation/sma_grid/summary.csv`, `gain_vs_sma.csv`, `heat_cagr_top10.png`; readable in `summary.md` | `run_ablation_sma_grid.py` |
| Section 7.4 (b): reverse-time test | `results/ablation/grid.csv` (periods `rev` and `rev_b`), `results/qqq_state/reverse/summary.md` | `run_ablation.py`, `run_qqq_reverse.py` |
| Tables 13 to 15, Figures 7, A1 and A2 | `results/ablation/blocks/headline_top10.csv`, `summary.csv`, `versus.csv`, `heat_*_top10.png`; readable in `summary.md` | `run_ablation_blocks.py` |
| Section 7.5: checks on other index funds | `results/qqq_state/QQQ_sma20_140/diagnostics.md` (part 7), `results/qqq_state/reverse/summary.md` | `diagnose_qqq_state.py`, `run_qqq_reverse.py` |
| Appendix A: the five cases of the proposal | `results/summary.md` | `run_experiments.py` |
| Appendix A: a longer history | `results/robustness/summary_robustness.md` | `run_robustness.py` |
| Appendix A: no SMA rule | `results/ml_only/summary.md` | `run_ml_only.py` |
| Appendix A: whole trades; buying what the classifier rejects | `results/meta_label/sma20_140/summary.md` | `run_meta_label.py` |
| Appendix A: a weaker base rule | `results/meta_label/grid/summary.txt` | `run_meta_grid.py` |
| Appendix A: ranking stocks with the same labels | `results/stocks/sma*/summary.md` (four moving-average pairs) | `run_stocks.py` |
| All table fragments in `reports/latex/tables/` | the files above | `make_report_tables.py` |

Every `summary.md` is written by the script named next to it and shows the same numbers as the CSV files in a readable form.

## What is kept in `results/`

The scripts write more than the report uses: per-run CSV files, equity curves and extra charts. The repository keeps the files that a table, a figure or a number in the report comes from, and `.gitignore` lists them one by one. The other outputs appear in `results/` when the scripts are run and are ignored by git.

A rerun in the environment above reproduces the kept files byte for byte. One row is the exception: the random-forest row of `results/cases/QQQ_SMA20_140/prediction_metrics.csv` can differ in the last digits of the 17 printed (a relative difference of about 1e-13 between two runs). The forest averages its trees on several cores, so the order of the floating-point sums is not fixed. The table fragments in `reports/latex/tables/` are not affected.

## Data and licence

The data was purchased by the repository owner from a commercial vendor and is included so that the results can be checked. It is for grading only: please keep this repository private and do not redistribute the data. Sources, file contents and known quality limits are in [data/README.md](data/README.md).
