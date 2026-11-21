# Data

Everything the code reads is in this folder. The data was purchased by the repository owner from a commercial vendor (Rethink; packages dated 2026-08-05). It is included so that the results can be reproduced.

All prices are split- and dividend-adjusted by the vendor, so they approximate total return; dividends are not added again. The sample used in the report is 2000–2025. Bars from 1999 initialize the moving averages, and 2026 bars are never used to complete a label.

## QQQ, SPY and two other index funds

| File | Content | SHA-256 |
| --- | --- | --- |
| `processed/QQQ.csv` | QQQ daily bars, 1999-03-10 to 2026-08-04 | `32bc10140c4f2956a71bba3698a32e0db0d299f2b41e55e7b5f856b8f1fed1df` |
| `processed/SPY.csv` | SPY daily bars, 1993-01-29 to 2026-08-04 | `6b07e1636b1ecc276f17d6d474bb9ec864f7efe829d955c1f83d9f84088f4313` |
| `processed/XLK.csv`, `processed/IWM.csv` | The bars of `raw/market/XLK.csv` and `raw/market/IWM.csv` in the same column layout | |

These four files are the vendor's bars with the columns renamed to `date, symbol, open, high, low, close, volume`. `src/ml_filter/data.py` reads them. QQQ is the traded asset of the report. SPY is one of the five cases of the first design and enters the cross-asset features. XLK and IWM are used only for the checks on other index funds in `scripts/diagnose_qqq_state.py` and `scripts/run_qqq_reverse.py`.

## Nasdaq-100 point-in-time members

| File | Content | SHA-256 |
| --- | --- | --- |
| `raw/nasdaq100/nasdaq100_split_div_adjusted_20260805.zip` | Vendor package: 480 per-security CSV files, including delisted names. Adjusted daily OHLCV, `Unadjusted Close`, and a daily `InIndex` flag | `f8ba17a64886d318e068dfbff734424c926a6858361504c2c87261eaea90ac89` |
| `raw/nasdaq100/security_master.csv` | One row per source file: security id, ticker, price coverage, membership coverage | `66dd94849793cdb5283a15ce5d712b234a064817da36e908860ad0db8cb68223` |
| `raw/nasdaq100/membership_intervals.csv` | Membership intervals reconstructed from the daily `InIndex` flag | `fbef73e99b37e20f3226503bdcb10efb27868408c45a59e9a5b2580d5cccc967` |
| `raw/nasdaq100/XNYS.csv` | NYSE session calendar used to build the weekly grid | |
| `raw/nasdaq100/quality_report.md` | Quality audit of the package (in Chinese), written when the data was checked | |

`src/ml_filter/universe.py` reads the ZIP file directly.

- **Membership.** A stock counts as a member on a decision date if an interval in `membership_intervals.csv` covers the session before that date.
- **Identity.** Securities are identified by source file. No predecessor or successor is merged.
- **Delisting.** A holding with no later quote is settled at its last known close.
- **Quality.** The audit found no blocking issue. It lists 742 close-to-close moves flagged as outliers and some early dates with fewer than 100 members. No flagged value was removed or winsorized. The audit marks the package as a candidate pending review, which is why the report calls it a vendor candidate and asks for the stock results to be read as indicative.

## Market-state series

One vendor CSV per symbol in `raw/market/`, read by `src/ml_filter/xfeat.py`.

| File | Content | Starts |
| --- | --- | --- |
| `VIX.csv`, `VIX3M.csv` | Cboe volatility indices, daily OHLC (index levels, no adjustment) | 1990 / 2002 |
| `TLT.csv`, `IWM.csv` | ETFs, adjusted daily OHLCV | 2002 / 2000 |
| `XLB.csv`, `XLE.csv`, `XLF.csv`, `XLI.csv`, `XLK.csv`, `XLP.csv`, `XLU.csv`, `XLV.csv`, `XLY.csv` | Sector SPDR ETFs, adjusted daily OHLCV | 1998-12 |

The 16 features of the report use VIX, IWM and the nine sector funds. `VIX3M` and `TLT` start after 2000 and enter only the `all+late` feature set of the 2,160-configuration grid in `scripts/run_qqq_state.py`, with median imputation.

## Caches

The stock scripts build three caches in `processed/` on first use: `nasdaq100_panel.pkl`, `nasdaq100_sma_grid.pkl` and `nasdaq100_xfeat.pkl`, about 460 MB together. They are not in the repository and are rebuilt automatically, which takes a few minutes.
