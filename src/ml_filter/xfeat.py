"""Extended features (exploratory): long-horizon stock characteristics and market-state variables.

Everything is computed from data available at the d_t close.

stock_features(P)  -> dict of (weeks x assets) arrays, cached in data/processed/nasdaq100_xfeat.pkl
market_features(P) -> DataFrame indexed by decision date d: VIX, breadth, dispersion, cross-asset, QQQ trend state
"""

from __future__ import annotations

import io
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_daily, weekly_table
from .universe import DATA_END, ZIP

ROOT = Path(__file__).resolve().parents[2]
MARKET = ROOT / "data" / "raw" / "market"
XCACHE = ROOT / "data" / "processed" / "nasdaq100_xfeat.pkl"

STOCK_FEATURES = [
    "mom12_1", "mom6_1", "ret252", "ret126", "ret63", "ret21", "ret5",  # momentum / reversal
    "hi52", "lo52", "dist140", "spread140", "price_dev20", "dd60",  # position in range / trend strength
    "vol20", "vol60", "vol252", "max21", "skew60", "beta252", "ivol60",  # risk
    "fip", "amihud60", "dvol_ratio",  # path smoothness, liquidity, volume trend
]


def stock_features(P: dict, force: bool = False) -> dict[str, np.ndarray]:
    if XCACHE.exists() and not force:
        return pickle.loads(XCACHE.read_bytes())
    cal, weeks, files = P["cal"], P["weeks"], P["files"]
    i_d = weeks.i_d.to_numpy()
    out = {f: np.full((len(weeks), len(files)), np.nan, dtype=np.float32) for f in STOCK_FEATURES}
    q = load_daily("QQQ").set_index("date")["close"].pct_change()
    with zipfile.ZipFile(ZIP) as z:
        names = {n.split("/")[-1]: n for n in z.namelist()}
        for j, f in enumerate(files):
            raw = pd.read_csv(io.TextIOWrapper(z.open(names[f]), encoding="utf-8-sig"), usecols=["Date", "Open", "Close", "Volume"])
            raw.columns = ["date", "open", "close", "volume"]
            raw["date"] = pd.to_datetime(raw["date"])
            raw = raw[(raw.date <= DATA_END) & (raw.open > 0) & (raw.close > 0)].drop_duplicates("date").sort_values("date").set_index("date")
            c, r = raw["close"], raw["close"].pct_change()
            m = q.reindex(raw.index)
            dv = (raw["close"] * raw["volume"]).where(raw["volume"] > 0)
            s20, s140 = c.rolling(20).mean(), c.rolling(140).mean()
            beta = r.rolling(252, min_periods=126).cov(m) / m.rolling(252, min_periods=126).var()
            res_var = r.rolling(60).var() - (r.rolling(60).cov(m) ** 2) / m.rolling(60).var()
            d = {
                "mom12_1": c.shift(21) / c.shift(252) - 1, "mom6_1": c.shift(21) / c.shift(126) - 1,
                "ret252": c / c.shift(252) - 1, "ret126": c / c.shift(126) - 1, "ret63": c / c.shift(63) - 1,
                "ret21": c / c.shift(21) - 1, "ret5": c / c.shift(5) - 1,
                "hi52": c / c.rolling(252, min_periods=126).max(), "lo52": c / c.rolling(252, min_periods=126).min() - 1,
                "dist140": c / s140 - 1, "spread140": s20 / s140 - 1, "price_dev20": c / s20 - 1, "dd60": c / c.rolling(60).max() - 1,
                "vol20": r.rolling(20).std(), "vol60": r.rolling(60).std(), "vol252": r.rolling(252, min_periods=126).std(),
                "max21": r.rolling(21).max(), "skew60": r.rolling(60).skew(), "beta252": beta, "ivol60": np.sqrt(res_var.clip(lower=0)),
                "fip": np.sign(c / c.shift(252) - 1) * ((r < 0).rolling(252).mean() - (r > 0).rolling(252).mean()),
                "amihud60": np.log((r.abs() / dv).rolling(60, min_periods=30).mean()),
                "dvol_ratio": dv.rolling(20, min_periods=10).mean() / dv.rolling(252, min_periods=126).mean(),
            }
            for k, v in d.items():
                out[k][:, j] = v.replace([np.inf, -np.inf], np.nan).reindex(cal).to_numpy()[i_d]
    XCACHE.write_bytes(pickle.dumps(out, protocol=5))
    return out


def _mkt(sym: str) -> pd.Series:
    df = pd.read_csv(MARKET / f"{sym}.csv", encoding="utf-8-sig", parse_dates=["Date"])
    return df.drop_duplicates("Date").set_index("Date")["Close"].sort_index()


def market_features(P: dict, S: dict[int, np.ndarray]) -> pd.DataFrame:
    """One row per QQQ decision week. S must hold stock SMA windows 1 (close), 50, 140, 200."""
    weeks, F, mem = P["weeks"], P["feat"], P["tradable"]
    qd = load_daily("QQQ")
    qw = weekly_table(qd)
    c = qd.set_index("date")["close"]
    r = c.pct_change()
    X = pd.DataFrame(index=qw.d)

    def at(s: pd.Series) -> np.ndarray:
        return s.reindex(c.index).ffill().reindex(qw.d).to_numpy()

    # --- QQQ trend state and price features
    s20, s60, s140 = c.rolling(20).mean(), c.rolling(60).mean(), c.rolling(140).mean()
    X["slow_up"] = at((s20 > s140).astype(float).where(s140.notna()))
    X["fast_up"] = at((s20 > s60).astype(float).where(s60.notna()))
    for k in ("ret5", "ret20", "sma_spread", "price_dev", "sma60_slope", "vol20", "dd60"):
        X[k] = qw[k].to_numpy()
    X["ret60"], X["ret120"], X["ret250"] = at(c / c.shift(60) - 1), at(c / c.shift(120) - 1), at(c / c.shift(250) - 1)
    X["dist140"], X["spread140"] = at(c / s140 - 1), at(s20 / s140 - 1)
    X["dd250"], X["vol60"] = at(c / c.rolling(250, min_periods=120).max() - 1), at(r.rolling(60).std())
    # --- implied volatility
    vix, vix3m = _mkt("VIX"), _mkt("VIX3M")
    X["vix"] = at(vix)
    X["vix_rel_1y"] = at(vix / vix.rolling(252).mean())
    X["vix_chg20"] = at(vix / vix.shift(20) - 1)
    X["vrp"] = X["vix"] ** 2 - (X["vol20"] * np.sqrt(252) * 100) ** 2
    X["vix_term"] = at(vix3m / vix.reindex(vix3m.index))  # NaN before 2002
    # --- breadth / dispersion / correlation from the PIT members
    C = S[1]
    lagmax = np.fmax.reduce([np.vstack([np.full((i, C.shape[1]), np.nan), C[: C.shape[0] - i]]) if i else C for i in range(52)])

    def share(cond, valid):
        v = mem & valid
        return np.where(v.sum(1) >= 20, (cond & v).sum(1) / np.maximum(v.sum(1), 1), np.nan)

    B = pd.DataFrame(index=weeks.d)
    B["breadth50"] = share(C > S[50], ~np.isnan(S[50]))
    B["breadth140"] = share(F["sma20"] > S[140], ~np.isnan(S[140]))
    B["breadth200"] = share(C > S[200], ~np.isnan(S[200]))
    B["newhigh_share"] = share(C / lagmax >= 0.98, ~np.isnan(lagmax))
    with np.errstate(all="ignore"):
        B["dispersion"] = np.nanstd(np.where(mem, F["ret20"], np.nan), axis=1)
        B["avg_stock_vol"] = np.nanmean(np.where(mem, F["vol20"], np.nan), axis=1)
    # short-term breadth: washed-out or overextended members
    lagmin = np.fmin.reduce([np.vstack([np.full((i, C.shape[1]), np.nan), C[: C.shape[0] - i]]) if i else C for i in range(52)])
    B["breadth20"] = share(C > F["sma20"], ~np.isnan(F["sma20"]))
    B["adv_share5"] = share(F["ret5"] > 0, ~np.isnan(F["ret5"]))
    B["adv_share20"] = share(F["ret20"] > 0, ~np.isnan(F["ret20"]))
    B["newlow_share"] = share(C / lagmin <= 1.02, ~np.isnan(lagmin))
    B["oversold_share"] = share(F["price_dev"] < -0.05, ~np.isnan(F["price_dev"]))
    with np.errstate(all="ignore"):
        B["med_ret5"] = np.nanmedian(np.where(mem, F["ret5"], np.nan), axis=1)
        B["med_dd60"] = np.nanmedian(np.where(mem, F["dd60"], np.nan), axis=1)
    B = B.reindex(qw.d)
    for k in B:
        X[k] = B[k].to_numpy()
    X["breadth50_chg4w"] = X["breadth50"] - X["breadth50"].shift(4)
    X["corr_proxy"] = X["vol20"] / X["avg_stock_vol"]
    # --- cross-asset
    etf = {s: _mkt(s) for s in ("TLT", "IWM", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")}
    spy = load_daily("SPY").set_index("date")["close"]
    r60 = {s: p / p.shift(60) - 1 for s, p in etf.items()}
    X["spy_rel60"] = at(c / c.shift(60) - 1) - at(spy / spy.shift(60) - 1)
    X["iwm_rel60"] = at(r60["IWM"]) - at(spy / spy.shift(60) - 1)
    X["def_cyc60"] = np.mean([at(r60[s]) for s in ("XLP", "XLU", "XLV")], axis=0) - np.mean([at(r60[s]) for s in ("XLY", "XLK", "XLI", "XLF")], axis=0)
    X["sector_breadth"] = np.mean([at((etf[s] > etf[s].rolling(100).mean()).astype(float).where(etf[s].rolling(100).mean().notna()))
                                   for s in ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")], axis=0)
    X["tlt_ret20"], X["tlt_ret60"] = at(etf["TLT"] / etf["TLT"].shift(20) - 1), at(r60["TLT"])  # NaN before 2002
    # --- targets (open-to-open, starting at the next open)
    X["R1"] = qw.R.to_numpy()
    for h in (4, 13):
        X[f"R{h}"] = (1 + qw.R).rolling(h).apply(np.prod, raw=True).shift(-(h - 1)).to_numpy() - 1
        X[f"t{h}"] = qw.e_next.shift(-(h - 1)).to_numpy()  # when the h-week label is complete
    X["t1"] = qw.e_next.to_numpy()
    return X.reset_index()


FEATURE_SETS = {
    "price7": ["ret5", "ret20", "sma_spread", "price_dev", "sma60_slope", "vol20", "dd60"],
    "vix": ["vix", "vix_rel_1y", "vix_chg20", "vrp", "vol20"],
    "breadth": ["breadth50", "breadth140", "breadth200", "newhigh_share", "dispersion", "corr_proxy", "breadth50_chg4w"],
    "breadth_st": ["breadth20", "adv_share5", "adv_share20", "newlow_share", "oversold_share", "med_ret5", "med_dd60"],
    "cross": ["spy_rel60", "iwm_rel60", "def_cyc60", "sector_breadth"],
    "trend": ["slow_up", "fast_up", "ret60", "ret120", "ret250", "dist140", "spread140", "dd250", "vol60"],
    "late": ["vix_term", "tlt_ret20", "tlt_ret60"],  # history starts in 2002: needs imputation or NaN-tolerant models
}
