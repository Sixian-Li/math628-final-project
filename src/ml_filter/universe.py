"""Point-in-time Nasdaq-100 stock panel on the weekly decision grid.

Weeks follow the same convention as ``data.weekly_table`` but on the XNYS calendar:
d = last session of a W-FRI week, e = next session, e_next = first session of the
following week. Each stock's indicators use its own observed rows. Membership is the
InIndex interval status observed one session before d. Missing opens/closes are filled
with the last known close (settlement proxy for exits only; entries need a real bar).
"""

from __future__ import annotations

import io
import pickle
import zipfile

import numpy as np
import pandas as pd

from .data import FEATURES, PROJECT_ROOT, add_indicators

RAW = PROJECT_ROOT / "data" / "raw" / "nasdaq100"
ZIP = RAW / "nasdaq100_split_div_adjusted_20260805.zip"
CACHE = PROJECT_ROOT / "data" / "processed" / "nasdaq100_panel.pkl"
DATA_END = pd.Timestamp("2026-08-04")
START = pd.Timestamp("1995-01-01")
SMA_COLS = [f"sma{k}" for k in (10, 20, 30, 40, 60, 120)]


def _calendar() -> pd.DatetimeIndex:
    cal = pd.to_datetime(pd.read_csv(RAW / "XNYS.csv")["date"])
    return pd.DatetimeIndex(cal[(cal >= START) & (cal <= DATA_END)]).sort_values()


def _weeks(cal: pd.DatetimeIndex) -> pd.DataFrame:
    s = pd.Series(np.arange(len(cal)), index=cal)
    last = s.groupby(cal.to_period("W-FRI")).max().to_numpy()
    i_d = last[:-2]
    w = pd.DataFrame({"i_d": i_d, "i_e": i_d + 1, "i_e_next": last[1:-1] + 1})
    w["d"], w["e"], w["e_next"] = cal[w.i_d], cal[w.i_e], cal[w.i_e_next]
    w["year"] = w["e"].dt.year
    return w


def build_panel(force: bool = False) -> dict:
    if CACHE.exists() and not force:
        return pickle.loads(CACHE.read_bytes())
    cal = _calendar()
    weeks = _weeks(cal)
    master = pd.read_csv(RAW / "security_master.csv")
    intervals = pd.read_csv(RAW / "membership_intervals.csv", parse_dates=["effective_start", "effective_end"])
    files = list(master["source_file"])
    T, J, K = len(cal), len(files), len(weeks)
    O = np.full((T, J), np.nan)
    C = np.full((T, J), np.nan)
    cols = FEATURES + SMA_COLS
    feat = {c: np.full((K, J), np.nan) for c in cols}
    with zipfile.ZipFile(ZIP) as z:
        names = {n.split("/")[-1]: n for n in z.namelist()}
        for j, f in enumerate(files):
            raw = pd.read_csv(io.TextIOWrapper(z.open(names[f]), encoding="utf-8-sig"), usecols=["Date", "Open", "Close"])
            raw.columns = ["date", "open", "close"]
            raw["date"] = pd.to_datetime(raw["date"])
            raw = raw[(raw.date <= DATA_END) & (raw.open > 0) & (raw.close > 0)].drop_duplicates("date").sort_values("date")
            ind = add_indicators(raw).set_index("date").reindex(cal)
            O[:, j], C[:, j] = ind["open"].to_numpy(), ind["close"].to_numpy()
            at_d = ind.iloc[weeks.i_d.to_numpy()]
            for c in cols:
                feat[c][:, j] = at_d[c].to_numpy()
    real = ~np.isnan(O) & ~np.isnan(C)
    Cf = pd.DataFrame(C).ffill().to_numpy()
    Of = np.where(np.isnan(O), Cf, O)
    # lagged membership at decision close
    known = cal[weeks.i_d.to_numpy() - 1]
    member = np.zeros((K, J), bool)
    col = {f: j for j, f in enumerate(files)}
    for r in intervals.itertuples():
        j = col[r.source_file]
        member[:, j] |= (known >= r.effective_start) & (known <= r.effective_end)
    i_d, i_e, i_n = weeks.i_d.to_numpy(), weeks.i_e.to_numpy(), weeks.i_e_next.to_numpy()
    tradable = member & real[i_d] & real[i_e]
    R = Of[i_n] / Of[i_e] - 1
    panel = dict(cal=cal, weeks=weeks, files=files, tickers=list(master["display_ticker"]),
                 Of=Of, Cf=Cf, member=member, tradable=tradable, R=R, feat=feat)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_bytes(pickle.dumps(panel, protocol=5))
    return panel


def load_qqq(cal: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    q = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "QQQ.csv", parse_dates=["date"]).set_index("date").reindex(cal)
    c = q["close"].ffill().to_numpy()
    o = np.where(q["open"].isna(), c, q["open"].to_numpy())
    return o[:, None], c[:, None]
