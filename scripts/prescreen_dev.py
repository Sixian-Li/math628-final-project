"""Pre-screen of candidate ideas on the DEVELOPMENT period only (<= 2018); 2019-2025 is not touched.

This is the exploratory look that motivated scripts/run_stock_rank.py and scripts/run_qqq_state.py:
  1. stock rankings inside SMA20/140 candidates (momentum, 52-week high, volatility, reversal)
  2. QQQ weekly returns by fast/slow trend state
  3. VIX / breadth / dispersion features against the forward QQQ return in base-rule weeks
  4. simple QQQ filters on holding CAGR
Because the feature families were chosen after seeing these numbers, 2005-2018 is not a clean test for them.
Usage: .venv/bin/python scripts/prescreen_dev.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ml_filter.data import load_daily, weekly_table  # noqa: E402
from ml_filter.meta import sma_windows  # noqa: E402
from ml_filter.portfolio import capped_equal, weekly_ledger  # noqa: E402
from ml_filter.universe import build_panel  # noqa: E402

pd.set_option("display.width", 220)
DEV = (2005, 2018)
P = build_panel()
weeks, F, Of = P["weeks"], P["feat"], P["Of"]
nW, nJ = P["R"].shape
S = sma_windows(P, [1, 20, 50, 140, 200])
C = S[1]  # close at d_t
cand = P["tradable"] & ~np.isnan(S[140]) & (F["sma20"] > S[140]) & (weeks.year >= 2000).to_numpy()[:, None]
G = Of[weeks.i_e_next.to_numpy()] / Of[weeks.i_e.to_numpy()]


def stats(sel, y0, y1, cost=5):
    rows = np.flatnonzero(((weeks.year >= y0) & (weeks.year <= y1) & (weeks.e_next <= pd.Timestamp(f"{y1}-12-31"))).to_numpy())
    W = capped_equal(sel[rows])
    led = weekly_ledger(W, G[rows], cost)
    eq = led["eq_end"]
    wk = weeks.iloc[rows]
    yrs = (wk.e_next.iloc[-1] - wk.e.iloc[0]).days / 365.25
    r = np.diff(np.r_[1.0, eq]) / np.r_[1.0, eq[:-1]]
    return dict(cagr=(eq[-1] ** (1 / yrs) - 1) * 100, sharpe=r.mean() / r.std(ddof=1) * np.sqrt(52),
                maxdd=(eq / np.maximum.accumulate(np.maximum(eq, 1)) - 1).min() * 100,
                turn=(led["turnover"].sum() + led["final_turn"]) / yrs, names=sel[rows].sum(1).mean())


def lag(X, k):
    out = np.full_like(X, np.nan)
    out[k:] = X[:-k]
    return out


def topk_hold(score, k=20, every=4, buffer=None):
    """Re-rank every `every` weeks among candidates; between rebalances keep names while they stay candidates."""
    sel = np.zeros((nW, nJ), bool)
    cur = np.zeros(nJ, bool)
    for t in range(nW):
        ok = cand[t] & ~np.isnan(score[t])
        if t % every == 0:
            s = np.where(ok, score[t], -np.inf)
            order = np.argsort(-s, kind="stable")
            rank = np.empty(nJ, int)
            rank[order] = np.arange(nJ)
            keep = cur & ok & (rank < (buffer or 0))
            new = keep.copy()
            for j in order:
                if new.sum() >= k or not ok[j]:
                    break
                new[j] = True
            cur = new
        cur = cur & cand[t]
        sel[t] = cur
    return sel


print("=========== Screen 1: stock ranking inside SMA20/140 candidates, DEV 2005-2018, 5 bps")
scores = {
    "mom12_1": lag(C, 4) / lag(C, 52) - 1,
    "mom6_1": lag(C, 4) / lag(C, 26) - 1,
    "mom12_0": C / lag(C, 52) - 1,
    "hi52": C / np.fmax.reduce([lag(C, i) if i else C for i in range(52)]),
    "ret20(old F1)": F["ret20"],
    "lowvol": -F["vol20"],
    "highvol": F["vol20"],
    "reversal4w": -(C / lag(C, 4) - 1),
}
rng = np.random.default_rng(1)
res = [{"rule": "S0 all candidates", **stats(cand, *DEV)}]
plc = {}
for every in (1, 4):
    r = np.array([stats(topk_hold(np.where(cand, rng.random(cand.shape), np.nan), 20, every), *DEV)["cagr"] for _ in range(60)])
    plc[every] = r
    res.append({"rule": f"random20 every{every}w (median of 60)", "cagr": np.median(r)})
for name, sc in scores.items():
    for every, buf in ((1, None), (4, None), (4, 40)):
        st = stats(topk_hold(sc, 20, every, buf), *DEV)
        res.append({"rule": f"{name} top20 every{every}w" + (f" buf{buf}" if buf else ""), **st, "plc_pct": (plc[every] < st["cagr"]).mean() * 100})
print(pd.DataFrame(res).round(2).to_string(index=False))

print("\n=========== Screen 2: QQQ fast/slow states (fast SMA20/60, slow SMA20/140), weekly open-to-open R")
qd = load_daily("QQQ")
qw = weekly_table(qd)
fast, slow = qw.sma20 > qw.sma60, qw.sma20 > qw.sma140
qw["state"] = np.select([fast & slow, ~fast & slow, fast & ~slow], ["Bull (both up)", "Correction (slow up, fast down)", "Rebound (fast up, slow down)"], "Bear (both down)")
for nm, (a, b) in {"DEV 2000-2018": (2000, 2018)}.items():
    g = qw[qw.year.between(a, b) & qw.sma140.notna()].groupby("state").R
    print(nm)
    print(pd.DataFrame({"weeks": g.size(), "mean_bps": g.mean() * 1e4, "sd_bps": g.std() * 1e4, "up_pct": g.apply(lambda x: (x > 0).mean() * 100),
                        "ann_ret_pct": ((1 + g.mean()) ** 52 - 1) * 100, "t": g.mean() / g.std() * np.sqrt(g.size())}).round(2).to_string())

print("\n=========== Screen 3: new-information features vs forward QQQ return, base-on weeks (SMA20>SMA140), DEV 2000-2018")
vix = pd.read_csv(ROOT / "data" / "raw" / "market" / "VIX.csv", encoding="utf-8-sig", parse_dates=["Date"]).set_index("Date")["Close"]
mem = P["tradable"]
def share(cond, valid):
    v = mem & valid
    return np.where(v.sum(1) > 0, (cond & v).sum(1) / np.maximum(v.sum(1), 1), np.nan)
br140 = pd.Series(share(F["sma20"] > S[140], ~np.isnan(S[140])), index=weeks.d)
br50 = pd.Series(share(C > S[50], ~np.isnan(S[50])), index=weeks.d)
br200 = pd.Series(share(C > S[200], ~np.isnan(S[200])), index=weeks.d)
disp = pd.Series(np.nanstd(np.where(mem, F["ret20"], np.nan), axis=1), index=weeks.d)
avgvol = pd.Series(np.nanmean(np.where(mem, F["vol20"], np.nan), axis=1), index=weeks.d)
hi = C / np.fmax.reduce([lag(C, i) if i else C for i in range(52)])
newhi = pd.Series(share(hi >= 0.98, ~np.isnan(hi)), index=weeks.d)
X = pd.DataFrame(index=qw.d)
X["vix"] = vix.reindex(qw.d).to_numpy()
X["vix_chg4w"] = X.vix / X.vix.shift(4) - 1
X["vix_rel_1y"] = X.vix / X.vix.rolling(52).mean()
X["vrp"] = X.vix ** 2 - (qw.vol20.to_numpy() * np.sqrt(252) * 100) ** 2
X["rv_qqq"] = qw.vol20.to_numpy()
for nm, s_ in {"breadth140": br140, "breadth50": br50, "breadth200": br200, "dispersion": disp, "avg_stock_vol": avgvol, "newhigh_share": newhi}.items():
    X[nm] = s_.reindex(qw.d).to_numpy()
X["breadth50_chg4w"] = X.breadth50 - X.breadth50.shift(4)
X["corr_proxy"] = X.rv_qqq / X.avg_stock_vol
X["R1"] = qw.R.to_numpy()
X["R4"] = (1 + qw.R).rolling(4).apply(np.prod, raw=True).shift(-3).to_numpy() - 1
X["year"] = qw.year.to_numpy()
on = (qw.sma20 > qw.sma140).to_numpy() & qw.sma140.notna().to_numpy()
D = X[on & X.year.between(2000, 2018)]
out = []
for c in [c for c in X.columns if c not in ("R1", "R4", "year")]:
    d = D.dropna(subset=[c, "R4"])
    q = pd.qcut(d[c], 3, labels=["lo", "mid", "hi"])
    m = d.groupby(q, observed=True).R1.mean() * 1e4
    out.append({"feature": c, "n": len(d), "spearman_R1": d[c].corr(d.R1, method="spearman"), "spearman_R4": d[c].corr(d.R4, method="spearman"),
                "R1_bps_lo": m["lo"], "R1_bps_mid": m["mid"], "R1_bps_hi": m["hi"]})
print(f"base-on weeks: {len(D)}, mean R1 = {D.R1.mean() * 1e4:.1f} bps")
print(pd.DataFrame(out).round(3).to_string(index=False))
# same split by sub-period to see sign stability
print("\nsign stability of spearman_R4 across sub-periods (base-on weeks)")
rows = []
for c in [c for c in X.columns if c not in ("R1", "R4", "year")]:
    rows.append({"feature": c, **{f"{a}-{b}": X[on & X.year.between(a, b)].dropna(subset=[c, "R4"])[[c, "R4"]].corr(method="spearman").iloc[0, 1]
                                  for a, b in ((2000, 2006), (2007, 2012), (2013, 2018))}})
print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n=========== Screen 4: QQQ holding CAGR on DEV 2005-2018 (5 bps), base = SMA20/140")
from ml_filter.backtest import equity_curve, metrics  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

FE = ["vix", "vrp", "corr_proxy", "dispersion", "breadth50", "newhigh_share", "rv_qqq", "breadth140"]
Xf = X.reset_index(drop=True)
qw2 = qw.reset_index(drop=True)
ok = Xf[FE + ["R4"]].notna().all(axis=1).to_numpy()
fd = qw2.groupby("year")["d"].min().to_dict()
lab_done = qw2.e_next.shift(-3)  # the 4-week label of week t ends at e_next of week t+3
fastu = (qw2.sma20 > qw2.sma60).to_numpy()
union = on | fastu

def wf(model_fn, cand_mask, q):
    keep = np.zeros(len(qw2), bool)
    for yr in range(2005, 2019):
        tr = cand_mask & ok & (qw2.year >= 2000).to_numpy() & (lab_done <= fd[yr]).to_numpy()
        te = cand_mask & Xf[FE].notna().all(axis=1).to_numpy() & (qw2.year == yr).to_numpy()
        m = model_fn().fit(Xf.loc[tr, FE], Xf.loc[tr, "R4"])
        thr = np.quantile(m.predict(Xf.loc[tr, FE]), q)
        keep[te] = m.predict(Xf.loc[te, FE]) >= thr
    return keep

def expanding_rule(col, q, above=True):
    s = Xf[col].where(on)
    thr = s.expanding(min_periods=100).quantile(q).shift(1)
    return on & ((Xf[col] >= thr) if above else (Xf[col] <= thr)).to_numpy()

ev = qw2[(qw2.year.between(2005, 2018)) & (qw2.e_next <= pd.Timestamp("2018-12-31"))]
rules = {
    "buy_hold": np.ones(len(qw2), bool), "base 20/140": on, "SMA20/60": fastu & qw2.sma60.notna().to_numpy(), "union (slow up OR fast up)": union,
    "base & VIX >= expanding median": expanding_rule("vix", 0.5), "base & VIX >= expanding 33%": expanding_rule("vix", 1 / 3),
    "base & corr_proxy >= exp. 33%": expanding_rule("corr_proxy", 1 / 3), "base & breadth50 <= exp. 67%": expanding_rule("breadth50", 2 / 3, False),
    "base & Ridge(R4) top 2/3": wf(lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)), on, 1 / 3),
    "base & Ridge(R4) top 1/2": wf(lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)), on, 1 / 2),
    "base & GBreg(R4) top 2/3": wf(lambda: GradientBoostingRegressor(n_estimators=50, max_depth=1, learning_rate=0.05, min_samples_leaf=20, random_state=42), on, 1 / 3),
    "union & Ridge(R4) top 2/3": wf(lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0)), union, 1 / 3),
    "union & GBreg(R4) top 2/3": wf(lambda: GradientBoostingRegressor(n_estimators=50, max_depth=1, learning_rate=0.05, min_samples_leaf=20, random_state=42), union, 1 / 3),
}
rows = []
for nm, w in rules.items():
    m = metrics(equity_curve(qd, ev, w[ev.index].astype(float), 5))
    rows.append({"rule": nm, "cagr": m["cagr_pct"], "holding_cagr": m["holding_cagr_pct"], "maxdd": m["max_dd_pct"], "sharpe": m["sharpe"],
                 "exposure": m["exposure_pct"], "trades_py": m["trades_per_year"]})
print(pd.DataFrame(rows).round(2).to_string(index=False))
