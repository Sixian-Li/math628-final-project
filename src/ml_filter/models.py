"""Model specifications and annual expanding-window walk-forward of the first design (Section 5.1 of the report)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TRAIN_START_YEAR = 2000
VOL_QUANTILE = 0.75
THRESHOLD = 0.50


@dataclass(frozen=True)
class ModelSpec:
    family: str  # logit | rf | gb
    params: tuple[tuple[str, float], ...]

    @property
    def label(self) -> str:
        return self.family + "(" + ", ".join(f"{k}={v}" for k, v in self.params) + ")"

    def build(self):
        p = dict(self.params)
        if self.family == "logit":
            return make_pipeline(
                StandardScaler(),
                LogisticRegression(C=p["C"], solver="lbfgs", max_iter=1000, class_weight=None),
            )
        if self.family == "rf":
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=int(p["max_depth"]),
                min_samples_leaf=int(p["min_samples_leaf"]),
                max_features="sqrt",
                criterion="gini",
                bootstrap=True,
                class_weight=None,
                random_state=42,
                n_jobs=-1,
            )
        if self.family == "gb":
            return GradientBoostingClassifier(
                n_estimators=int(p["n_estimators"]),
                max_depth=int(p["max_depth"]),
                learning_rate=0.05,
                min_samples_leaf=20,
                loss="log_loss",
                random_state=42,
            )
        raise ValueError(self.family)


# Candidate grids, listed from most to least regularized / simplest so that exact
# Brier ties resolve to the earlier entry.
LOGIT_GRID = [ModelSpec("logit", (("C", c),)) for c in (0.01, 0.1, 1.0)]
RF_GRID = [
    ModelSpec("rf", (("max_depth", d), ("min_samples_leaf", m)))
    for d in (2, 3)
    for m in (40, 20)
]
GB_GRID = [
    ModelSpec("gb", (("n_estimators", n), ("max_depth", d)))
    for d in (1, 2)
    for n in (50, 100)
]


def training_mask(weeks: pd.DataFrame, cand: np.ndarray, year: int) -> np.ndarray:
    """Candidate weeks whose labels are fully observed before the first decision of ``year``."""
    first_decision = weeks.loc[weeks["year"] == year, "d"].min()
    return cand & (weeks["year"].to_numpy() >= TRAIN_START_YEAR) & (weeks["e_next"] <= first_decision).to_numpy()


@dataclass
class WalkForward:
    proba: pd.Series  # predicted P(up) for candidate weeks in the evaluation years
    base_rate: pd.Series  # training up-share used as the constant forecast
    vol_q: pd.Series  # training 75th percentile of vol20 used by the volatility filter
    train_sizes: dict[int, int]
    last_model: object


def walk_forward(
    weeks: pd.DataFrame,
    cand: np.ndarray,
    years: list[int],
    features: list[str],
    spec: ModelSpec | None,
) -> WalkForward:
    """Refit once per year on all completed candidate weeks; predict that year's candidates."""
    complete = weeks[features].notna().all(axis=1).to_numpy()
    cand = cand.astype(bool) & complete
    proba = pd.Series(np.nan, index=weeks.index)
    base_rate = pd.Series(np.nan, index=weeks.index)
    vol_q = pd.Series(np.nan, index=weeks.index)
    sizes: dict[int, int] = {}
    model = None
    for year in years:
        tr = training_mask(weeks, cand, year)
        te = cand & (weeks["year"].to_numpy() == year)
        sizes[year] = int(tr.sum())
        X, y = weeks.loc[tr, features].to_numpy(), weeks.loc[tr, "y"].to_numpy()
        base_rate[te] = y.mean()
        vol_q[te] = np.quantile(weeks.loc[tr, "vol20"].to_numpy(), VOL_QUANTILE)  # linear interpolation
        if spec is not None and te.any():
            model = spec.build().fit(X, y)
            proba[te] = model.predict_proba(weeks.loc[te, features].to_numpy())[:, 1]
    return WalkForward(proba, base_rate, vol_q, sizes, model)


def fit_predict(weeks: pd.DataFrame, train: np.ndarray, test: np.ndarray, features: list[str], spec: ModelSpec) -> np.ndarray:
    model = spec.build().fit(weeks.loc[train, features].to_numpy(), weeks.loc[train, "y"].to_numpy())
    return model.predict_proba(weeks.loc[test, features].to_numpy())[:, 1]


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def select(weeks: pd.DataFrame, cand: np.ndarray, years: list[int], features: list[str], grid: list[ModelSpec]):
    rows, best, best_score = [], None, np.inf
    for spec in grid:
        wf = walk_forward(weeks, cand, years, features, spec)
        m = wf.proba.notna()
        score = brier(wf.proba[m].to_numpy(), weeks.loc[m, "y"].to_numpy())
        const = brier(wf.base_rate[m].to_numpy(), weeks.loc[m, "y"].to_numpy())
        rows.append({"model": spec.label, "family": spec.family, "val_brier": score, "val_const_brier": const, "n_val": int(m.sum())})
        if score < best_score - 1e-12:
            best, best_score = spec, score
    return best, pd.DataFrame(rows)
