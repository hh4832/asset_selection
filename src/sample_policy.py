from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CommonSample:
    returns: pd.DataFrame
    analysis_start: pd.Timestamp | None
    analysis_end: pd.Timestamp | None
    weeks: int


def restrict_years(returns: pd.DataFrame, years: int) -> pd.DataFrame:
    if returns.empty:
        return returns.copy()
    end = pd.Timestamp(returns.index.max())
    start = end - pd.DateOffset(years=int(years))
    return returns.loc[returns.index >= start].copy()


def common_sample(returns: pd.DataFrame, assets: list[str] | None = None) -> CommonSample:
    selected = list(assets or returns.columns)
    missing = sorted(set(selected) - set(returns.columns))
    if missing:
        raise ValueError(f"Assets unavailable for common sample: {missing}")
    fixed = returns[selected].replace([np.inf, -np.inf], np.nan).dropna(how="any")
    return CommonSample(
        returns=fixed,
        analysis_start=pd.Timestamp(fixed.index.min()) if not fixed.empty else None,
        analysis_end=pd.Timestamp(fixed.index.max()) if not fixed.empty else None,
        weeks=len(fixed),
    )


def pairwise_overlap(returns: pd.DataFrame, asset_a: str, asset_b: str) -> pd.DataFrame:
    if asset_a not in returns or asset_b not in returns:
        raise ValueError(f"Pair unavailable: {asset_a}, {asset_b}")
    return returns[[asset_a, asset_b]].replace([np.inf, -np.inf], np.nan).dropna(how="any")


def history_cohort(history_years: float) -> str:
    if pd.isna(history_years) or history_years < 3:
        return "Insufficient history"
    if history_years < 5:
        return "Short history"
    if history_years < 10:
        return "Medium history"
    return "Long history"


def coverage_table(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset in returns.columns:
        values = returns[asset].dropna()
        if values.empty:
            first = last = pd.NaT
            years = np.nan
        else:
            first, last = pd.Timestamp(values.index.min()), pd.Timestamp(values.index.max())
            years = (last - first).days / 365.25
        rows.append({
            "asset": asset,
            "first_valid_date": first,
            "last_valid_date": last,
            "history_years": years,
            "non_na_weeks": int(len(values)),
            "cohort": history_cohort(years),
        })
    return pd.DataFrame(rows)

