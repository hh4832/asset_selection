from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


NORMALIZED_PRICE_COLUMNS = [
    "date", "ticker", "market", "source", "raw_close", "adjusted_close",
]


@dataclass
class DataQualityResult:
    data: pd.DataFrame
    issues: pd.DataFrame

    @property
    def has_fatal(self) -> bool:
        return bool(not self.issues.empty and self.issues["severity"].eq("fatal").any())


def normalize_price_records(
    frame: pd.DataFrame,
    *,
    ticker: str,
    market: str,
    source: str,
    date_column: str,
    raw_close_column: str,
    adjusted_close_column: str,
) -> pd.DataFrame:
    """Convert provider output to the immutable normalized price schema."""
    required = {date_column, raw_close_column, adjusted_close_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing normalized price input columns: {sorted(missing)}")
    out = pd.DataFrame({
        "date": pd.to_datetime(frame[date_column], errors="coerce", utc=True)
        .dt.tz_convert(None).dt.normalize(),
        "ticker": str(ticker).strip().upper(),
        "market": str(market).strip().upper(),
        "source": source,
        "raw_close": pd.to_numeric(frame[raw_close_column], errors="coerce"),
        "adjusted_close": pd.to_numeric(frame[adjusted_close_column], errors="coerce"),
    })
    return out[NORMALIZED_PRICE_COLUMNS]


def validate_normalized_prices(frame: pd.DataFrame) -> DataQualityResult:
    missing_columns = set(NORMALIZED_PRICE_COLUMNS) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Normalized schema missing columns: {sorted(missing_columns)}")
    data = frame[NORMALIZED_PRICE_COLUMNS].copy()
    issues: list[dict] = []
    for ticker, group in data.groupby("ticker", sort=True, dropna=False):
        label = str(ticker)
        invalid_dates = group["date"].isna()
        if invalid_dates.any():
            issues.append({"ticker": label, "check": "invalid_date", "count": int(invalid_dates.sum()), "severity": "fatal"})
        duplicates = group["date"].duplicated(keep=False) & group["date"].notna()
        if duplicates.any():
            issues.append({"ticker": label, "check": "duplicate_dates", "count": int(duplicates.sum()), "severity": "fatal"})
        valid_dates = group.loc[group["date"].notna(), "date"]
        if not valid_dates.is_monotonic_increasing:
            issues.append({"ticker": label, "check": "non_monotonic_dates", "count": 1, "severity": "fatal"})
        missing_adjusted = group["adjusted_close"].isna()
        if missing_adjusted.any():
            issues.append({"ticker": label, "check": "missing_adjusted_prices", "count": int(missing_adjusted.sum()), "severity": "warning"})
        non_positive = group["adjusted_close"].notna() & group["adjusted_close"].le(0)
        if non_positive.any():
            issues.append({"ticker": label, "check": "non_positive_adjusted_prices", "count": int(non_positive.sum()), "severity": "fatal"})
    data = data.sort_values(["ticker", "date"]).reset_index(drop=True)
    issue_frame = pd.DataFrame(issues, columns=["ticker", "check", "count", "severity"])
    return DataQualityResult(data=data, issues=issue_frame)


def adjusted_price_matrix(frame: pd.DataFrame, *, market: str | None = None) -> pd.DataFrame:
    quality = validate_normalized_prices(frame)
    if quality.has_fatal:
        raise ValueError(f"Fatal price-data quality errors: {quality.issues.to_dict('records')}")
    data = quality.data
    markets = sorted(data["market"].dropna().unique())
    if market is not None:
        market = market.upper()
        data = data.loc[data["market"].eq(market)]
        markets = sorted(data["market"].dropna().unique())
    if len(markets) > 1:
        raise ValueError("Mixed-market analysis is not supported in v1.")
    matrix = data.pivot(index="date", columns="ticker", values="adjusted_close").sort_index()
    matrix.columns.name = None
    return matrix


def weekly_adjusted_prices(frame: pd.DataFrame, *, market: str | None = None) -> pd.DataFrame:
    """Use the last observed adjusted close in each week; never forward fill."""
    daily = adjusted_price_matrix(frame, market=market)
    if daily.empty:
        return daily
    return daily.resample("W-FRI").last()


def weekly_returns_from_prices(prices: pd.DataFrame, *, impossible_return: float = 5.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    impossible = returns.abs().gt(impossible_return)
    issues = []
    for ticker in returns.columns:
        count = int(impossible[ticker].sum())
        if count:
            issues.append({"ticker": ticker, "check": "impossible_return", "count": count, "severity": "fatal"})
    return returns, pd.DataFrame(issues, columns=["ticker", "check", "count", "severity"])
