import numpy as np
import pandas as pd

import pytest

from src.normalization import adjusted_price_matrix, normalize_price_records, weekly_adjusted_prices, weekly_returns_from_prices
from src.sample_policy import common_sample, pairwise_overlap


def test_common_sample_is_fixed_for_all_assets():
    dates = pd.date_range("2020-01-03", periods=5, freq="W-FRI")
    returns = pd.DataFrame({
        "A": [0.01, 0.02, 0.03, 0.04, 0.05],
        "B": [np.nan, 0.01, 0.02, 0.03, 0.04],
        "C": [0.02, 0.01, 0.00, 0.01, np.nan],
    }, index=dates)
    sample = common_sample(returns)
    assert sample.returns.index.tolist() == dates[1:4].tolist()
    assert sample.weeks == 3


def test_pairwise_overlap_is_pair_specific_and_correct():
    dates = pd.date_range("2020-01-03", periods=4, freq="W-FRI")
    returns = pd.DataFrame({"A": [1, 2, np.nan, 4], "B": [1, np.nan, 3, 4]}, index=dates)
    pair = pairwise_overlap(returns, "A", "B")
    assert pair.index.tolist() == [dates[0], dates[3]]


def test_missing_week_is_not_forward_filled():
    raw = pd.DataFrame({
        "date": ["2026-01-02", "2026-01-16"],
        "close": [100.0, 121.0], "adj": [100.0, 121.0],
    })
    normalized = normalize_price_records(
        raw, ticker="ABC", market="US", source="synthetic",
        date_column="date", raw_close_column="close", adjusted_close_column="adj",
    )
    weekly = weekly_adjusted_prices(normalized)
    returns, _ = weekly_returns_from_prices(weekly)
    assert pd.isna(weekly.loc[pd.Timestamp("2026-01-09"), "ABC"])
    assert pd.isna(returns.loc[pd.Timestamp("2026-01-16"), "ABC"])


def test_normalized_adjusted_price_schema():
    raw = pd.DataFrame({"d": ["2026-01-02"], "c": [10.0], "a": [9.5]})
    result = normalize_price_records(
        raw, ticker="abc", market="us", source="test",
        date_column="d", raw_close_column="c", adjusted_close_column="a",
    )
    assert list(result.columns) == ["date", "ticker", "market", "source", "raw_close", "adjusted_close"]
    assert result.loc[0, "ticker"] == "ABC"
    assert result.loc[0, "adjusted_close"] == 9.5


def test_mixed_market_analysis_is_rejected():
    us = normalize_price_records(
        pd.DataFrame({"d": ["2026-01-02"], "c": [10], "a": [10]}),
        ticker="VOO", market="US", source="test", date_column="d",
        raw_close_column="c", adjusted_close_column="a",
    )
    tw = normalize_price_records(
        pd.DataFrame({"d": ["2026-01-02"], "c": [10], "a": [10]}),
        ticker="0050", market="TW", source="test", date_column="d",
        raw_close_column="c", adjusted_close_column="a",
    )
    with pytest.raises(ValueError, match="Mixed-market analysis is not supported in v1"):
        adjusted_price_matrix(pd.concat([us, tw], ignore_index=True))
