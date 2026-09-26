import pandas as pd
import pytest

from src.portfolio_analysis import portfolio_addition_test, weighted_portfolio


def test_portfolio_addition_weight_math():
    dates = pd.date_range("2025-01-03", periods=3, freq="W-FRI")
    reference = pd.Series([0.10, -0.10, 0.0], index=dates)
    candidate = pd.Series([0.0, 0.10, 0.20], index=dates)
    result = weighted_portfolio(reference, candidate, 0.20)
    assert result.tolist() == pytest.approx([0.08, -0.06, 0.04])


def test_portfolio_addition_uses_one_common_window():
    dates = pd.date_range("2020-01-03", periods=30, freq="W-FRI")
    returns = pd.DataFrame({"REF": [-0.01] * 30, "A": [0.01] * 30, "B": [0.0] * 30}, index=dates)
    table = portfolio_addition_test(returns, "REF", ["A", "B"], 0.2)
    assert table["common_weeks"].eq(30).all()
    assert table["analysis_start"].nunique() == 1
    assert table["analysis_end"].nunique() == 1

