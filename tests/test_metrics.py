import numpy as np
import pandas as pd
import pytest

from src.metrics import annualized_volatility, cagr, max_drawdown, metric_summary


def test_metric_summary_uses_interpretable_raw_metrics():
    returns = pd.Series([0.01] * 52)
    summary = metric_summary(returns, annual_rf=0.0)
    assert summary["CAGR"] == pytest.approx((1.01 ** 52) - 1)
    assert summary["Annualized volatility"] == pytest.approx(0.0)
    assert summary["Max Drawdown"] == pytest.approx(0.0)


def test_drawdown_and_invalid_total_loss():
    assert max_drawdown(pd.Series([0.10, -0.20, 0.05])) == pytest.approx(-0.20)
    assert np.isnan(cagr(pd.Series([0.01, -1.0])))
    assert np.isnan(annualized_volatility(pd.Series([0.01])))

