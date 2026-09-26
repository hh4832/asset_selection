from __future__ import annotations

import numpy as np
import pandas as pd


WEEKS_PER_YEAR = 52
DEFAULT_ANNUAL_RF = 0.04


def weekly_risk_free_rate(annual_rate: float = DEFAULT_ANNUAL_RF) -> float:
    return (1.0 + annual_rate) ** (1.0 / WEEKS_PER_YEAR) - 1.0


def _clean(returns: pd.Series) -> pd.Series:
    return pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def cagr(returns: pd.Series) -> float:
    values = _clean(returns)
    if values.empty or (values <= -1).any():
        return np.nan
    gross = float((1.0 + values).prod())
    years = len(values) / WEEKS_PER_YEAR
    return gross ** (1.0 / years) - 1.0 if gross > 0 and years > 0 else np.nan


def annualized_volatility(returns: pd.Series) -> float:
    values = _clean(returns)
    return float(values.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR)) if len(values) >= 2 else np.nan


def downside_deviation(returns: pd.Series, target: float = 0.0) -> float:
    values = _clean(returns)
    if values.empty:
        return np.nan
    downside = np.minimum(values - target, 0.0)
    return float(np.sqrt(np.mean(np.square(downside))) * np.sqrt(WEEKS_PER_YEAR))


def sharpe(returns: pd.Series, annual_rf: float = DEFAULT_ANNUAL_RF) -> float:
    values = _clean(returns)
    if len(values) < 2:
        return np.nan
    excess = values - weekly_risk_free_rate(annual_rf)
    denominator = excess.std(ddof=1)
    return float(excess.mean() / denominator * np.sqrt(WEEKS_PER_YEAR)) if denominator > 0 else np.nan


def sortino(returns: pd.Series, annual_rf: float = DEFAULT_ANNUAL_RF) -> float:
    values = _clean(returns)
    if len(values) < 2:
        return np.nan
    denominator = downside_deviation(values)
    numerator = (values.mean() - weekly_risk_free_rate(annual_rf)) * WEEKS_PER_YEAR
    return float(numerator / denominator) if denominator and denominator > 0 else np.nan


def max_drawdown(returns: pd.Series) -> float:
    values = _clean(returns)
    if values.empty or (values <= -1).any():
        return np.nan
    wealth = (1.0 + values).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def calmar(returns: pd.Series) -> float:
    annual_return = cagr(returns)
    drawdown = max_drawdown(returns)
    return float(annual_return / abs(drawdown)) if pd.notna(annual_return) and pd.notna(drawdown) and drawdown != 0 else np.nan


def metric_summary(returns: pd.Series, annual_rf: float = DEFAULT_ANNUAL_RF) -> dict[str, float]:
    return {
        "CAGR": cagr(returns),
        "Annualized volatility": annualized_volatility(returns),
        "Sharpe": sharpe(returns, annual_rf),
        "Sortino": sortino(returns, annual_rf),
        "Max Drawdown": max_drawdown(returns),
        "Downside deviation": downside_deviation(returns),
        "Calmar": calmar(returns),
    }


def metrics_table(returns: pd.DataFrame, annual_rf: float = DEFAULT_ANNUAL_RF) -> pd.DataFrame:
    rows = [{"asset": asset, **metric_summary(returns[asset], annual_rf)} for asset in returns.columns]
    return pd.DataFrame(rows)


def rolling_sharpe(returns: pd.Series, window: int = 52, annual_rf: float = DEFAULT_ANNUAL_RF) -> pd.Series:
    values = pd.to_numeric(returns, errors="coerce")
    excess = values - weekly_risk_free_rate(annual_rf)
    result = excess.rolling(window, min_periods=window).mean() / excess.rolling(window, min_periods=window).std(ddof=1)
    return (result * np.sqrt(WEEKS_PER_YEAR)).replace([np.inf, -np.inf], np.nan)


def ewma_sharpe(returns: pd.Series, halflife: int = 52, annual_rf: float = DEFAULT_ANNUAL_RF) -> pd.Series:
    values = pd.to_numeric(returns, errors="coerce") - weekly_risk_free_rate(annual_rf)
    mean = values.ewm(halflife=halflife, adjust=False, min_periods=12).mean()
    variance = values.ewm(halflife=halflife, adjust=False, min_periods=12).var(bias=False)
    return (mean / np.sqrt(variance) * np.sqrt(WEEKS_PER_YEAR)).replace([np.inf, -np.inf], np.nan)

