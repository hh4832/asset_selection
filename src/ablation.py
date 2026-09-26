from __future__ import annotations

import pandas as pd

from src.metrics import metric_summary


def equal_weight_portfolio(returns: pd.DataFrame) -> pd.Series:
    if returns.empty or returns.shape[1] == 0:
        return pd.Series(dtype=float)
    if returns.isna().any().any():
        raise ValueError("Ablation requires a fixed complete common sample")
    return returns.mean(axis=1)


def fixed_sample_ablation(common_returns: pd.DataFrame, *, annual_rf: float = 0.04) -> pd.DataFrame:
    """Remove each asset without changing the baseline's dates."""
    if common_returns.shape[1] < 2:
        return pd.DataFrame()
    if common_returns.isna().any().any():
        raise ValueError("Ablation requires a fixed complete common sample")
    fixed_index = common_returns.index.copy()
    baseline = metric_summary(equal_weight_portfolio(common_returns), annual_rf)
    rows = []
    for asset in common_returns.columns:
        reduced = common_returns.drop(columns=asset)
        if not reduced.index.equals(fixed_index):
            raise RuntimeError("Ablation sample window changed unexpectedly")
        metrics = metric_summary(equal_weight_portfolio(reduced), annual_rf)
        rows.append({
            "removed_asset": asset,
            "ΔSharpe": metrics["Sharpe"] - baseline["Sharpe"],
            "ΔSortino": metrics["Sortino"] - baseline["Sortino"],
            "ΔMaxDD": metrics["Max Drawdown"] - baseline["Max Drawdown"],
            "ΔVolatility": metrics["Annualized volatility"] - baseline["Annualized volatility"],
            "ΔCAGR": metrics["CAGR"] - baseline["CAGR"],
            "ΔCalmar": metrics["Calmar"] - baseline["Calmar"],
            "analysis_start": fixed_index.min(),
            "analysis_end": fixed_index.max(),
            "common_weeks": len(fixed_index),
        })
    return pd.DataFrame(rows)

