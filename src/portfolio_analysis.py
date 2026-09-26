from __future__ import annotations

import numpy as np
import pandas as pd

from src.metrics import metric_summary


def weighted_portfolio(reference: pd.Series, candidate: pd.Series, candidate_weight: float) -> pd.Series:
    if not 0 <= candidate_weight <= 1:
        raise ValueError("Candidate weight must be between 0 and 1")
    pair = pd.concat([reference.rename("reference"), candidate.rename("candidate")], axis=1).dropna()
    return (1.0 - candidate_weight) * pair["reference"] + candidate_weight * pair["candidate"]


def portfolio_addition_test(
    common_returns: pd.DataFrame,
    reference: str,
    candidates: list[str],
    candidate_weight: float,
    *,
    annual_rf: float = 0.04,
    minimum_stress_observations: int = 20,
    thresholds: tuple[float, ...] = (0.0, -0.01, -0.02),
) -> pd.DataFrame:
    if common_returns.empty:
        return pd.DataFrame()
    if common_returns.isna().any().any():
        raise ValueError("Portfolio addition requires a fixed complete common sample")
    if reference not in common_returns:
        raise ValueError(f"Reference asset unavailable: {reference}")
    reference_returns = common_returns[reference]
    reference_metrics = metric_summary(reference_returns, annual_rf)
    rows = []
    for candidate in candidates:
        if candidate == reference or candidate not in common_returns:
            continue
        portfolio = weighted_portfolio(reference_returns, common_returns[candidate], candidate_weight)
        metrics = metric_summary(portfolio, annual_rf)
        reference_down = common_returns[reference] < 0
        reference_down_returns = reference_returns.loc[reference_down]
        portfolio_down_returns = portfolio.loc[reference_down]
        capture = (
            portfolio_down_returns.mean() / reference_down_returns.mean()
            if len(portfolio_down_returns) >= minimum_stress_observations and reference_down_returns.mean() != 0
            else np.nan
        )
        row = {
            "Candidate": candidate,
            "Candidate weight": candidate_weight,
            **metrics,
            "ΔCAGR vs reference": metrics["CAGR"] - reference_metrics["CAGR"],
            "ΔVolatility": metrics["Annualized volatility"] - reference_metrics["Annualized volatility"],
            "ΔSharpe": metrics["Sharpe"] - reference_metrics["Sharpe"],
            "ΔSortino": metrics["Sortino"] - reference_metrics["Sortino"],
            "ΔMaxDD": metrics["Max Drawdown"] - reference_metrics["Max Drawdown"],
            "ΔCalmar": metrics["Calmar"] - reference_metrics["Calmar"],
            "downside capture vs reference": capture,
            "analysis_start": common_returns.index.min(),
            "analysis_end": common_returns.index.max(),
            "common_weeks": len(common_returns),
        }
        for threshold in thresholds:
            label = "0pct" if threshold == 0 else f"{threshold * 100:g}pct"
            mask = reference_returns < threshold
            row[f"mean portfolio return when reference < {label}"] = (
                float(portfolio.loc[mask].mean()) if int(mask.sum()) >= minimum_stress_observations else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)
