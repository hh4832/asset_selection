from __future__ import annotations

import numpy as np
import pandas as pd

from src.sample_policy import pairwise_overlap


def _safe_corr(a: pd.Series, b: pd.Series, minimum: int) -> float:
    return float(a.corr(b)) if len(a) >= minimum and a.std(ddof=1) > 0 and b.std(ddof=1) > 0 else np.nan


def conditioned_downside_metrics(
    returns: pd.DataFrame,
    reference: str,
    candidate: str,
    *,
    thresholds: tuple[float, ...] = (0.0, -0.01, -0.02),
    minimum_observations: int = 20,
) -> dict[str, object]:
    pair = pairwise_overlap(returns, reference, candidate)
    result: dict[str, object] = {
        "reference": reference,
        "candidate": candidate,
        "analysis_start": pair.index.min() if not pair.empty else pd.NaT,
        "analysis_end": pair.index.max() if not pair.empty else pd.NaT,
        "overlap_weeks": len(pair),
    }
    for threshold in thresholds:
        label = "down" if threshold == 0 else f"below_{abs(threshold) * 100:g}pct"
        conditioned = pair.loc[pair[reference] < threshold]
        count = len(conditioned)
        result[f"reference_{label}_weeks"] = count
        enough = count >= minimum_observations
        result[f"mean_candidate_return_ref_{label}"] = float(conditioned[candidate].mean()) if enough else np.nan
        result[f"positive_rate_ref_{label}"] = float(conditioned[candidate].gt(0).mean()) if enough else np.nan
        if threshold == 0:
            result["correlation_when_reference_down"] = _safe_corr(
                conditioned[candidate], conditioned[reference], minimum_observations,
            )
            variance = conditioned[reference].var(ddof=1)
            result["downside_beta"] = (
                float(conditioned[candidate].cov(conditioned[reference]) / variance)
                if enough and pd.notna(variance) and variance > 0 else np.nan
            )
            reference_mean = conditioned[reference].mean()
            result["downside_capture"] = (
                float(conditioned[candidate].mean() / reference_mean)
                if enough and pd.notna(reference_mean) and reference_mean != 0 else np.nan
            )
    result["insufficient_sample_warning"] = any(
        int(result[f"reference_{'down' if t == 0 else f'below_{abs(t) * 100:g}pct'}_weeks"])
        < minimum_observations for t in thresholds
    )
    return result


def directional_downside_table(
    returns: pd.DataFrame,
    reference: str,
    candidates: list[str],
    *,
    minimum_observations: int = 20,
) -> pd.DataFrame:
    return pd.DataFrame([
        conditioned_downside_metrics(
            returns, reference, candidate, minimum_observations=minimum_observations,
        ) for candidate in candidates if candidate != reference
    ])


def symmetric_downside_correlation(returns: pd.DataFrame, minimum_observations: int = 20) -> pd.DataFrame:
    assets = list(returns.columns)
    result = pd.DataFrame(np.nan, index=assets, columns=assets, dtype=float)
    for asset in assets:
        result.loc[asset, asset] = 1.0
    for i, asset_a in enumerate(assets):
        for asset_b in assets[i + 1:]:
            pair = pairwise_overlap(returns, asset_a, asset_b)
            downside = pair.loc[pair.mean(axis=1) < 0]
            correlation = _safe_corr(downside[asset_a], downside[asset_b], minimum_observations)
            result.loc[asset_a, asset_b] = result.loc[asset_b, asset_a] = correlation
    return result

