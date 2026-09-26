from __future__ import annotations

import numpy as np
import pandas as pd

from src.sample_policy import pairwise_overlap


def directional_asymmetry(
    returns: pd.DataFrame,
    reference: str,
    candidates: list[str],
    *,
    minimum_observations: int = 20,
) -> pd.DataFrame:
    rows = []
    for candidate in candidates:
        if candidate == reference:
            continue
        pair = pairwise_overlap(returns, reference, candidate)
        reference_down = pair.loc[pair[reference] < 0]
        candidate_down = pair.loc[pair[candidate] < 0]

        def enough(frame: pd.DataFrame) -> bool:
            return len(frame) >= minimum_observations

        rows.append({
            "reference": reference,
            "candidate": candidate,
            "E[candidate | reference < 0]": float(reference_down[candidate].mean()) if enough(reference_down) else np.nan,
            "E[reference | candidate < 0]": float(candidate_down[reference].mean()) if enough(candidate_down) else np.nan,
            "P(candidate > 0 | reference < 0)": float(reference_down[candidate].gt(0).mean()) if enough(reference_down) else np.nan,
            "P(reference > 0 | candidate < 0)": float(candidate_down[reference].gt(0).mean()) if enough(candidate_down) else np.nan,
            "Corr(reference,candidate | reference < 0)": float(reference_down[reference].corr(reference_down[candidate])) if enough(reference_down) else np.nan,
            "Corr(reference,candidate | candidate < 0)": float(candidate_down[reference].corr(candidate_down[candidate])) if enough(candidate_down) else np.nan,
            "reference_down_weeks": len(reference_down),
            "candidate_down_weeks": len(candidate_down),
            "analysis_start": pair.index.min() if not pair.empty else pd.NaT,
            "analysis_end": pair.index.max() if not pair.empty else pd.NaT,
            "overlap_weeks": len(pair),
        })
    return pd.DataFrame(rows)

