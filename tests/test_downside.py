import numpy as np
import pandas as pd
import pytest

from src.downside import conditioned_downside_metrics
from src.pair_analysis import directional_asymmetry


def test_directional_downside_is_asymmetric():
    dates = pd.date_range("2025-01-03", periods=8, freq="W-FRI")
    returns = pd.DataFrame({
        "A": [-0.03, -0.02, -0.01, 0.04, 0.03, 0.02, 0.01, 0.01],
        "B": [0.02, 0.01, 0.03, -0.04, -0.03, -0.02, 0.01, 0.02],
    }, index=dates)
    result = directional_asymmetry(returns, "A", ["B"], minimum_observations=3).iloc[0]
    assert result["E[candidate | reference < 0]"] > 0
    assert result["E[reference | candidate < 0]"] > 0
    assert result["reference_down_weeks"] != result["candidate_down_weeks"] or (
        result["Corr(reference,candidate | reference < 0)"]
        != result["Corr(reference,candidate | candidate < 0)"]
    )


def test_downside_beta_and_capture_formula():
    dates = pd.date_range("2025-01-03", periods=6, freq="W-FRI")
    reference = pd.Series([-0.01, -0.02, -0.03, 0.01, 0.02, 0.03], index=dates)
    candidate = 0.5 * reference
    returns = pd.DataFrame({"REF": reference, "CAND": candidate})
    result = conditioned_downside_metrics(returns, "REF", "CAND", minimum_observations=3)
    assert result["downside_beta"] == pytest.approx(0.5)
    assert result["downside_capture"] == pytest.approx(0.5)


def test_insufficient_stress_sample_returns_nan_and_warning():
    dates = pd.date_range("2025-01-03", periods=5, freq="W-FRI")
    returns = pd.DataFrame({"REF": [-0.01, 0.01, 0.02, 0.03, 0.04], "CAND": [0.01] * 5}, index=dates)
    result = conditioned_downside_metrics(returns, "REF", "CAND", minimum_observations=3)
    assert np.isnan(result["downside_beta"])
    assert result["insufficient_sample_warning"] is True

