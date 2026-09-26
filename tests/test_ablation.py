import numpy as np
import pandas as pd

from src.ablation import fixed_sample_ablation
from src.sample_policy import common_sample


def test_ablation_never_extends_sample_after_removal():
    dates = pd.date_range("2020-01-03", periods=8, freq="W-FRI")
    raw = pd.DataFrame({
        "A": np.linspace(0.0, 0.07, 8),
        "B": [np.nan, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07],
        "C": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, np.nan],
    }, index=dates)
    fixed = common_sample(raw).returns
    result = fixed_sample_ablation(fixed)
    assert result["analysis_start"].eq(fixed.index.min()).all()
    assert result["analysis_end"].eq(fixed.index.max()).all()
    assert result["common_weeks"].eq(len(fixed)).all()

