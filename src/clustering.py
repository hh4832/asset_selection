from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def correlation_clusters(common_returns: pd.DataFrame, correlation_threshold: float = 0.45) -> tuple[pd.DataFrame, np.ndarray | None]:
    correlation = common_returns.corr()
    assets = list(correlation.columns)
    if len(assets) <= 1:
        return pd.DataFrame({"asset": assets, "cluster_id": [1] * len(assets)}), None
    # Undefined correlation (for example a constant-return asset) is treated
    # as zero only for descriptive clustering distance, never for performance
    # or downside claims.
    distance = (1.0 - correlation.fillna(0.0)).clip(lower=0, upper=2).to_numpy(copy=True)
    np.fill_diagonal(distance, 0.0)
    link = linkage(squareform(distance, checks=False), method="average")
    cluster_ids = fcluster(link, t=1.0 - correlation_threshold, criterion="distance")
    return pd.DataFrame({"asset": assets, "cluster_id": cluster_ids}), link
