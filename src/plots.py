from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram


def correlation_heatmap(matrix: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    values = matrix.to_numpy(dtype=float)
    image = ax.imshow(values, aspect="auto", vmin=-1, vmax=1, cmap="coolwarm")
    assets = list(matrix.columns)
    ax.set_xticks(range(len(assets)), labels=assets, rotation=90)
    ax.set_yticks(range(len(assets)), labels=assets)
    for row in range(len(assets)):
        for column in range(len(assets)):
            value = values[row, column]
            if np.isfinite(value):
                ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    return fig


def dendrogram_plot(linkage_matrix, labels: list[str]):
    fig, ax = plt.subplots(figsize=(10, 5))
    if linkage_matrix is None:
        ax.text(0.5, 0.5, "Need at least two assets.", ha="center", va="center")
        ax.axis("off")
    else:
        dendrogram(linkage_matrix, labels=labels, leaf_rotation=90, ax=ax)
        ax.set_title("Correlation clustering (average linkage)")
        ax.set_ylabel("Distance = 1 - correlation")
    fig.tight_layout()
    return fig


def rolling_metric_plot(frame: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    frame.plot(ax=ax)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Date")
    fig.tight_layout()
    return fig

