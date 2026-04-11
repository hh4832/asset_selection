import io
import math
import traceback
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform


# =============================
# Constants
# =============================
ANNUAL_RF = 0.04
WEEKS_PER_YEAR = 52
RF_WEEKLY = (1 + ANNUAL_RF) ** (1 / WEEKS_PER_YEAR) - 1
DEFAULT_YEARS = 10
CORR_LOOKBACK_YEARS = 5
MIN_CORR_OVERLAP_WEEKS = 5 * WEEKS_PER_YEAR
ROLLING_SHARPE_WINDOW = 52
EWMA_HALFLIFE_WEEKS = 52
CLUSTER_CORR_THRESHOLD = 0.45
CLUSTER_DISTANCE_THRESHOLD = 1 - CLUSTER_CORR_THRESHOLD
PAIR_HIGH_CORR_THRESHOLD = 0.45
MIN_HISTORY_YEARS_FOR_ABLATION = 4


# =============================
# Utility functions
# =============================
def safe_pct_rank(series: pd.Series, ascending: bool = True) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index)
    return s.rank(pct=True, ascending=ascending)


def annualized_return(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) == 0:
        return np.nan
    total = float((1 + r).prod())
    years = len(r) / WEEKS_PER_YEAR
    if years <= 0 or total <= 0:
        return np.nan
    return total ** (1 / years) - 1


def annualized_volatility(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, rf_weekly: float = RF_WEEKLY) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 2:
        return np.nan
    excess = r - rf_weekly
    vol = excess.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return float((excess.mean() / vol) * np.sqrt(WEEKS_PER_YEAR))


def downside_deviation(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) == 0:
        return np.nan
    downside = r[r < 0]
    if len(downside) == 0:
        return 0.0
    return float(downside.std(ddof=1) * np.sqrt(WEEKS_PER_YEAR)) if len(downside) > 1 else 0.0


def sortino_ratio(returns: pd.Series, rf_weekly: float = RF_WEEKLY) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) < 2:
        return np.nan
    dd = downside_deviation(r)
    if dd == 0 or np.isnan(dd):
        return np.nan
    excess_ann = (r.mean() - rf_weekly) * WEEKS_PER_YEAR
    return float(excess_ann / dd)


def max_drawdown_from_returns(returns: pd.Series) -> float:
    r = pd.to_numeric(returns, errors="coerce").dropna()
    if len(r) == 0:
        return np.nan
    wealth = (1 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1
    return float(dd.min())


def calmar_ratio(returns: pd.Series) -> float:
    cagr = annualized_return(returns)
    mdd = max_drawdown_from_returns(returns)
    if pd.isna(cagr) or pd.isna(mdd) or mdd == 0:
        return np.nan
    return float(cagr / abs(mdd))


def rolling_sharpe(returns: pd.Series, window: int = ROLLING_SHARPE_WINDOW) -> pd.Series:
    r = pd.to_numeric(returns, errors="coerce")
    mean_excess = (r - RF_WEEKLY).rolling(window).mean()
    vol_excess = (r - RF_WEEKLY).rolling(window).std(ddof=1)
    rs = (mean_excess / vol_excess) * np.sqrt(WEEKS_PER_YEAR)
    return rs.replace([np.inf, -np.inf], np.nan)


def ewma_sharpe_series(returns: pd.Series, halflife_weeks: int = EWMA_HALFLIFE_WEEKS) -> pd.Series:
    r = pd.to_numeric(returns, errors="coerce")
    excess = r - RF_WEEKLY
    ewma_mean = excess.ewm(halflife=halflife_weeks, adjust=False, min_periods=12).mean()
    ewma_var = (excess ** 2).ewm(halflife=halflife_weeks, adjust=False, min_periods=12).mean() - ewma_mean ** 2
    ewma_std = np.sqrt(np.maximum(ewma_var, 0))
    result = (ewma_mean / ewma_std) * np.sqrt(WEEKS_PER_YEAR)
    return result.replace([np.inf, -np.inf], np.nan)


def trailing_window(series: pd.Series, years: int) -> pd.Series:
    n = years * WEEKS_PER_YEAR
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s.iloc[-n:] if len(s) >= n else s


def parse_tickers_from_text(text: str) -> List[str]:
    parts = [x.strip().upper() for x in text.replace("\n", ",").split(",")]
    return [x for x in parts if x]


def make_equal_weight_portfolio(returns_df: pd.DataFrame) -> pd.Series:
    aligned = returns_df.dropna(how="any")
    if aligned.empty:
        return pd.Series(dtype=float)
    return aligned.mean(axis=1)


def classify_assets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    aq80 = out["AssetQualityScore"].quantile(0.8)
    ads80 = out["AssetDiversifierScore"].quantile(0.8)
    ads20 = out["AssetDiversifierScore"].quantile(0.2)
    aq20 = out["AssetQualityScore"].quantile(0.2)
    ewma20 = out["EWMA_pct"].quantile(0.2) if "EWMA_pct" in out.columns else np.nan

    def _label(row: pd.Series) -> str:
        core_count = sum([
            row.get("AssetQualityScore", np.nan) >= aq80 if pd.notna(row.get("AssetQualityScore", np.nan)) else False,
            row.get("EWMA_latest", np.nan) >= 0.5 if pd.notna(row.get("EWMA_latest", np.nan)) else False,
            row.get("ΔSharpe", np.nan) < 0 if pd.notna(row.get("ΔSharpe", np.nan)) else False,
            row.get("ΔCalmar", np.nan) < 0 if pd.notna(row.get("ΔCalmar", np.nan)) else False,
        ])
        if core_count >= 3:
            return "Core Asset"
        if pd.notna(row.get("AssetDiversifierScore", np.nan)) and row.get("AssetDiversifierScore", np.nan) >= ads80:
            return "Diversifier"
        remove_count = sum([
            row.get("AssetDiversifierScore", np.nan) <= ads20 if pd.notna(row.get("AssetDiversifierScore", np.nan)) else False,
            row.get("AssetQualityScore", np.nan) <= aq20 if pd.notna(row.get("AssetQualityScore", np.nan)) else False,
            row.get("EWMA_pct", np.nan) <= ewma20 if pd.notna(row.get("EWMA_pct", np.nan)) and pd.notna(ewma20) else False,
        ])
        if remove_count >= 2:
            return "Consider remove"
        return "Others"

    out["SummaryLabel"] = out.apply(_label, axis=1)
    return out


def compute_downside_correlation(returns: pd.DataFrame) -> pd.DataFrame:
    assets = returns.columns
    result = pd.DataFrame(np.nan, index=assets, columns=assets)

    for a, b in combinations(assets, 2):
        pair = returns[[a, b]].dropna()
        if len(pair) < 52:  # 至少1年
            continue

        baseline = pair.mean(axis=1)
        downside = pair.loc[baseline < 0]

        if len(downside) < 20:
            continue

        corr = downside[a].corr(downside[b])
        result.loc[a, b] = corr
        result.loc[b, a] = corr

    for a in assets:
        result.loc[a, a] = 1.0

    return result


# =============================
# Data download
# =============================
def download_weekly_prices(tickers: List[str], years: int = DEFAULT_YEARS) -> Tuple[pd.DataFrame, Dict[str, str]]:
    errors: Dict[str, str] = {}
    if not tickers:
        return pd.DataFrame(), {"input": "No tickers provided."}

    end = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    start = end - pd.DateOffset(years=years, months=2)

    try:
        raw = yf.download(
            tickers=tickers,
            start=start,
            end=end,
            interval="1wk",
            auto_adjust=True,
            actions=True,
            repair=True,
            threads=False,
            group_by="ticker",
            progress=False,
            keepna=False,
        )
    except Exception as e:
        return pd.DataFrame(), {"download": str(e)}

    prices = pd.DataFrame()
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if t not in raw.columns.get_level_values(0):
                    errors[t] = "Ticker missing in downloaded data"
                    continue
                sub = raw[t]
                col = "Close" if "Close" in sub.columns else sub.columns[0]
                s = pd.to_numeric(sub[col], errors="coerce").dropna()
            else:
                col = "Close" if "Close" in raw.columns else raw.columns[0]
                s = pd.to_numeric(raw[col], errors="coerce").dropna()
            if s.empty:
                errors[t] = "No price data after cleaning"
                continue
            prices[t] = s
        except Exception as e:
            errors[t] = str(e)

    prices = prices.sort_index().dropna(how="all")
    return prices, errors


# =============================
# Core computations
# =============================
def compute_single_asset_metrics(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in returns.columns:
        try:
            s = returns[ticker].dropna()
            if len(s) < 52:
                continue
            s_1y = trailing_window(s, 1)
            s_5y = trailing_window(s, 5)
            s_10y = trailing_window(s, 10)
            row = {
                "asset": ticker,
                "CAGR_5y": annualized_return(s_5y),
                "CAGR_10y": annualized_return(s_10y),
                "Volatility": annualized_volatility(s_10y),
                "Sharpe_1y": sharpe_ratio(s_1y),
                "Sharpe_5y": sharpe_ratio(s_5y),
                "Sharpe_10y": sharpe_ratio(s_10y),
                "Sortino": sortino_ratio(s_5y),
                "DownsideDeviation": downside_deviation(s_5y),
                "MaxDD_10y": max_drawdown_from_returns(s_10y),
            }
            rows.append(row)
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["Sharpe_5y_rank"] = safe_pct_rank(df["Sharpe_5y"], ascending=True)
    df["Sharpe_10y_rank"] = safe_pct_rank(df["Sharpe_10y"], ascending=True)
    df["CAGR_5y_rank"] = safe_pct_rank(df["CAGR_5y"], ascending=True)
    df["Sortino_rank"] = safe_pct_rank(df["Sortino"], ascending=True)
    df["MDD_10y_rev_rank"] = safe_pct_rank(df["MaxDD_10y"], ascending=True)

    df["AssetQualityScore"] = (
        0.35 * df["Sharpe_5y_rank"]
        + 0.15 * df["Sharpe_10y_rank"]
        + 0.25 * df["CAGR_5y_rank"]
        + 0.15 * df["Sortino_rank"]
        + 0.10 * df["MDD_10y_rev_rank"]
    )
    return df.sort_values("AssetQualityScore", ascending=False).reset_index(drop=True)


def compute_correlation_matrix_5y(returns: pd.DataFrame) -> pd.DataFrame:
    tickers = list(returns.columns)
    corr = pd.DataFrame(np.nan, index=tickers, columns=tickers, dtype=float)
    lookback = returns.tail(CORR_LOOKBACK_YEARS * WEEKS_PER_YEAR)
    for i in tickers:
        corr.loc[i, i] = 1.0
    for a, b in combinations(tickers, 2):
        pair = lookback[[a, b]].dropna()
        if len(pair) >= MIN_CORR_OVERLAP_WEEKS:
            c = pair[a].corr(pair[b])
            corr.loc[a, b] = c
            corr.loc[b, a] = c
    return corr


def compute_pair_study(
    returns: pd.DataFrame,
    asset_metrics: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    downside_corr: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    metric_map = asset_metrics.set_index("asset")

    for a, b in combinations(returns.columns, 2):
        try:
            pair_data = returns[[a, b]].dropna()
            if len(pair_data) < MIN_CORR_OVERLAP_WEEKS:
                continue

            a_ret = pair_data[a]
            b_ret = pair_data[b]
            pair_ret = pair_data.mean(axis=1)  # 50/50 equal weight

            a_sharpe = sharpe_ratio(a_ret)
            b_sharpe = sharpe_ratio(b_ret)
            pair_sharpe = sharpe_ratio(pair_ret)

            a_sortino = sortino_ratio(a_ret)
            b_sortino = sortino_ratio(b_ret)
            pair_sortino = sortino_ratio(pair_ret)

            a_mdd = max_drawdown_from_returns(a_ret)
            b_mdd = max_drawdown_from_returns(b_ret)
            pair_mdd = max_drawdown_from_returns(pair_ret)

            sharpe_gain = pair_sharpe - ((a_sharpe + b_sharpe) / 2)
            sortino_gain = pair_sortino - ((a_sortino + b_sortino) / 2)
            maxdd_gain = pair_mdd - ((a_mdd + b_mdd) / 2)

            rows.append({
                "asset_a": a,
                "asset_b": b,
                "Sharpe_gain": sharpe_gain,
                "Sortino_gain": sortino_gain,
                "MaxDD_gain": maxdd_gain,
                "correlation": corr_matrix.loc[a, b],
            })
        except Exception:
            continue

    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        return pair_df, pd.DataFrame(columns=["asset", "AssetDiversifierScore"])

    pair_df["Sharpe_gain_rank"] = safe_pct_rank(pair_df["Sharpe_gain"], ascending=True)
    pair_df["Sortino_gain_rank"] = safe_pct_rank(pair_df["Sortino_gain"], ascending=True)
    pair_df["MaxDD_gain_rank"] = safe_pct_rank(pair_df["MaxDD_gain"], ascending=True)
    pair_df["Pair_gain_score"] = (
        0.35 * pair_df["Sharpe_gain_rank"]
        + 0.35 * pair_df["MaxDD_gain_rank"]
        + 0.30 * pair_df["Sortino_gain_rank"]
    )

    long_pairs = pd.concat([
        pair_df[["asset_a", "Pair_gain_score"]].rename(columns={"asset_a": "asset"}),
        pair_df[["asset_b", "Pair_gain_score"]].rename(columns={"asset_b": "asset"}),
    ], ignore_index=True)

    agg = long_pairs.groupby("asset")["Pair_gain_score"].agg(["median", "mean"]).reset_index()

    downside_avg = downside_corr.copy()
    downside_avg = downside_avg.apply(
        lambda row: row.drop(row.name).mean(), axis=1
    ).reset_index()
    downside_avg.columns = ["asset", "Downside_Corr_Avg"]

    asset_div = agg.merge(downside_avg, on="asset", how="left")

    asset_div["AssetDiversifierScore"] = (
        0.5 * asset_div["median"]
        + 0.5 * asset_div["mean"]
        - 0.4 * asset_div["Downside_Corr_Avg"]
    )

    asset_div = asset_div.rename(columns={
        "median": "PairGain_median",
        "mean": "PairGain_mean",
    })

    return pair_df, asset_div[["asset", "PairGain_median", "PairGain_mean", "Downside_Corr_Avg", "AssetDiversifierScore"]]


def compute_baseline_portfolio_metrics(returns: pd.DataFrame) -> Tuple[pd.Series, Dict[str, float], pd.Series]:
    baseline = make_equal_weight_portfolio(returns)
    metrics = {
        "Portfolio Sharpe": sharpe_ratio(baseline),
        "Portfolio Sortino": sortino_ratio(baseline),
        "Portfolio MaxDD": max_drawdown_from_returns(baseline),
        "Portfolio Volatility": annualized_volatility(baseline),
        "Portfolio CAGR": annualized_return(baseline),
        "Portfolio Calmar Ratio": calmar_ratio(baseline),
    }
    roll = rolling_sharpe(baseline, window=ROLLING_SHARPE_WINDOW)
    return baseline, metrics, roll

def get_mature_assets_for_ablation(
    returns: pd.DataFrame,
    min_years: int = MIN_HISTORY_YEARS_FOR_ABLATION
) -> List[str]:
    min_weeks = min_years * WEEKS_PER_YEAR
    valid_counts = returns.notna().sum()
    mature_assets = valid_counts[valid_counts >= min_weeks].index.tolist()
    return mature_assets

def compute_ablation(returns: pd.DataFrame, baseline_metrics: Dict[str, float]) -> pd.DataFrame:
    rows = []
    assets = list(returns.columns)
    for asset in assets:
        try:
            subset = returns.drop(columns=[asset])
            if subset.shape[1] == 0:
                continue
            new_port = make_equal_weight_portfolio(subset)
            row = {
                "asset": asset,
                "ΔSharpe": sharpe_ratio(new_port) - baseline_metrics["Portfolio Sharpe"],
                "ΔSortino": sortino_ratio(new_port) - baseline_metrics["Portfolio Sortino"],
                "ΔMaxDD": max_drawdown_from_returns(new_port) - baseline_metrics["Portfolio MaxDD"],
                "ΔVolatility": annualized_volatility(new_port) - baseline_metrics["Portfolio Volatility"],
                "ΔCAGR": annualized_return(new_port) - baseline_metrics["Portfolio CAGR"],
                "ΔCalmar": calmar_ratio(new_port) - baseline_metrics["Portfolio Calmar Ratio"],
            }
            rows.append(row)
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["ΔSharpe_rank"] = safe_pct_rank(df["ΔSharpe"], ascending=True)
    df["ΔCalmar_rank"] = safe_pct_rank(df["ΔCalmar"], ascending=True)
    df["ΔSortino_rank"] = safe_pct_rank(df["ΔSortino"], ascending=True)
    df["ΔCAGR_rank"] = safe_pct_rank(df["ΔCAGR"], ascending=True)
    df["-ΔVol_rank"] = safe_pct_rank(-df["ΔVolatility"], ascending=True)

    df["AblationScore"] = (
        0.25 * df["ΔSharpe_rank"]
        + 0.35 * df["ΔCalmar_rank"]
        + 0.15 * df["ΔSortino_rank"]
        + 0.15 * df["ΔCAGR_rank"]
        + 0.10 * df["-ΔVol_rank"]
    )
    return df.sort_values("AblationScore", ascending=False).reset_index(drop=True)


def compute_ewma_table(returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    latest_rows = []
    ts_dict = {}
    for asset in returns.columns:
        try:
            s = ewma_sharpe_series(returns[asset])
            ts_dict[asset] = s
            latest_rows.append({"asset": asset, "EWMA_latest": s.dropna().iloc[-1] if s.dropna().size else np.nan})
        except Exception:
            latest_rows.append({"asset": asset, "EWMA_latest": np.nan})
    latest_df = pd.DataFrame(latest_rows)
    ts_df = pd.DataFrame(ts_dict)
    return latest_df, ts_df

def build_data_coverage_table(
    returns: pd.DataFrame,
    mature_assets_for_ablation: List[str] | None = None,
) -> pd.DataFrame:
    rows = []

    mature_set = set(mature_assets_for_ablation or [])

    for asset in returns.columns:
        s = returns[asset].dropna()

        if len(s) == 0:
            rows.append({
                "asset": asset,
                "first_valid_date": pd.NaT,
                "last_valid_date": pd.NaT,
                "history_years": np.nan,
                "non_na_weeks": 0,
                "used_in_ablation": asset in mature_set,
                "meets_5y_week_threshold": False,
            })
            continue

        first_valid = s.index.min()
        last_valid = s.index.max()
        history_years = (last_valid - first_valid).days / 365.25

        rows.append({
            "asset": asset,
            "first_valid_date": first_valid.date(),
            "last_valid_date": last_valid.date(),
            "history_years": round(history_years, 2),
            "non_na_weeks": int(len(s)),
            "used_in_ablation": asset in mature_set,
            "meets_5y_week_threshold": len(s) >= MIN_CORR_OVERLAP_WEEKS,
        })

    df = pd.DataFrame(rows)

    return df.sort_values(
        ["used_in_ablation", "history_years", "non_na_weeks"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

def build_cluster_info(corr_matrix: pd.DataFrame, asset_metrics: pd.DataFrame) -> pd.DataFrame:
    assets = list(corr_matrix.index)
    if len(assets) <= 1:
        out = pd.DataFrame({"asset": assets, "cluster_id": [1] * len(assets)})
        score_map = asset_metrics.set_index("asset")["AssetQualityScore"] if not asset_metrics.empty else pd.Series(dtype=float)
        out["AssetQualityScore"] = out["asset"].map(score_map)
        out["cluster_rank"] = out.groupby("cluster_id")["AssetQualityScore"].rank(ascending=False, method="dense")
        return out[["asset", "cluster_id", "cluster_rank"]]

    dist = 1 - corr_matrix.fillna(0)
    np.fill_diagonal(dist.values, 0.0)
    condensed = squareform(dist.values, checks=False)
    link = linkage(condensed, method="average")
    cluster_ids = fcluster(link, t=CLUSTER_DISTANCE_THRESHOLD, criterion="distance")
    out = pd.DataFrame({"asset": assets, "cluster_id": cluster_ids})
    score_map = asset_metrics.set_index("asset")["AssetQualityScore"] if not asset_metrics.empty else pd.Series(dtype=float)
    out["AssetQualityScore"] = out["asset"].map(score_map)
    out["cluster_rank"] = out.groupby("cluster_id")["AssetQualityScore"].rank(ascending=False, method="dense")
    return out[["asset", "cluster_id", "cluster_rank"]]


def build_dashboard_tables(
    asset_metrics: pd.DataFrame,
    asset_div: pd.DataFrame,
    ablation_df: pd.DataFrame,
    ewma_latest_df: pd.DataFrame,
    cluster_info: pd.DataFrame,
) -> pd.DataFrame:
    df = asset_metrics.merge(asset_div, on="asset", how="left")
    df = df.merge(ablation_df, on="asset", how="left")
    df = df.merge(ewma_latest_df, on="asset", how="left")
    df = df.merge(cluster_info, on="asset", how="left")
    df["EWMA_pct"] = safe_pct_rank(df["EWMA_latest"], ascending=True)
    return classify_assets(df).sort_values("AssetQualityScore", ascending=False).reset_index(drop=True)


# =============================
# Plots
# =============================
def plot_rolling_sharpe(rolling_series: pd.Series):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(rolling_series.index, rolling_series.values)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("Portfolio Rolling Sharpe (52 weeks)")
    ax.set_ylabel("Sharpe")
    ax.set_xlabel("Date")
    fig.tight_layout()
    return fig


def plot_pair_heatmap(pair_df: pd.DataFrame, assets: List[str]):
    heat_arr = np.full((len(assets), len(assets)), np.nan, dtype=float)
    np.fill_diagonal(heat_arr, 0.0)
    heat = pd.DataFrame(heat_arr, index=assets, columns=assets)

    for _, row in pair_df.iterrows():
        heat.loc[row["asset_a"], row["asset_b"]] = row["Pair_gain_score"]
        heat.loc[row["asset_b"], row["asset_a"]] = row["Pair_gain_score"]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(heat.values, aspect="auto")

    ax.set_xticks(range(len(assets)))
    ax.set_xticklabels(assets, rotation=90)
    ax.set_yticks(range(len(assets)))
    ax.set_yticklabels(assets)

    for i in range(len(assets)):
        for j in range(len(assets)):
            val = heat.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title("Pair Gain Score Heatmap")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    return fig, heat


def plot_downside_corr_heatmap(corr_matrix: pd.DataFrame):
    assets = list(corr_matrix.index)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix.values, aspect="auto", vmin=-1, vmax=1)

    ax.set_xticks(range(len(assets)))
    ax.set_xticklabels(assets, rotation=90)
    ax.set_yticks(range(len(assets)))
    ax.set_yticklabels(assets)

    for i in range(len(assets)):
        for j in range(len(assets)):
            val = corr_matrix.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title("Downside Correlation Heatmap (portfolio < 0)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    return fig


def plot_dendrogram(corr_matrix: pd.DataFrame):
    assets = list(corr_matrix.index)
    if len(assets) <= 1:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "Need at least 2 assets for dendrogram.", ha="center", va="center")
        ax.axis("off")
        return fig
    dist = (1 - corr_matrix.fillna(0)).to_numpy(copy=True)
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    link = linkage(condensed, method="average")
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(link, labels=assets, ax=ax, leaf_rotation=90)
    ax.axhline(CLUSTER_DISTANCE_THRESHOLD, linestyle="--", linewidth=1)
    ax.set_title("Hierarchical Clustering Dendrogram (average linkage)")
    ax.set_ylabel("Distance = 1 - correlation")
    fig.tight_layout()
    return fig


# =============================
# Streamlit app
# =============================
st.set_page_config(page_title="Asset Selection Pipeline", layout="wide")
st.title("Asset Selection Pipeline")
st.caption("依照你提供的規格撰寫：weekly、10y、pair study、ablation、EWMA、dashboard。")

with st.sidebar:
    st.header("Inputs")
    txt_file = st.file_uploader("上傳 txt 檔（股票代碼逗號分隔）", type=["txt"])
    manual_text = st.text_area("或直接貼上股票代碼", value="NVDA, TSM, GOOGL, AVGO, LLY, NEM, JPM, WMT")
    years = st.number_input("資料年數", min_value=5, max_value=20, value=10, step=1)
    run_btn = st.button("開始計算", type="primary")

if run_btn:
    progress = st.progress(0, text="初始化中...")
    status_box = st.empty()

    try:
        progress.progress(5, text="讀取 ticker 清單...")
        input_text = txt_file.getvalue().decode("utf-8") if txt_file else manual_text
        tickers = parse_tickers_from_text(input_text)
        if len(tickers) < 2:
            st.error("請至少輸入 2 檔股票代碼。")
            st.stop()
        status_box.info(f"Tickers: {', '.join(tickers)}")

        progress.progress(15, text="下載 weekly 價格資料...")
        prices, download_errors = download_weekly_prices(tickers, years=years)
        if prices.empty:
            st.error(f"無法取得價格資料：{download_errors}")
            st.stop()

        returns = prices.pct_change().replace([np.inf, -np.inf], np.nan)
        returns = returns.dropna(how="all")

        progress.progress(25, text="計算 single asset metrics...")
        asset_metrics = compute_single_asset_metrics(returns)
        valid_assets = asset_metrics["asset"].tolist()
        if len(valid_assets) < 2:
            st.error("可用資產少於 2 檔，無法進行後續分析。")
            st.stop()
        returns_valid = returns[valid_assets]

        progress.progress(38, text="計算 correlation 與 clustering...")
        corr_matrix = compute_correlation_matrix_5y(returns_valid)
        cluster_info = build_cluster_info(corr_matrix, asset_metrics)
        downside_corr = compute_downside_correlation(returns_valid)

        progress.progress(50, text="計算 pair study...")
        pair_df, asset_div = compute_pair_study(
            returns_valid,
            asset_metrics,
            corr_matrix,
            downside_corr,
        )

        progress.progress(62, text="建立 baseline portfolio...")
        baseline_ret, portfolio_metrics, rolling_portfolio_sharpe = compute_baseline_portfolio_metrics(returns_valid)

        progress.progress(72, text="進行 ablation test...")
        mature_assets_for_ablation = get_mature_assets_for_ablation(
            returns_valid,
            min_years=MIN_HISTORY_YEARS_FOR_ABLATION,
        )

        if len(mature_assets_for_ablation) >= 2:
            returns_ablation = returns_valid[mature_assets_for_ablation]
            _, ablation_portfolio_metrics, _ = compute_baseline_portfolio_metrics(returns_ablation)
            ablation_df = compute_ablation(returns_ablation, ablation_portfolio_metrics)
        else:
            ablation_df = pd.DataFrame(columns=[
                "asset", "ΔSharpe", "ΔSortino", "ΔMaxDD", "ΔVolatility", "ΔCAGR", "ΔCalmar",
                "ΔSharpe_rank", "ΔCalmar_rank", "ΔSortino_rank", "ΔCAGR_rank", "-ΔVol_rank", "AblationScore"
            ])

        excluded_ablation_assets = [a for a in valid_assets if a not in mature_assets_for_ablation]

        data_coverage_df = build_data_coverage_table(
            returns_valid,
            mature_assets_for_ablation=mature_assets_for_ablation,
        )

        progress.progress(82, text="計算 EWMA Sharpe...")
        ewma_latest_df, ewma_ts_df = compute_ewma_table(returns_valid)

        progress.progress(90, text="整合 dashboard tables...")
        dashboard_df = build_dashboard_tables(asset_metrics, asset_div, ablation_df, ewma_latest_df, cluster_info)

        progress.progress(96, text="繪製圖表...")
        rolling_fig = plot_rolling_sharpe(rolling_portfolio_sharpe)
        pair_fig, pair_heat = plot_pair_heatmap(pair_df, valid_assets)
        downside_fig = plot_downside_corr_heatmap(downside_corr)
        dendro_fig = plot_dendrogram(corr_matrix)

        progress.progress(100, text="完成")
        status_box.success("所有階段完成。")

        st.subheader("1. Portfolio Summary")
        port_df = pd.DataFrame([portfolio_metrics]).T.reset_index()
        port_df.columns = ["Metric", "Value"]
        st.dataframe(port_df, use_container_width=True)

        st.subheader("2. Portfolio Rolling Sharpe Figure")
        st.pyplot(rolling_fig)

        st.subheader("3. Asset Metrics Table")
        st.dataframe(
            dashboard_df[
                [
                    "asset", "AssetQualityScore", "CAGR_5y", "CAGR_10y", "Volatility",
                    "Sharpe_1y", "Sharpe_5y", "Sharpe_10y", "Sortino", "DownsideDeviation",
                    "MaxDD_10y", "EWMA_latest", "AssetDiversifierScore", "cluster_id", "cluster_rank",
                    "SummaryLabel",
                ]
            ],
            use_container_width=True,
        )

        st.subheader("4. Ablation Table")
        st.dataframe(ablation_df, use_container_width=True)

        if excluded_ablation_assets:
            st.caption(
                f"Ablation 僅納入至少 {MIN_HISTORY_YEARS_FOR_ABLATION} 年資料的資產；"
                f"以下資產因資料較短未納入正式 ablation：{', '.join(excluded_ablation_assets)}"
            )

        st.subheader("4-2. Data Coverage Table")
        st.dataframe(data_coverage_df, use_container_width=True)

        st.subheader("5. Pair Gain Score Heatmap")
        st.pyplot(pair_fig)
        high_corr_pairs = pair_df[pair_df["correlation"] > PAIR_HIGH_CORR_THRESHOLD].sort_values("correlation", ascending=False)
        with st.expander("顯示 correlation > 0.45 的配對"):
            st.dataframe(high_corr_pairs, use_container_width=True)

        st.subheader("5-2. Downside Correlation Heatmap")
        st.pyplot(downside_fig)

        st.subheader("6. Hierarchical Clustering Dendrogram")
        st.pyplot(dendro_fig)

        st.subheader("7. 總結論")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown("**Core Asset**")
            st.dataframe(dashboard_df[dashboard_df["SummaryLabel"] == "Core Asset"][["asset", "AssetQualityScore", "EWMA_latest", "ΔSharpe", "ΔCalmar"]], use_container_width=True)
        with c2:
            st.markdown("**Diversifier**")
            st.dataframe(dashboard_df[dashboard_df["SummaryLabel"] == "Diversifier"][["asset", "AssetDiversifierScore", "AssetQualityScore"]], use_container_width=True)
        with c3:
            st.markdown("**Consider remove**")
            st.dataframe(dashboard_df[dashboard_df["SummaryLabel"] == "Consider remove"][["asset", "AssetDiversifierScore", "AssetQualityScore", "EWMA_latest"]], use_container_width=True)
        with c4:
            st.markdown("**Others**")
            st.dataframe(dashboard_df[dashboard_df["SummaryLabel"] == "Others"][["asset", "AssetQualityScore", "AssetDiversifierScore", "EWMA_latest"]], use_container_width=True)

        with st.expander("下載結果 CSV"):
            csv_buf = io.StringIO()
            dashboard_df.to_csv(csv_buf, index=False)
            st.download_button(
                "下載 dashboard_table.csv",
                data=csv_buf.getvalue().encode("utf-8-sig"),
                file_name="dashboard_table.csv",
                mime="text/csv",
            )

        with st.expander("查看下載錯誤 / 忽略項目"):
            st.json(download_errors)

        with st.expander("EWMA Sharpe 時間序列"):
            st.dataframe(ewma_ts_df, use_container_width=True)

    except Exception as e:
        progress.progress(100, text="失敗")
        st.error(f"執行失敗：{e}")
        st.code(traceback.format_exc())
else:
    st.info("上傳 txt 檔或貼上 ticker，然後按『開始計算』。")
