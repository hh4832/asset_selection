from __future__ import annotations

import traceback
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

from src.ablation import fixed_sample_ablation
from src.clustering import correlation_clusters
from src.downside import directional_downside_table, symmetric_downside_correlation
from src.loaders import load_finlab_prices, load_tiingo_prices
from src.metrics import ewma_sharpe, metrics_table, rolling_sharpe
from src.normalization import validate_normalized_prices, weekly_adjusted_prices, weekly_returns_from_prices
from src.pair_analysis import directional_asymmetry
from src.plots import correlation_heatmap, dendrogram_plot, rolling_metric_plot
from src.portfolio_analysis import portfolio_addition_test
from src.sample_policy import common_sample, coverage_table, restrict_years


st.set_page_config(page_title="Asset Evaluation and Portfolio Diversification Research", layout="wide")


def parse_assets(text: str, reference: str) -> list[str]:
    values = text.replace("\r", "\n").replace(",", "\n").splitlines()
    normalized = [value.strip().upper().replace(" ", "") for value in values]
    return [asset for asset in dict.fromkeys(normalized) if asset and asset != reference]


@st.cache_data(ttl=3600, show_spinner=False)
def load_market_data(market: str, tickers: tuple[str, ...], start_date: date):
    if market == "US":
        prices, actions, errors = load_tiingo_prices(list(tickers), start_date=start_date)
        return prices, actions, errors, "Tiingo"
    prices, errors = load_finlab_prices(list(tickers), start_date=start_date)
    return prices, pd.DataFrame(), errors, "FinLab"


def legacy_exploratory(common_metrics: pd.DataFrame, ablation: pd.DataFrame) -> pd.DataFrame:
    """Expose legacy heuristics without using them for a decision or classification."""
    if common_metrics.empty:
        return pd.DataFrame()
    out = common_metrics.copy()
    positive = ["CAGR", "Sharpe", "Sortino", "Calmar"]
    negative = ["Annualized volatility"]
    ranks = [out[column].rank(pct=True) for column in positive]
    ranks.extend((-out[column]).rank(pct=True) for column in negative)
    out["AssetQualityScore"] = pd.concat(ranks, axis=1).mean(axis=1)
    if not ablation.empty:
        legacy_ablation = ablation[["removed_asset", "ΔSharpe", "ΔSortino", "ΔMaxDD", "ΔVolatility", "ΔCAGR", "ΔCalmar"]].copy()
        legacy_ablation["AblationScore"] = legacy_ablation[[
            "ΔSharpe", "ΔSortino", "ΔMaxDD", "ΔCAGR", "ΔCalmar",
        ]].rank(pct=True).mean(axis=1)
        out = out.merge(
            legacy_ablation[["removed_asset", "AblationScore"]],
            left_on="asset", right_on="removed_asset", how="left",
        ).drop(columns="removed_asset")
    out["AssetDiversifierScore"] = np.nan
    out["validation_status"] = "Legacy exploratory heuristic – not validated"
    return out


def download_csv(label: str, frame: pd.DataFrame, filename: str) -> None:
    st.download_button(
        label,
        data=frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
    )


st.title("Asset Evaluation and Portfolio Diversification Research")
st.caption(
    "Raw risk/return, reference-conditioned downside behavior, and portfolio impact. "
    "No validated composite score is used as the primary conclusion."
)

with st.sidebar:
    st.header("Analysis controls")
    market = st.selectbox("Market", ["US", "TW"])
    default_reference = "VOO" if market == "US" else "0050"
    reference = st.text_input("Reference Asset", value=default_reference).strip().upper().replace(" ", "")
    default_candidates = "GLD\nVGT\nSGOV" if market == "US" else "006208\n0056\n00679B"
    uploaded = st.file_uploader("Candidate Assets (.txt)", type=["txt"])
    candidate_text = st.text_area("Candidate Assets (one per line or comma-separated)", value=default_candidates)
    years_option = st.selectbox("Analysis years", [5, 10, 15, "Custom"], index=1)
    years = int(st.number_input("Custom years", 3, 30, 10)) if years_option == "Custom" else int(years_option)
    sample_mode = st.radio("Sample mode", ["Common sample", "Pairwise overlap"], horizontal=False)
    st.markdown("Stress thresholds (weekly reference return)")
    threshold_down = st.number_input("Down threshold (%)", -20.0, 5.0, 0.0, 0.5) / 100
    threshold_1 = st.number_input("Stress threshold 1 (%)", -20.0, 0.0, -1.0, 0.5) / 100
    threshold_2 = st.number_input("Stress threshold 2 (%)", -30.0, 0.0, -2.0, 0.5) / 100
    weight_option = st.selectbox("Portfolio candidate weight", [10, 20, "Custom"], index=1)
    weight_pct = float(st.number_input("Custom weight (%)", 1.0, 50.0, 20.0)) if weight_option == "Custom" else float(weight_option)
    minimum_stress = int(st.number_input("Minimum stress observations", 5, 104, 20))
    run = st.button("Run analysis", type="primary", use_container_width=True)

if not run:
    st.info("Choose a market, reference asset, and candidates, then run the analysis.")
    st.stop()

try:
    uploaded_text = uploaded.getvalue().decode("utf-8-sig") if uploaded else ""
    candidates = parse_assets(f"{candidate_text}\n{uploaded_text}", reference)
    if not reference:
        raise ValueError("Reference asset is required")
    if not candidates:
        raise ValueError("At least one candidate asset is required")
    requested = [reference, *candidates]
    if market not in {"US", "TW"}:
        raise ValueError("Mixed-market analysis is not supported in v1.")

    with st.spinner("Loading adjusted prices and running data-quality checks..."):
        # Retrieve provider history once. The requested-year rule is applied in
        # the analysis layer, while descriptive metrics retain available history.
        normalized, corporate_actions, load_errors, source = load_market_data(
            market, tuple(requested), date(1990, 1, 1),
        )
        if normalized.empty:
            raise RuntimeError(f"No usable data returned by {source}: {load_errors.to_dict('records')}")
        quality = validate_normalized_prices(normalized)
        if quality.has_fatal:
            raise RuntimeError(f"Fatal data-quality errors: {quality.issues.to_dict('records')}")
        weekly_prices = weekly_adjusted_prices(quality.data, market=market)
        all_returns, return_issues = weekly_returns_from_prices(weekly_prices)
        if not return_issues.empty and return_issues["severity"].eq("fatal").any():
            raise RuntimeError(f"Impossible returns detected: {return_issues.to_dict('records')}")

    if reference not in all_returns or all_returns[reference].dropna().empty:
        raise RuntimeError(f"Reference asset unavailable: {reference}")
    available_candidates = [candidate for candidate in candidates if candidate in all_returns and not all_returns[candidate].dropna().empty]
    unavailable = sorted(set(candidates) - set(available_candidates))
    if not available_candidates:
        raise RuntimeError("No candidate has usable adjusted-price returns")
    assets = [reference, *available_candidates]
    requested_returns = restrict_years(all_returns[assets], years)
    fixed = common_sample(requested_returns, assets)
    if fixed.weeks < 52:
        raise RuntimeError(f"Insufficient common sample: only {fixed.weeks} complete weeks")

    descriptive = metrics_table(all_returns[assets])
    common_metrics = metrics_table(fixed.returns)
    coverage = coverage_table(all_returns[assets])
    coverage["common_start"] = fixed.analysis_start
    coverage["common_end"] = fixed.analysis_end
    coverage["common_weeks"] = fixed.weeks

    pair_input = fixed.returns if sample_mode == "Common sample" else requested_returns
    thresholds = (threshold_down, threshold_1, threshold_2)
    defense = directional_downside_table(
        pair_input, reference, available_candidates, minimum_observations=minimum_stress,
    )
    if thresholds != (0.0, -0.01, -0.02):
        from src.downside import conditioned_downside_metrics
        defense = pd.DataFrame([
            conditioned_downside_metrics(
                pair_input, reference, candidate,
                thresholds=thresholds, minimum_observations=minimum_stress,
            ) for candidate in available_candidates
        ])
    asymmetry = directional_asymmetry(
        pair_input, reference, available_candidates, minimum_observations=minimum_stress,
    )
    ordinary_correlation = fixed.returns.corr()
    symmetric_downside = symmetric_downside_correlation(fixed.returns, minimum_observations=minimum_stress)
    cluster_table, linkage_matrix = correlation_clusters(fixed.returns)
    addition = portfolio_addition_test(
        fixed.returns, reference, available_candidates, weight_pct / 100,
        minimum_stress_observations=minimum_stress,
        thresholds=thresholds,
    )
    ablation = fixed_sample_ablation(fixed.returns)
    legacy = legacy_exploratory(common_metrics, ablation)
    rolling = pd.DataFrame({asset: rolling_sharpe(fixed.returns[asset]) for asset in assets})
    ewma = pd.DataFrame({asset: ewma_sharpe(fixed.returns[asset]) for asset in assets})

    metadata = pd.DataFrame([{
        "market": market, "reference_asset": reference,
        "candidate_assets": ",".join(available_candidates), "data_source": source,
        "requested_years": years, "sample_mode": sample_mode,
        "effective_common_start": fixed.analysis_start,
        "effective_common_end": fixed.analysis_end, "common_weeks": fixed.weeks,
        "candidate_weight": weight_pct / 100,
    }])

    st.header("1. Run Metadata")
    st.dataframe(metadata, use_container_width=True, hide_index=True)
    if unavailable:
        st.warning(f"Unavailable candidates skipped: {', '.join(unavailable)}")
    if not load_errors.empty:
        st.warning("Some provider requests failed; see Data Quality diagnostics.")

    st.header("2. Data Coverage")
    st.dataframe(coverage, use_container_width=True, hide_index=True)
    short_assets = coverage.loc[coverage["cohort"].eq("Insufficient history"), "asset"].tolist()
    if short_assets:
        st.warning(f"Insufficient history (<3 years): {', '.join(short_assets)}")

    st.header("3. Descriptive Single Asset Metrics")
    st.caption("Asset-specific available history. Do not use this table for formal cross-sectional ranking.")
    st.dataframe(descriptive.merge(coverage, on="asset", how="left"), use_container_width=True, hide_index=True)

    st.header("4. Common-Sample Asset Comparison")
    st.caption("Every asset is measured over the identical fixed common window shown above.")
    st.dataframe(common_metrics, use_container_width=True, hide_index=True)

    st.header("5. Correlation / Clustering")
    left, right = st.columns(2)
    with left:
        st.pyplot(correlation_heatmap(ordinary_correlation, "Ordinary correlation — fixed common sample"))
    with right:
        st.pyplot(correlation_heatmap(symmetric_downside, "Symmetric Downside Correlation"))
    st.caption("Descriptive only. This metric is symmetric and does not imply directional downside protection.")
    st.pyplot(dendrogram_plot(linkage_matrix, list(ordinary_correlation.columns)))
    st.dataframe(cluster_table, use_container_width=True, hide_index=True)

    st.header("6. Reference-Conditioned Defense")
    st.caption(f"Conditioning reference: {reference}. Pair sample policy: {sample_mode}.")
    st.dataframe(defense, use_container_width=True, hide_index=True)
    if not defense.empty and defense["insufficient_sample_warning"].any():
        st.warning("One or more stress regimes have insufficient observations; affected metrics are NaN.")

    st.header("7. Directional Asymmetry Table")
    st.dataframe(asymmetry, use_container_width=True, hide_index=True)

    st.header("8. Portfolio Addition Test")
    st.caption("All candidates use the same fixed common window; the baseline is 100% reference asset.")
    st.dataframe(addition, use_container_width=True, hide_index=True)

    st.header("9. Fixed-Sample Remove-One Ablation")
    st.dataframe(ablation, use_container_width=True, hide_index=True)

    st.header("10. Rolling / EWMA Regime Metrics")
    st.pyplot(rolling_metric_plot(rolling, "Rolling 52-week Sharpe"))
    st.pyplot(rolling_metric_plot(ewma, "EWMA Sharpe (52-week half-life)"))

    with st.expander("11. Legacy exploratory metrics — not validated", expanded=False):
        st.warning("These heuristic scores have not been validated out-of-sample. They are not decision outputs.")
        st.caption("AssetDiversifierScore is disabled because symmetric downside correlation no longer drives classification.")
        st.dataframe(legacy, use_container_width=True, hide_index=True)

    with st.expander("12. Data Quality diagnostics", expanded=not load_errors.empty):
        st.write("Provider request errors")
        st.dataframe(load_errors, use_container_width=True, hide_index=True)
        st.write("Normalized price checks")
        st.dataframe(quality.issues, use_container_width=True, hide_index=True)
        st.write("Return checks")
        st.dataframe(return_issues, use_container_width=True, hide_index=True)
        if not corporate_actions.empty:
            st.write("Provider corporate actions")
            st.dataframe(corporate_actions, use_container_width=True, hide_index=True)

    with st.expander("13. Export / Download CSV", expanded=False):
        download_csv("Download data coverage", coverage, "data_coverage.csv")
        download_csv("Download common metrics", common_metrics, "common_sample_metrics.csv")
        download_csv("Download reference-conditioned defense", defense, "reference_conditioned_defense.csv")
        download_csv("Download directional asymmetry", asymmetry, "directional_asymmetry.csv")
        download_csv("Download portfolio addition", addition, "portfolio_addition.csv")
        download_csv("Download ablation", ablation, "fixed_sample_ablation.csv")

except Exception as exc:
    st.error(f"Analysis failed: {exc}")
    with st.expander("Technical details"):
        st.code(traceback.format_exc())
