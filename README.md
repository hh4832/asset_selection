# Asset Evaluation and Portfolio Diversification Research

A reproducible Streamlit research tool for evaluating US or Taiwan assets with interpretable risk/return metrics, explicit data coverage, reference-conditioned downside behavior, and portfolio-level diversification tests.

The production dashboard is `app.py`. The former `asset_selection_app_v3.02.py` remains for reproducibility but is deprecated: its composite scores and Core Asset / Diversifier / Consider Remove classifications are exploratory heuristics that have not been validated out-of-sample.

## Research design

The user selects one reference asset and a set of candidates. `VOO` is only the default US reference and `0050` is only the default Taiwan reference; either can be changed in the sidebar.

Primary outputs are raw metrics and portfolio impact:

- descriptive full-available-history metrics for each asset;
- fixed common-sample metrics for fair cross-sectional comparisons;
- reference-conditioned downside behavior;
- directional asymmetry;
- adding each candidate to a 100% reference portfolio; and
- fixed-window remove-one ablation.

No new subjective composite score is used as the primary conclusion.

## Data sources and credentials

### US — Tiingo

Set `TIINGO_API_TOKEN` in the environment or as a Colab Secret. The loader retrieves `close`, `adjClose`, `divCash`, and `splitFactor`. The normalized schema preserves both raw and adjusted close, but all return analysis uses `adjClose`.

### Taiwan — FinLab

Use FinLab's supported browser/CLI login flow:

```bash
python -m finlab login
```

For a headless environment, export the credentials produced by `python -m finlab token --env`: `FINLAB_REFRESH_TOKEN`, `FINLAB_SESSION_ID`, and `FINLAB_API_KEY`. Legacy `FINLAB_API_TOKEN` remains compatibility-only and is not embedded in source code.

The loader retrieves `price:收盤價` for audit and `etl:adj_close` for return analysis. Credentials are never written to source code.

Both providers are normalized to:

```text
date, ticker, market, source, raw_close, adjusted_close
```

The production path does not use yfinance. Prices and returns are never forward-filled.

## Sample policy

Three uses of history are deliberately separated:

1. **Descriptive metrics** use each asset's available provider history and show first date, last date, history years, non-missing weeks, and cohort.
2. **Common-sample comparison** computes each asset's weekly returns first, then retains only weeks available for every selected asset. Portfolio addition and ablation always use this fixed index.
3. **Pairwise overlap** may be selected for reference/candidate downside tables. Each row reports its own analysis dates, overlap weeks, and stress-regime counts.

History cohorts are:

- Long: at least 10 years;
- Medium: at least 5 but under 10 years;
- Short: at least 3 but under 5 years; and
- Insufficient: under 3 years.

Full-history values from different periods are descriptive and should not be treated as a formal ranking.

## Reference-conditioned downside analysis

For a reference return $r_{ref}$ and candidate return $r_i$, the dashboard reports:

- correlation and beta conditional on $r_{ref}<0$;
- mean candidate return and positive-return rate when the reference is below 0%, −1%, and −2%;
- downside capture relative to the reference; and
- observation counts and overlap dates.

It separately shows the reverse question—what the reference does when the candidate is down—because downside protection is directional. The symmetric downside-correlation heatmap is retained only as a descriptive view and does not imply defensive protection.

## Portfolio addition test

For candidate weight $w$, the weekly test portfolio is:

```text
(1 - w) × reference + w × candidate
```

It compares CAGR, volatility, Sharpe, Sortino, maximum drawdown, Calmar, and their differences from 100% reference, plus stress-period portfolio returns and downside capture. Every candidate uses the same common window.

## Fixed-sample ablation

The baseline is an equal-weight portfolio of all selected assets on the established common index. Removing an asset changes only the columns and never the dates. Reported changes include Sharpe, Sortino, maximum drawdown, volatility, CAGR, and Calmar.

## Data-quality policy

The analysis checks duplicate/non-monotonic dates, missing or non-positive adjusted prices, impossible returns, short histories, insufficient overlap, missing reference/candidates, and failed provider requests. Fatal problems stop the run; non-fatal limitations remain visible in diagnostics. No API data is modified and no missing observation is silently forward-filled.

## Local use

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export TIINGO_API_TOKEN='...'
streamlit run app.py
```

For Taiwan analysis, configure FinLab instead of Tiingo.

Run tests with:

```bash
pytest -q
```

## Google Colab

Open [`notebooks/asset_selection_colab.ipynb`](notebooks/asset_selection_colab.ipynb). It clones the latest `main`, installs requirements, reads the Tiingo or supported FinLab credentials from Colab Secrets (or runs interactive FinLab login), starts Streamlit, and creates a temporary Cloudflare public URL. No app source edit is required.

## Project structure

```text
app.py
src/
  loaders/tiingo_loader.py
  loaders/finlab_loader.py
  normalization.py
  metrics.py
  sample_policy.py
  downside.py
  pair_analysis.py
  portfolio_analysis.py
  ablation.py
  clustering.py
  plots.py
tests/
notebooks/asset_selection_colab.ipynb
asset_selection_app_v3.02.py  # deprecated legacy app
```

## Limitations

- Mixed US/Taiwan portfolios are not supported in v1 because USD/TWD FX conversion is not implemented.
- Short-history assets may have too few stress observations; the affected metrics are returned as NaN with a warning.
- Weekly close data cannot resolve intraday protection or execution effects.
- Historical downside behavior does not guarantee future protection.
- The legacy heuristic scores have not been walk-forward or out-of-sample validated.
