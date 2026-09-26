from __future__ import annotations

from datetime import date

import pandas as pd

from src.normalization import NORMALIZED_PRICE_COLUMNS, validate_normalized_prices


def load_finlab_prices(
    tickers: list[str],
    *,
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Taiwan raw and adjusted closes through the supported FinLab SDK flow.

    Authentication is owned by FinLab. Run ``python -m finlab login`` or use
    the supported headless credential environment variables before starting
    Streamlit. No credential is persisted by this project.
    """
    try:
        import finlab
        from finlab import data
    except ImportError as exc:
        raise RuntimeError("FinLab is not installed; install requirements.txt") from exc

    try:
        finlab.login()
    except Exception as exc:
        raise RuntimeError(
            "FinLab authentication failed. Run `python -m finlab login`, or configure "
            "FINLAB_REFRESH_TOKEN, FINLAB_SESSION_ID, and FINLAB_API_KEY. "
            "Legacy FINLAB_API_TOKEN remains compatibility-only."
        ) from exc

    requested = list(dict.fromkeys(str(t).strip().upper() for t in tickers if str(t).strip()))
    if not requested:
        raise ValueError("No Taiwan tickers requested")
    try:
        data.set_market("tw")
        raw_close = data.get("price:收盤價")
        adjusted_close = data.get("etl:adj_close")
    except Exception as exc:
        raise RuntimeError("FinLab adjusted price retrieval failed") from exc

    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize() if end_date is not None else pd.Timestamp.today().normalize()
    rows: list[pd.DataFrame] = []
    errors: list[dict] = []
    for ticker in requested:
        if ticker not in raw_close.columns or ticker not in adjusted_close.columns:
            errors.append({"ticker": ticker, "error": "ticker unavailable in raw or adjusted FinLab prices"})
            continue
        frame = pd.DataFrame({
            "raw_close": pd.to_numeric(raw_close[ticker], errors="coerce"),
            "adjusted_close": pd.to_numeric(adjusted_close[ticker], errors="coerce"),
        }).loc[start:end]
        frame = frame.loc[frame[["raw_close", "adjusted_close"]].notna().any(axis=1)].reset_index()
        date_column = frame.columns[0]
        frame = frame.rename(columns={date_column: "date"})
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        frame["ticker"] = ticker
        frame["market"] = "TW"
        frame["source"] = "FinLab"
        quality = validate_normalized_prices(frame[NORMALIZED_PRICE_COLUMNS])
        if quality.has_fatal:
            errors.append({"ticker": ticker, "error": f"fatal data-quality failure: {quality.issues.to_dict('records')}"})
            continue
        rows.append(quality.data)
    prices = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=NORMALIZED_PRICE_COLUMNS)
    return prices, pd.DataFrame(errors, columns=["ticker", "error"])
