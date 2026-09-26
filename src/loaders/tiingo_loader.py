from __future__ import annotations

import os
from datetime import date
from typing import Any

import pandas as pd
import requests

from src.normalization import normalize_price_records, validate_normalized_prices


TIINGO_URL = "https://api.tiingo.com/tiingo/daily/{ticker}/prices"
REQUIRED_FIELDS = {"date", "close", "adjClose", "divCash", "splitFactor"}


def load_tiingo_prices(
    tickers: list[str],
    *,
    start_date: str | date | pd.Timestamp,
    end_date: str | date | pd.Timestamp | None = None,
    token: str | None = None,
    session: Any | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Tiingo EOD data and return normalized prices, corporate actions, errors."""
    token = token or os.getenv("TIINGO_API_TOKEN")
    if not token:
        raise RuntimeError("TIINGO_API_TOKEN is not set")
    requested = list(dict.fromkeys(str(t).strip().upper() for t in tickers if str(t).strip()))
    if not requested:
        raise ValueError("No US tickers requested")
    client = session or requests.Session()
    headers = {"Authorization": f"Token {token}", "Content-Type": "application/json"}
    params = {"startDate": str(pd.Timestamp(start_date).date()), "resampleFreq": "daily"}
    if end_date is not None:
        params["endDate"] = str(pd.Timestamp(end_date).date())
    normalized: list[pd.DataFrame] = []
    actions: list[pd.DataFrame] = []
    errors: list[dict] = []
    for ticker in requested:
        try:
            response = client.get(
                TIINGO_URL.format(ticker=ticker), params=params, headers=headers, timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            raw = pd.DataFrame(payload)
            missing = sorted(REQUIRED_FIELDS - set(raw.columns))
            if raw.empty or missing:
                raise RuntimeError(f"empty response or missing fields: {missing}")
            price = normalize_price_records(
                raw, ticker=ticker, market="US", source="Tiingo",
                date_column="date", raw_close_column="close", adjusted_close_column="adjClose",
            )
            quality = validate_normalized_prices(price)
            if quality.has_fatal:
                raise RuntimeError(f"fatal data-quality failure: {quality.issues.to_dict('records')}")
            normalized.append(quality.data)
            action = raw[["date", "divCash", "splitFactor"]].copy()
            action["date"] = pd.to_datetime(action["date"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
            action["ticker"] = ticker
            action["source"] = "Tiingo"
            action = action.loc[
                pd.to_numeric(action["divCash"], errors="coerce").fillna(0).ne(0)
                | pd.to_numeric(action["splitFactor"], errors="coerce").fillna(1).ne(1)
            ]
            actions.append(action)
        except Exception as exc:
            errors.append({"ticker": ticker, "error": str(exc)})
    prices = pd.concat(normalized, ignore_index=True) if normalized else pd.DataFrame(columns=[
        "date", "ticker", "market", "source", "raw_close", "adjusted_close",
    ])
    corporate_actions = pd.concat(actions, ignore_index=True) if actions else pd.DataFrame(columns=[
        "date", "ticker", "divCash", "splitFactor", "source",
    ])
    return prices, corporate_actions, pd.DataFrame(errors, columns=["ticker", "error"])

