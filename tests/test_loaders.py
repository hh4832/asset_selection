import sys
import types

import pandas as pd

from src.loaders.finlab_loader import load_finlab_prices
from src.loaders.tiingo_loader import load_tiingo_prices


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self, payloads):
        self.payloads = payloads

    def get(self, url, **kwargs):
        ticker = url.split("/")[-2]
        return FakeResponse(self.payloads.get(ticker, []))


def test_tiingo_loader_uses_adjusted_close_and_corporate_actions():
    payloads = {"ABC": [
        {"date": "2026-01-02T00:00:00Z", "close": 100, "adjClose": 50, "divCash": 0, "splitFactor": 2},
        {"date": "2026-01-05T00:00:00Z", "close": 101, "adjClose": 51, "divCash": 1, "splitFactor": 1},
    ]}
    prices, actions, errors = load_tiingo_prices(
        ["ABC"], start_date="2026-01-01", token="test", session=FakeSession(payloads),
    )
    assert errors.empty
    assert prices["adjusted_close"].tolist() == [50, 51]
    assert prices["raw_close"].tolist() == [100, 101]
    assert len(actions) == 2


def test_tiingo_loader_reports_failed_ticker_without_silent_substitution():
    prices, _, errors = load_tiingo_prices(
        ["MISSING"], start_date="2026-01-01", token="test", session=FakeSession({}),
    )
    assert prices.empty
    assert errors.loc[0, "ticker"] == "MISSING"


def test_finlab_loader_normalizes_raw_and_adjusted_prices(monkeypatch):
    dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
    datasets = {
        "price:收盤價": pd.DataFrame({"0050": [100.0, 101.0]}, index=dates),
        "etl:adj_close": pd.DataFrame({"0050": [95.0, 96.5]}, index=dates),
    }

    class FakeData:
        market = None

        @classmethod
        def set_market(cls, market):
            cls.market = market

        @staticmethod
        def get(name):
            return datasets[name]

    fake_finlab = types.ModuleType("finlab")
    fake_finlab.data = FakeData
    fake_finlab.login = lambda: None
    monkeypatch.setitem(sys.modules, "finlab", fake_finlab)
    prices, errors = load_finlab_prices(["0050"], start_date="2026-01-01", end_date="2026-01-31")
    assert errors.empty
    assert FakeData.market == "tw"
    assert prices["raw_close"].tolist() == [100.0, 101.0]
    assert prices["adjusted_close"].tolist() == [95.0, 96.5]
    assert prices["market"].eq("TW").all()

