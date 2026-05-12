from datetime import date

import pandas as pd

from squeeze_hunter.config import UniverseCfg
from squeeze_hunter.universe import build_universe


def test_universe_rejects_nonpositive_price() -> None:
    """R10.10 regression: a ticker with `close <= 0` (delisted, halted, or
    upstream data error) must be excluded with a distinct reason — silently
    sliding past the price-floor check would let zero-priced rows through to
    the scan/score pipeline where they'd corrupt z-scores or divide by zero.
    """
    rows = pd.DataFrame(
        [
            {"ticker": "ZERO", "market_cap": 1e9, "close": 0.0, "days_listed": 365},
            {"ticker": "NEG", "market_cap": 1e9, "close": -1.0, "days_listed": 365},
        ]
    )
    out = build_universe(rows, as_of=date(2024, 5, 13), cfg=UniverseCfg())
    by_ticker = out.set_index("ticker")
    assert not bool(by_ticker.loc["ZERO", "included"])
    assert not bool(by_ticker.loc["NEG", "included"])
    # Distinct exclusion reasons so the operator can debug; "price_below_floor"
    # is fine but a dedicated "price_nonpositive" tag is better for ops.
    assert by_ticker.loc["ZERO", "exclusion_reason"] in {
        "price_nonpositive",
        "price_below_floor",
    }


def test_universe_rejects_nonpositive_market_cap() -> None:
    """R10.10: same defensive check for market cap. A negative market cap is
    a data-quality bug, not a real signal."""
    rows = pd.DataFrame(
        [
            {"ticker": "BAD", "market_cap": -1.0, "close": 10.0, "days_listed": 365},
            {"ticker": "ZERO_CAP", "market_cap": 0.0, "close": 10.0, "days_listed": 365},
        ]
    )
    out = build_universe(rows, as_of=date(2024, 5, 13), cfg=UniverseCfg())
    assert not bool(out.set_index("ticker").loc["BAD", "included"])
    assert not bool(out.set_index("ticker").loc["ZERO_CAP", "included"])


def test_universe_filters_by_cap_price_and_listing() -> None:
    rows = pd.DataFrame(
        [
            {"ticker": "GME", "market_cap": 1e9, "close": 18.0, "days_listed": 365},
            {"ticker": "AAPL", "market_cap": 3e12, "close": 200.0, "days_listed": 5000},  # over cap
            {
                "ticker": "PNNY",
                "market_cap": 1e8,
                "close": 1.0,
                "days_listed": 365,
            },  # below cap & price
            {"ticker": "NEW", "market_cap": 5e8, "close": 10.0, "days_listed": 10},  # too new
            {"ticker": "HTZ", "market_cap": 2e9, "close": 6.0, "days_listed": 1000},
        ]
    )
    cfg = UniverseCfg()
    out = build_universe(rows, as_of=date(2024, 5, 13), cfg=cfg)
    included = sorted(out.loc[out["included"], "ticker"].tolist())
    assert included == ["GME", "HTZ"]
