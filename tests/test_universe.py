from datetime import date

import pandas as pd

from squeeze_hunter.config import UniverseCfg
from squeeze_hunter.universe import build_universe


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
