"""Signal Protocol + Factor schema. Each signal is a pure function:

    compute(universe_tickers, provider, clock) -> Factor

Factor is a dataframe keyed by ticker with the raw value and (later) z-score.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from squeeze_hunter.data.protocol import DataProvider


@dataclass
class Factor:
    name: str
    as_of: datetime
    values: pd.DataFrame  # columns: ticker, raw_value, evidence (optional dict)


SignalFn = Callable[[list[str], DataProvider, datetime], Awaitable[Factor]]
