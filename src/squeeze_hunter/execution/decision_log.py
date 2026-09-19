"""Decision log (P8): why each candidate was or was not entered, per day.

The runner returned only a trade log, so a Gate 1 outcome could not be
explained after the fact ("why no trade in HTZ on 2025-04-21?"). Both the
backtest and the live premarket path now record one row per candidate per
day with the gate reason; `squeeze-hunter explain` reads them back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd

from squeeze_hunter.execution.decisions import EntryDecision

COLUMNS = [
    "date",
    "ticker",
    "score",
    "setup_type",
    "accepted",
    "reason",
    "size_usd",
    "source",
]


@dataclass
class DecisionLog:
    rows: list[dict[str, Any]] = field(default_factory=list)

    def record(self, as_of: datetime, decisions: list[EntryDecision], *, source: str) -> None:
        day = as_of.date().isoformat()
        for d in decisions:
            self.rows.append(
                {
                    "date": day,
                    "ticker": d.ticker,
                    "score": float(d.score),
                    "setup_type": d.setup_type,
                    "accepted": bool(d.accepted),
                    "reason": d.reason or "accepted",
                    "size_usd": float(d.size_usd),
                    "source": source,
                }
            )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=COLUMNS)


def explain(frame: pd.DataFrame, *, ticker: str | None = None, date: str | None = None) -> str:
    """Human-readable lines for the matching decision rows."""
    if frame.empty:
        return "no decisions recorded"
    sub = frame
    if ticker:
        sub = sub[sub["ticker"].str.upper() == ticker.upper()]
    if date:
        sub = sub[sub["date"] == date]
    if sub.empty:
        return f"no decisions for ticker={ticker or '*'} date={date or '*'}"
    lines = []
    for _, r in sub.sort_values(["date", "ticker"]).iterrows():
        verdict = "ENTER" if bool(r["accepted"]) else "skip "
        lines.append(
            f"{r['date']}  {r['ticker']:6s} {verdict}  score={r['score']:.2f}  "
            f"setup={r['setup_type']:5s} reason={r['reason']}  size=${r['size_usd']:.0f}"
            f"  [{r['source']}]"
        )
    return "\n".join(lines)
