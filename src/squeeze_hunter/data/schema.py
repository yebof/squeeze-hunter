"""Domain schemas. All timestamps UTC; only display layers convert."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, Field, computed_field, model_validator


class Bar(BaseModel):
    ticker: str
    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    @model_validator(mode="after")
    def _ohlc_consistent(self: Bar) -> Bar:
        if not (self.low <= self.open <= self.high and self.low <= self.close <= self.high):
            raise ValueError("OHLC inconsistent: low/high must bound open and close")
        return self


class Quote(BaseModel):
    ticker: str
    ts: datetime
    bid: float
    ask: float
    last: float
    bid_size: int = 0
    ask_size: int = 0


class OptionQuote(BaseModel):
    strike: float
    expiry: date
    right: Literal["C", "P"]
    open_interest: int
    volume: int
    implied_vol: float
    bid: float = 0.0
    ask: float = 0.0


class OptionChain(BaseModel):
    underlying: str
    as_of: datetime
    spot: float
    quotes: list[OptionQuote] = Field(default_factory=list)

    def near_money_calls(self: OptionChain, window_pct: float = 0.05) -> list[OptionQuote]:
        lo = self.spot * (1 - window_pct)
        hi = self.spot * (1 + window_pct)
        return [q for q in self.quotes if q.right == "C" and lo <= q.strike <= hi]

    def total_call_oi(self: OptionChain, *, near_money_window_pct: float | None = None) -> int:
        if near_money_window_pct is None:
            return sum(q.open_interest for q in self.quotes if q.right == "C")
        return sum(q.open_interest for q in self.near_money_calls(near_money_window_pct))


class ShortInterest(BaseModel):
    ticker: str
    settlement_date: date
    si_shares: int
    si_pct_float: float
    avg_daily_volume_20d: int

    @computed_field  # type: ignore[prop-decorator]
    @property
    def days_to_cover(self: ShortInterest) -> float:
        if self.avg_daily_volume_20d <= 0:
            return float("inf")
        return self.si_shares / self.avg_daily_volume_20d


class EarningsEvent(BaseModel):
    ticker: str
    report_at: datetime
    actual_eps: float | None = None
    estimate_eps: float | None = None
    actual_revenue: float | None = None
    estimate_revenue: float | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def surprise_pct(self: EarningsEvent) -> float | None:
        if self.actual_eps is None or self.estimate_eps in (None, 0):
            return None
        assert self.estimate_eps is not None
        return (self.actual_eps - self.estimate_eps) / abs(self.estimate_eps)


class RedditMention(BaseModel):
    ticker: str
    as_of: datetime
    subreddit: str
    count_24h: int
    baseline_30d_mean: float
    baseline_30d_std: float

    @computed_field  # type: ignore[prop-decorator]
    @property
    def z_score(self: RedditMention) -> float:
        if self.baseline_30d_std <= 0:
            return 0.0
        return (self.count_24h - self.baseline_30d_mean) / self.baseline_30d_std
