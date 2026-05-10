"""Conservative cost model for stocks and options."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class Fill:
    fill_price: float
    commission_usd: float


@dataclass
class StockCostModel:
    commission_per_share: float = 0.005
    slippage_high_price_bps: float = 5
    slippage_low_price_bps: float = 15
    low_price_threshold: float = 10.0
    open_window_extra_bps: float = 10

    def _slippage_bps(self: StockCostModel, price: float, is_open_5min: bool) -> float:
        base = (
            self.slippage_high_price_bps
            if price >= self.low_price_threshold
            else self.slippage_low_price_bps
        )
        return base + (self.open_window_extra_bps if is_open_5min else 0)

    def simulate_buy(
        self: StockCostModel, reference_price: float, qty: int, is_open_5min: bool = False
    ) -> Fill:
        bps = self._slippage_bps(reference_price, is_open_5min)
        return Fill(
            fill_price=reference_price * (1 + bps / 10_000),
            commission_usd=qty * self.commission_per_share,
        )

    def simulate_sell(
        self: StockCostModel, reference_price: float, qty: int, is_open_5min: bool = False
    ) -> Fill:
        bps = self._slippage_bps(reference_price, is_open_5min)
        return Fill(
            fill_price=reference_price * (1 - bps / 10_000),
            commission_usd=qty * self.commission_per_share,
        )


@dataclass
class OptionCostModel:
    commission_per_contract: float = 0.65

    def simulate_buy(self: OptionCostModel, mid: float, spread: float, qty: int) -> Fill:
        return Fill(
            fill_price=mid + 0.25 * spread,
            commission_usd=qty * self.commission_per_contract,
        )

    def simulate_sell(self: OptionCostModel, mid: float, spread: float, qty: int) -> Fill:
        return Fill(
            fill_price=mid - 0.25 * spread,
            commission_usd=qty * self.commission_per_contract,
        )
