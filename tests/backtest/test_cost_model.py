import pytest

from squeeze_hunter.backtest.cost_model import StockCostModel


def test_high_price_slippage_5_bps() -> None:
    m = StockCostModel()
    fill = m.simulate_buy(reference_price=100.0, qty=100, is_open_5min=False)
    assert fill.fill_price == pytest.approx(100.0 * (1 + 0.0005))
    assert fill.commission_usd == pytest.approx(0.005 * 100)


def test_low_price_slippage_15_bps() -> None:
    m = StockCostModel()
    fill = m.simulate_buy(reference_price=7.0, qty=100, is_open_5min=False)
    assert fill.fill_price == pytest.approx(7.0 * (1 + 0.0015))


def test_open_window_adds_10_bps() -> None:
    m = StockCostModel()
    fill = m.simulate_buy(reference_price=100.0, qty=100, is_open_5min=True)
    assert fill.fill_price == pytest.approx(100.0 * (1 + 0.0005 + 0.0010))


def test_sell_pays_negative_slippage() -> None:
    m = StockCostModel()
    fill = m.simulate_sell(reference_price=100.0, qty=100, is_open_5min=False)
    assert fill.fill_price == pytest.approx(100.0 * (1 - 0.0005))
