from squeeze_hunter.monitor.metrics import MetricsRegistry


def test_registry_records_orders() -> None:
    r = MetricsRegistry()
    r.record_order_submitted("buy", "pending")
    r.record_order_submitted("buy", "filled")
    r.record_order_submitted("sell", "filled")
    out = r.render()
    assert 'sh_orders_submitted_total{side="buy",status="pending"} 1.0' in out
    assert 'sh_orders_submitted_total{side="buy",status="filled"} 1.0' in out


def test_registry_sets_equity() -> None:
    r = MetricsRegistry()
    r.set_equity(123_456.78)
    out = r.render()
    assert "sh_equity_usd 123456.78" in out


def test_registry_kill_switch() -> None:
    r = MetricsRegistry()
    r.set_kill_switch_active("monthly_drawdown")
    out = r.render()
    assert 'sh_kill_switch_active{reason="monthly_drawdown"} 1.0' in out
