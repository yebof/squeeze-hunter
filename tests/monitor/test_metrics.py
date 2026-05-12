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


def test_registry_kill_switch_resets_to_inactive() -> None:
    """R9.4 regression: when the killswitch clears, the Prometheus gauge must
    return to 0.0 so Grafana dashboards reflect the current state. Prior code
    only had a "set active" path, leaving the gauge stuck at 1.0 forever after
    the first trip and breaking ops trust in the metric.
    """
    r = MetricsRegistry()
    r.set_kill_switch_active("monthly_drawdown")
    r.set_kill_switch_inactive("monthly_drawdown")
    out = r.render()
    assert 'sh_kill_switch_active{reason="monthly_drawdown"} 0.0' in out


def test_registry_kill_switch_inactive_without_prior_active() -> None:
    """Calling inactive when no prior active was recorded should still result
    in a 0.0 gauge (idempotent), not raise.
    """
    r = MetricsRegistry()
    r.set_kill_switch_inactive("monthly_drawdown")
    out = r.render()
    assert 'sh_kill_switch_active{reason="monthly_drawdown"} 0.0' in out
