from squeeze_hunter.broker.base import BrokerHealth, IBroker


def test_protocol_attributes() -> None:
    assert "connect" in IBroker.__dict__
    assert "disconnect" in IBroker.__dict__
    assert "fetch_quote" in IBroker.__dict__


def test_broker_health_dataclass() -> None:
    h = BrokerHealth(connected=True, last_ping_ms=42, account="DU0")
    assert h.connected is True
    assert h.last_ping_ms == 42
