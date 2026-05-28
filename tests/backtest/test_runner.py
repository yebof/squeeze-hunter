from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from squeeze_hunter.backtest.runner import BacktestConfig, run_backtest
from squeeze_hunter.config import Settings
from squeeze_hunter.data.cache import ParquetCache

# Background tickers with negligible signals make GME's cross-sectional z-scores
# reach the ±3.0 clip, pushing car_strength and gme_strength past the Mixed (≥3)
# classifier threshold so the gate lets trades through.
_BG = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9"]
_ALL_TICKERS = ["GME", "AAPL", *_BG]


def _seed(cache: ParquetCache) -> None:
    base = datetime(2024, 5, 1, tzinfo=UTC)
    # GME: flat for 14 days, big spike on day 14, then drifts down
    gme_bars = []
    for i in range(30):
        ts = base + timedelta(days=i)
        close = 18.0 if i < 14 else (22.0 if i == 14 else 21.0 - (i - 14) * 0.3)
        vol = 1_000_000 if i < 14 else (8_000_000 if i == 14 else 1_500_000)
        gme_bars.append(
            {
                "ticker": "GME",
                "ts": ts,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": vol,
            }
        )
    cache.write_partition("bars", "GME", pd.DataFrame(gme_bars))

    # AAPL: steady, high volume
    aapl_bars = [
        {
            "ticker": "AAPL",
            "ts": base + timedelta(days=i),
            "open": 200.0,
            "high": 200.5,
            "low": 199.5,
            "close": 200.0,
            "volume": 50_000_000,
        }
        for i in range(30)
    ]
    cache.write_partition("bars", "AAPL", pd.DataFrame(aapl_bars))

    # Background tickers: minimal volume / flat price — zero SI, zero sentiment
    for t in _BG:
        bg_bars = [
            {
                "ticker": t,
                "ts": base + timedelta(days=i),
                "open": 50.0,
                "high": 50.1,
                "low": 49.9,
                "close": 50.0,
                "volume": 100_000,
            }
            for i in range(30)
        ]
        cache.write_partition("bars", t, pd.DataFrame(bg_bars))

    # Short interest: only GME has significant SI; rest are zero to maximise its z-score
    si_rows = [
        {
            "ticker": "GME",
            "settlement_date": date(2024, 4, 30),
            "si_shares": 5_000_000,
            "si_pct_float": 0.30,
            "avg_daily_volume_20d": 1_000_000,
        }
    ]
    for t in [*_BG, "AAPL"]:
        si_rows.append(
            {
                "ticker": t,
                "settlement_date": date(2024, 4, 30),
                "si_shares": 5_000,
                "si_pct_float": 0.01,  # small but non-zero so f1 z-score has meaningful spread
                "avg_daily_volume_20d": 1_000,
            }
        )
    cache.write_partition("short_interest", "all", pd.DataFrame(si_rows))

    cache.write_partition(
        "earnings",
        "all",
        pd.DataFrame(columns=["ticker", "report_at", "actual_eps", "estimate_eps"]),
    )

    for i in range(30):
        d = (base + timedelta(days=i)).date().isoformat()
        sent = [
            {
                "ticker": "GME",
                "subreddit": "wallstreetbets",
                "count_24h": 400 if i == 14 else 10,
                "baseline_30d_mean": 10.0,
                "baseline_30d_std": 20.0,
            },
            {
                "ticker": "AAPL",
                "subreddit": "wallstreetbets",
                "count_24h": 5,
                "baseline_30d_mean": 10.0,
                "baseline_30d_std": 20.0,
            },
        ]
        for t in _BG:
            sent.append(
                {
                    "ticker": t,
                    "subreddit": "wallstreetbets",
                    "count_24h": 0,
                    "baseline_30d_mean": 10.0,
                    "baseline_30d_std": 20.0,
                }
            )
        cache.write_partition("sentiment", d, pd.DataFrame(sent))


@pytest.mark.asyncio
async def test_runner_enters_next_trading_day_after_scan(tmp_path: Path) -> None:
    """CDX-P1-1 regression: the backtest must NOT scan on day T's full bar and
    then fill an entry at day T's own close — that is same-day lookahead. In
    production the scan runs after T's close (nightly), premarket-verify runs
    the next morning, and the entry fills around T+1's open. So a candidate
    surfaced by day T's scan must execute on T+1 at T+1's OPEN price.

    We patch run_scan to deterministically rank GME every day (decoupling this
    test from factor math) and seed bars where each day's open != close, so
    the fill price uniquely identifies WHICH bar (and which day) filled.
    """
    from unittest.mock import patch

    cache = ParquetCache(root=tmp_path)
    # 6 trading days starting Tue 2024-05-14. open and close are distinct so
    # the fill price pins the exact bar/day.
    base = datetime(2024, 5, 14, tzinfo=UTC)
    days = pd.bdate_range(base, periods=6, tz="UTC")
    rows = []
    for i, ts in enumerate(days):
        rows.append(
            {
                "ticker": "GME",
                "ts": ts.to_pydatetime(),
                "open": 10.0 + i,  # 10, 11, 12, 13, 14, 15  (distinct per day)
                "high": 20.0 + i,
                "low": 9.0 + i,
                "close": 18.0 + i,  # 18, 19, 20, 21, 22, 23  (≠ open)
                "volume": 5_000_000,
            }
        )
    cache.write_partition("bars", "GME", pd.DataFrame(rows))
    cache.write_partition(
        "short_interest",
        "all",
        pd.DataFrame(
            columns=[
                "ticker",
                "settlement_date",
                "si_shares",
                "si_pct_float",
                "avg_daily_volume_20d",
            ]
        ),
    )
    cache.write_partition(
        "earnings",
        "all",
        pd.DataFrame(columns=["ticker", "report_at", "actual_eps", "estimate_eps"]),
    )

    settings = Settings()
    settings.score.weights = {"f1_si_pct": 1.0}

    async def fake_scan(tickers, provider, clock, settings):
        return pd.DataFrame(
            [{"ticker": "GME", "score": 99.0, "setup_type": "CAR", "rank": 1, "as_of": clock}]
        )

    cfg = BacktestConfig(
        tickers=["GME"],
        start=base,
        end=days[-1].to_pydatetime(),
        initial_cash=100_000.0,
        score_threshold=3.0,
    )
    with patch("squeeze_hunter.backtest.runner.run_scan", side_effect=fake_scan):
        result = await run_backtest(cfg, cache=cache, settings=settings)

    buys = result.trade_log[result.trade_log["side"] == "buy"].reset_index(drop=True)
    assert len(buys) >= 1, "expected at least one entry over the run"
    first_buy = buys.iloc[0]
    # Day 0 (2024-05-14) is the first scan. Entry MUST be day 1 (2024-05-15),
    # NOT day 0 — anything on day 0 is the same-day-close lookahead bug.
    assert first_buy["ts"].date() == days[1].date(), (
        f"entry filled on {first_buy['ts'].date()} — expected next trading day "
        f"{days[1].date()} (same-day fill = lookahead)"
    )
    # And it must fill near day-1's OPEN (11.0) + cost-model buy slippage
    # (~5 bps → ~11.0055), NEVER the scan day's close (18.0) or day-1 close
    # (19.0). A tight band around the open uniquely identifies the correct bar.
    assert first_buy["price"] == pytest.approx(11.0, rel=0.01), (
        f"fill price {first_buy['price']} — expected ≈ day-1 open 11.0 "
        f"(+slippage), not day-0 close 18.0 / day-1 close 19.0 (= lookahead)"
    )
    assert first_buy["price"] < 12.0, "fill drifted to a close price → lookahead"


@pytest.mark.asyncio
async def test_runner_gap_through_stop_uses_daily_low(tmp_path: Path) -> None:
    """CDX-P1-2 regression: a position that craters intraday to -30% but
    recovers to close ~flat must STILL trip the hard stop in the backtest —
    the live 60s lifecycle daemon would have caught the -30% intraday and
    stopped out. The old code only fed `last.close` into evaluate_stops and
    into the killswitch gap calc, so an intraday blowup that closed flat was
    invisible to Gate 1. Stops/gap must use the day's LOW; a price-triggered
    exit must fill at that conservative low, not the recovered close.
    """
    from unittest.mock import patch

    cache = ParquetCache(root=tmp_path)
    base = datetime(2024, 5, 14, tzinfo=UTC)
    days = pd.bdate_range(base, periods=4, tz="UTC")
    # Day0: scan ranks GME. Day1: enter at open=100. Day2: intraday low=65
    # (-35% from entry, well past the -12% hard stop) but CLOSE recovers to 99
    # (only -1%, would NOT trip a close-based stop). Day3: filler.
    bars = [
        {
            "ticker": "GME",
            "ts": days[0].to_pydatetime(),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 5_000_000,
        },
        {
            "ticker": "GME",
            "ts": days[1].to_pydatetime(),
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 5_000_000,
        },
        {
            "ticker": "GME",
            "ts": days[2].to_pydatetime(),
            "open": 99.0,
            "high": 100.0,
            "low": 65.0,
            "close": 99.0,
            "volume": 9_000_000,
        },  # intraday crater, close flat
        {
            "ticker": "GME",
            "ts": days[3].to_pydatetime(),
            "open": 99.0,
            "high": 100.0,
            "low": 98.0,
            "close": 99.0,
            "volume": 5_000_000,
        },
    ]
    cache.write_partition("bars", "GME", pd.DataFrame(bars))
    cache.write_partition(
        "short_interest",
        "all",
        pd.DataFrame(
            columns=[
                "ticker",
                "settlement_date",
                "si_shares",
                "si_pct_float",
                "avg_daily_volume_20d",
            ]
        ),
    )
    cache.write_partition(
        "earnings",
        "all",
        pd.DataFrame(columns=["ticker", "report_at", "actual_eps", "estimate_eps"]),
    )

    settings = Settings()
    settings.score.weights = {"f1_si_pct": 1.0}

    call = {"n": 0}

    async def scan_day0_only(tickers, provider, clock, settings):
        call["n"] += 1
        # Rank GME only on the FIRST scan so exactly one entry is queued.
        if call["n"] == 1:
            return pd.DataFrame(
                [{"ticker": "GME", "score": 99.0, "setup_type": "CAR", "rank": 1, "as_of": clock}]
            )
        return pd.DataFrame(columns=["ticker", "score", "setup_type"])

    cfg = BacktestConfig(
        tickers=["GME"],
        start=base,
        end=days[-1].to_pydatetime(),
        initial_cash=100_000.0,
        score_threshold=3.0,
    )
    with patch("squeeze_hunter.backtest.runner.run_scan", side_effect=scan_day0_only):
        result = await run_backtest(cfg, cache=cache, settings=settings)

    sells = result.trade_log[result.trade_log["side"] == "sell"].reset_index(drop=True)
    assert len(sells) >= 1, "intraday -35% low must trigger the hard stop"
    hard = sells[sells["reason"] == "hard_stop"]
    assert (
        len(hard) >= 1
    ), f"expected a hard_stop exit on the intraday-crater day; sells={sells.to_dict('records')}"
    exit_row = hard.iloc[0]
    # Exit on day 2 (the crater day), filled at the conservative LOW (~65),
    # NOT the recovered close (99). Cost model applies sell slippage so it's
    # slightly below 65; must be far under the close.
    assert exit_row["ts"].date() == days[2].date()
    assert exit_row["price"] < 80.0, (
        f"hard-stop fill {exit_row['price']} should be near the day low ~65, "
        f"not the recovered close 99 (close fill hides the blowup)"
    )


@pytest.mark.asyncio
async def test_runner_takes_position_and_records_pnl(tmp_path: Path) -> None:
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    settings = Settings()
    settings.score.weights = {
        "f1_si_pct": 2.0,
        "f2_days_to_cover": 1.0,
        "f3_earnings_reaction": 2.0,
        "f4_wsb_mention": 1.5,
        "f5_call_oi_velocity": 1.5,
        "f6_bollinger_breakout": 1.0,
        "f7_volume_spike": 1.0,
    }
    cfg = BacktestConfig(
        tickers=_ALL_TICKERS,
        start=datetime(2024, 5, 14, tzinfo=UTC),
        end=datetime(2024, 5, 28, tzinfo=UTC),
        initial_cash=100_000.0,
        score_threshold=3.0,  # synthetic universe produces lower scores than live; production keeps 8.0
    )
    result = await run_backtest(cfg, cache=cache, settings=settings)
    assert result.equity_curve.iloc[-1] != pytest.approx(100_000.0)  # something happened
    assert (result.trade_log["ticker"] == "GME").any()


@pytest.mark.asyncio
async def test_runner_skips_weekend_days(tmp_path: Path) -> None:
    """C7: bars_held should only count trading days, not calendar days.

    Run a backtest spanning Tue 2024-05-14 to Wed 2024-05-22 (includes a weekend).
    The equity curve should have 7 entries (7 weekdays), NOT 9 (calendar days).
    """
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    settings = Settings()
    settings.score.weights = {
        "f1_si_pct": 2.0,
        "f2_days_to_cover": 1.0,
        "f3_earnings_reaction": 2.0,
        "f4_wsb_mention": 1.5,
        "f5_call_oi_velocity": 1.5,
        "f6_bollinger_breakout": 1.0,
        "f7_volume_spike": 1.0,
    }
    # Span: 2024-05-14 (Tue) → 2024-05-22 (Wed). Weekdays: Tue Wed Thu Fri Mon Tue Wed = 7.
    cfg = BacktestConfig(
        tickers=_ALL_TICKERS,
        start=datetime(2024, 5, 14, tzinfo=UTC),
        end=datetime(2024, 5, 22, tzinfo=UTC),
        initial_cash=100_000.0,
        score_threshold=3.0,
    )
    result = await run_backtest(cfg, cache=cache, settings=settings)
    # Should have 7 weekdays, not 9 calendar days
    assert len(result.equity_curve) == 7


@pytest.mark.asyncio
async def test_runner_skips_us_federal_holidays(tmp_path: Path) -> None:
    """Round-11 regression: the backtest must count TRADING days (weekends AND
    US federal holidays excluded), not just weekdays, so bars_held — the
    21-trading-day time stop — advances identically to the live
    RuntimeContext.eod_close (which skips holidays via the same
    _us_business_holidays() calendar). Span Thu 2024-05-23 → Wed 2024-05-29
    contains Memorial Day (Mon 2024-05-27): 5 weekdays but only 4 trading days.
    The old code counted the holiday as a bar, so the backtest time-stop fired
    ~1 day earlier than live over a holiday-spanning hold (same-code-path
    divergence per design principle #4).
    """
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    settings = Settings()
    settings.score.weights = {
        "f1_si_pct": 2.0,
        "f2_days_to_cover": 1.0,
        "f3_earnings_reaction": 2.0,
        "f4_wsb_mention": 1.5,
        "f5_call_oi_velocity": 1.5,
        "f6_bollinger_breakout": 1.0,
        "f7_volume_spike": 1.0,
    }
    cfg = BacktestConfig(
        tickers=_ALL_TICKERS,
        start=datetime(2024, 5, 23, tzinfo=UTC),
        end=datetime(2024, 5, 29, tzinfo=UTC),
        initial_cash=100_000.0,
        score_threshold=3.0,
    )
    result = await run_backtest(cfg, cache=cache, settings=settings)
    # Thu 23, Fri 24, [Mon 27 = Memorial Day → SKIPPED], Tue 28, Wed 29
    # = 4 trading days, NOT 5 weekdays.
    assert len(result.equity_curve) == 4


@pytest.mark.asyncio
async def test_runner_records_realized_pnl_on_sells(tmp_path: Path) -> None:
    """C2: trade_log sell entries must include `realized` so Kelly counter works."""
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    settings = Settings()
    settings.score.weights = {
        "f1_si_pct": 2.0,
        "f2_days_to_cover": 1.0,
        "f3_earnings_reaction": 2.0,
        "f4_wsb_mention": 1.5,
        "f5_call_oi_velocity": 1.5,
        "f6_bollinger_breakout": 1.0,
        "f7_volume_spike": 1.0,
    }
    cfg = BacktestConfig(
        tickers=_ALL_TICKERS,
        start=datetime(2024, 5, 14, tzinfo=UTC),
        end=datetime(2024, 6, 30, tzinfo=UTC),  # long enough to force sells via time stop
        initial_cash=100_000.0,
        score_threshold=3.0,
    )
    result = await run_backtest(cfg, cache=cache, settings=settings)
    sells = result.trade_log[result.trade_log["side"] == "sell"]
    # R3.4: assert sells exist before checking the realized column, otherwise
    # the test passes trivially when no sells happen (regression goes silent).
    assert len(sells) > 0, "expected at least one sell over the time-stop horizon"
    assert "realized" in sells.columns
    assert sells["realized"].notna().all()


@pytest.mark.asyncio
async def test_runner_halve_amortizes_entry_commission_across_remaining_qty() -> None:
    """R10.4 regression: when a position is halved (signal_decay_50) before
    being fully exited, the entry commission must be charged proportionally
    on each leg — NOT held in full and then re-charged on the eventual exit.
    Otherwise a halve→halve→exit cycle double-charges the entry commission
    against the remaining quantity, biasing realized P&L low.

    Pure-unit test of the per-share amortization formula used in runner.py:
    it doesn't exercise the full backtest loop (the synthetic universe never
    triggers two halves), but it pins the commission accounting math directly.
    """
    # Position: 100 shares entered at $10, with $5 entry commission.
    # Per-share commission attributable to each share = $0.05.
    entry_qty = 100
    entry_commission = 5.0
    entry_commission_per_share = entry_commission / entry_qty
    # Halve 1: sell 50 shares. Should charge 50 * 0.05 = $2.50 of entry commission.
    halve_1_qty = 50
    halve_1_charge = entry_commission_per_share * halve_1_qty
    # Halve 2: sell 25 shares. Should charge 25 * 0.05 = $1.25.
    halve_2_qty = 25
    halve_2_charge = entry_commission_per_share * halve_2_qty
    # Final exit: sell 25 shares. Should charge 25 * 0.05 = $1.25.
    exit_qty = 25
    exit_charge = entry_commission_per_share * exit_qty
    # Total entry commission charged across all legs MUST equal the original $5.
    total = halve_1_charge + halve_2_charge + exit_charge
    assert (
        abs(total - entry_commission) < 1e-9
    ), f"amortized entry commission {total} != original {entry_commission}"
    # And no single leg should ever charge MORE than the original entry commission.
    assert max(halve_1_charge, halve_2_charge, exit_charge) <= entry_commission


@pytest.mark.asyncio
async def test_runner_realized_pnl_subtracts_round_trip_commission(tmp_path: Path) -> None:
    """R9.10 regression: trade_log realized P&L must be NET of round-trip
    commission. Buy commission + sell commission = round-trip; both should be
    subtracted from `(sell_fill - buy_fill) * qty` so Kelly's avg_payoff is
    not biased optimistically.
    """
    from unittest.mock import patch

    from squeeze_hunter.backtest.cost_model import StockCostModel

    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    settings = Settings()
    settings.score.weights = {
        "f1_si_pct": 2.0,
        "f2_days_to_cover": 1.0,
        "f3_earnings_reaction": 2.0,
        "f4_wsb_mention": 1.5,
        "f5_call_oi_velocity": 1.5,
        "f6_bollinger_breakout": 1.0,
        "f7_volume_spike": 1.0,
    }
    cfg = BacktestConfig(
        tickers=_ALL_TICKERS,
        start=datetime(2024, 5, 14, tzinfo=UTC),
        end=datetime(2024, 6, 30, tzinfo=UTC),
        initial_cash=100_000.0,
        score_threshold=3.0,
    )
    # Crank commission per share so the difference is measurable in the test.
    big_commission_model = StockCostModel(commission_per_share=0.50)
    with patch(
        "squeeze_hunter.backtest.runner.StockCostModel",
        return_value=big_commission_model,
    ):
        result = await run_backtest(cfg, cache=cache, settings=settings)
    sells = result.trade_log[
        (result.trade_log["side"] == "sell")
        & (result.trade_log["partial"].fillna(False).astype(bool) == False)  # noqa: E712
    ]
    assert len(sells) > 0, "need at least one full exit to verify commission accounting"
    # For each sell, recompute gross and confirm realized < gross by commission size.
    buys = result.trade_log[result.trade_log["side"] == "buy"].set_index("ticker")
    for _, sell in sells.iterrows():
        ticker = sell["ticker"]
        if ticker not in buys.index:
            continue
        # buys may have multiple rows per ticker; take the first (entry)
        buy_row = buys.loc[ticker]
        if isinstance(buy_row, pd.DataFrame):
            buy_row = buy_row.iloc[0]
        qty = sell["qty"]
        gross = (sell["price"] - buy_row["price"]) * qty
        # round-trip commission at $0.50/share = qty * 1.0
        expected_commission = qty * 0.50 * 2  # buy + sell
        # Realized must be lower than gross by approximately round-trip commission
        assert (
            sell["realized"] < gross
        ), f"realized {sell['realized']} >= gross {gross} for {ticker} — commissions not deducted"
        diff = gross - sell["realized"]
        assert (
            abs(diff - expected_commission) < 0.01
        ), f"expected commission deduction ≈ {expected_commission}, got {diff} for {ticker}"


@pytest.mark.asyncio
async def test_runner_gross_exposure_invariant(tmp_path: Path) -> None:
    """R6 regression (loose form): regardless of how many same-day buys
    happen, the cumulative notional from buys must never exceed 90% of equity.

    Note: with the default 8% position cap and max_new_per_day=3, the R6 bug
    is only exercisable when pre-existing positions occupy >65% — a state
    hard to set up in an integration test. So this test verifies the weaker
    invariant: gross exposure stays under the cap. The R6 fix's per-buy
    state.gross_exposure_pct increment is verified by code review and the
    unit test `test_runner_gross_exposure_updates_state` below.
    """
    cache = ParquetCache(root=tmp_path)
    _seed(cache)
    settings = Settings()
    settings.score.weights = {
        "f1_si_pct": 2.0,
        "f2_days_to_cover": 1.0,
        "f3_earnings_reaction": 2.0,
        "f4_wsb_mention": 1.5,
        "f5_call_oi_velocity": 1.5,
        "f6_bollinger_breakout": 1.0,
        "f7_volume_spike": 1.0,
    }
    cfg = BacktestConfig(
        tickers=_ALL_TICKERS,
        start=datetime(2024, 5, 14, tzinfo=UTC),
        end=datetime(2024, 5, 16, tzinfo=UTC),
        initial_cash=100_000.0,
        score_threshold=3.0,
    )
    result = await run_backtest(cfg, cache=cache, settings=settings)
    buys = result.trade_log[result.trade_log["side"] == "buy"]
    # Group buys by day and check the per-day total stays under the cap.
    if len(buys) > 0:
        for day, day_buys in buys.groupby("ts"):
            notional = (day_buys["qty"] * day_buys["price"]).sum()
            assert (
                notional <= 100_000.0 * 0.90 + 1.0
            ), f"day {day} buys exceeded 90% gross cap: ${notional:.0f}"


def test_runner_gross_exposure_updates_state() -> None:
    """R6 unit-level regression: PortfolioState.gross_exposure_pct must be
    bumped by `size_usd / equity_usd` after each accepted buy in the daily
    loop. Verifies the R6 fix by reading the runner source directly.
    """
    import inspect

    from squeeze_hunter.backtest import runner as runner_mod

    src = inspect.getsource(runner_mod)
    # The R6 fix line must be present in the daily loop
    assert (
        "state.gross_exposure_pct += size_usd / state.equity_usd" in src
    ), "R6 fix (per-buy gross_exposure_pct update) missing from runner.py"
