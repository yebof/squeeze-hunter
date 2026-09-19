"""A deterministic synthetic universe for golden-number tests (P7).

Ten tickers, two years of NYSE sessions, seeded random walks with a handful
of scripted "squeeze episodes" (elevated short interest, an earnings report
the night before, a gap-up on 8x volume). Everything derives from one seed,
so the pipeline's Gate 1 metrics over it are reproducible to the last digit
and any change in the runner, the cost model, a factor or a metric must
update `golden/expected.json` explicitly.
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd

from squeeze_hunter.data.cache import ParquetCache
from squeeze_hunter.trading_calendar import trading_sessions

TICKERS = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH", "III", "JJJ"]
START = datetime(2023, 1, 2, tzinfo=UTC)
END = datetime(2024, 12, 31, tzinfo=UTC)
SEED = 20260920

# (ticker, episode session index) — the gap-up day; earnings the evening before.
EPISODES: list[tuple[str, int]] = [
    ("AAA", 60),
    ("CCC", 140),
    ("EEE", 230),
    ("BBB", 310),
    ("GGG", 390),
    ("DDD", 450),
]


def build_synthetic_universe(cache: ParquetCache) -> list[str]:
    rng = np.random.default_rng(SEED)
    sessions = [d.to_pydatetime() + timedelta(hours=5) for d in trading_sessions(START, END)]
    n = len(sessions)
    episode_by_ticker = {t: i for t, i in EPISODES}

    si_rows: list[dict] = []
    earnings_rows: list[dict] = []
    for k, t in enumerate(TICKERS):
        base = 8.0 + 3.0 * k
        drift = rng.normal(0.0002, 0.0001)
        vol = 0.02 + 0.005 * (k % 3)
        rets = rng.normal(drift, vol, size=n)
        volumes = rng.lognormal(mean=np.log(2_000_000 + 300_000 * k), sigma=0.25, size=n)
        ep = episode_by_ticker.get(t)
        if ep is not None:
            rets[ep] = 0.30
            rets[ep + 1 : ep + 4] = [0.06, 0.02, -0.03]
            rets[ep + 4 : ep + 15] = rng.normal(-0.006, 0.02, size=11)
            volumes[ep] *= 8.0
            volumes[ep + 1] *= 4.0
        closes = base * np.cumprod(1 + rets)
        rows = []
        for i, ts in enumerate(sessions):
            c = float(closes[i])
            prev = float(closes[i - 1]) if i > 0 else c
            gap = 0.22 if (ep is not None and i == ep) else float(rng.normal(0, 0.003))
            o = prev * (1 + gap)
            hi = max(o, c) * (1 + abs(float(rng.normal(0, 0.006))))
            lo = min(o, c) * (1 - abs(float(rng.normal(0, 0.006))))
            rows.append(
                {
                    "ticker": t,
                    "ts": ts,
                    "open": round(o, 4),
                    "high": round(hi, 4),
                    "low": round(lo, 4),
                    "close": round(c, 4),
                    "volume": int(volumes[i]),
                }
            )
        cache.write_partition("bars", t, pd.DataFrame(rows))

        # Biweekly short interest: 15th and month end. Elevated for ~60 days
        # before an episode.
        float_shares = 50_000_000 + 5_000_000 * k
        for d in pd.date_range(START, END, freq="SME"):
            settlement = d.date()
            si_pct = 0.04 + 0.02 * (k % 4)
            if ep is not None:
                ep_date = sessions[ep].date()
                if ep_date - timedelta(days=75) <= settlement <= ep_date + timedelta(days=5):
                    si_pct = 0.35 + 0.05 * (k % 3)
            adv = int(volumes.mean())
            si_rows.append(
                {
                    "ticker": t,
                    "settlement_date": settlement,
                    "si_shares": int(si_pct * float_shares),
                    "si_pct_float": si_pct,
                    "avg_daily_volume_20d": adv,
                }
            )

        # Quarterly earnings after the close; an extra report the evening
        # before each episode.
        for q_month in (2, 5, 8, 11):
            for year in (2023, 2024):
                report_day = date(year, q_month, 10 + k)
                earnings_rows.append(
                    {
                        "ticker": t,
                        "report_at": datetime.combine(report_day, datetime.min.time(), tzinfo=UTC)
                        + timedelta(hours=21),
                        "actual_eps": 0.5,
                        "estimate_eps": 0.4,
                    }
                )
        if ep is not None:
            eve = sessions[ep - 1].date()
            earnings_rows.append(
                {
                    "ticker": t,
                    "report_at": datetime.combine(eve, datetime.min.time(), tzinfo=UTC)
                    + timedelta(hours=21),
                    "actual_eps": 1.2,
                    "estimate_eps": 0.4,
                }
            )

    cache.write_partition("short_interest", "all", pd.DataFrame(si_rows))
    cache.write_partition("earnings", "all", pd.DataFrame(earnings_rows))
    return list(TICKERS)
