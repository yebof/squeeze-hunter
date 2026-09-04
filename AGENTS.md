# AGENTS.md — project-level guidance for Codex

This is a **production-bound quantitative trading system**. Bugs cost real money.
Treat correctness as a hard constraint, never a target.

## Design principles (load-bearing)

These were established during brainstorming and have governed every fix in the
codebase. They override Codex's defaults when they conflict.

1. **Conservative bias.** Smaller Kelly fractions, tighter stops, lower position
   caps, fewer concurrent positions. When choosing between two parameters,
   prefer the more conservative one unless backtest evidence says otherwise.
2. **Fewer high-quality data sources** beats many low-quality ones. Drop
   "placeholder" factors that depend on paid data we don't have. Re-introduce
   only when the data is actually plugged in.
3. **Simple, stable, clear methods.** Linear weighted z-score over ML;
   rule-based classifier over learned; fixed weights from backtest over adaptive
   online learning; single subreddit over multi-source sentiment.
4. **Same code path for backtest / paper / live.** Divergence is a bug — the
   `IBroker` and `DataProvider` Protocols exist precisely to enforce this.
5. **One config file (`config/settings.example.yml`) for all tuning.** No
   scattered constants in code.

## Style

- Write **TDD-style fixes**: write the failing test first, run it, then
  implement, then run it again. The fix-then-write-test order has caught
  fewer bugs in this codebase historically.
- Each bug fix gets a **regression test** named after the bug or its review
  finding (e.g., `test_lifecycle_skips_position_when_quote_is_zero` for I2).
- **Don't catch `Exception` broadly.** Catch the specific exceptions you expect
  to handle (`ConnectionError`, `TimeoutError`, `OSError` for transient broker
  errors). Programming errors (`AttributeError`, `NotImplementedError`,
  `TypeError`) MUST propagate up to `tick_safe` so they appear in structured
  logs.
- **Time:** internal timestamps are UTC. Display layers convert to ET. Backtest
  loops over **trading days** (`pd.bdate_range`), not calendar days.
- **Async:** never block the event loop with synchronous I/O. Wrap PRAW /
  yfinance / `_ib.disconnect()` etc. in `asyncio.to_thread`.
- **Pre-commit hooks must pass without `--no-verify`.** ruff, ruff-format, and
  ty run on every commit; pytest runs on pre-push. Fix the underlying issue
  rather than skipping.

## Areas with known subtleties

- **`signals/normalize.cross_sectional_z`** — drops NaN/Inf before computing
  mean/std so a single bad ticker doesn't pollute the universe.
- **`backtest/runner.py` daily loop** — uses `pd.bdate_range` (trading days),
  not `+= timedelta(days=1)` (calendar days). The 21-day time stop counts
  trading days. Don't change this without re-reading the C7 commit.
- **Kelly priors** — per-setup-type via `kelly_priors_for_setup`. CAR is
  (0.25, 3.5), GME is (0.15, 8.0), Mixed is (0.20, 5.5). All produce positive
  raw Kelly. Don't fall back to fixed (0.20, 2.0) — that's negative and
  bypasses the model.
- **Killswitch telemetry** — the 5 inputs come from `PortfolioTelemetry`,
  populated by `RuntimeContext.tick`. Never feed all-zero inputs to
  `evaluate_killswitch` (that disables the safety layer).
- **`f1_si_pct`** — depends on `si_pct_float` being merged from a Yahoo float
  lookup at INGEST time (see `ingest/backfill_finra.py`). The signal returns 0
  for tickers with `si_pct_float <= 0`, so old parquet data without the merge
  produces a dead factor. Re-run ingest after pulling the C5 fix.
- **`f5_call_oi_velocity`** — reads the 5-trading-day-prior chain via
  `provider.fetch_option_chain_at`. If you don't have an options-chain ingest
  job (Phase 4), the signal returns 0 in backtest. That's accepted.

## Phase status

- ✅ Phase 0–3 complete and tagged.
- 355 tests passing.
- Two review rounds + 30+ bug fixes have hardened the codebase.
- Next user-side step: end-to-end backtest + Gate 1 evaluation. Then paper
  trading for 30 days. Phase 4 (small live) only after Gate 2 passes.

## Files I don't want regenerated

- `docs/superpowers/specs/2026-05-10-squeeze-hunter-design.md` is the source
  of truth for design decisions. Edit, don't replace.
- `docs/superpowers/plans/*.md` are historical implementation plans. Keep
  intact.

## When asked to add features

- Follow the plan structure: design → spec update → implementation plan →
  TDD fixes. Don't skip stages.
- New factors go in `signals/`, exposed via `compute_all_factors` and weighted
  in `config/settings.example.yml`. Don't put them inline in `score/combiner.py`.
- New data sources implement `DataProvider`. Don't extend an existing provider
  to do double duty.

## When asked to change risk parameters

- Change `config/settings.example.yml` first. Re-run backtest. Then propagate
  the same value into `risk/kelly.py` / `risk/gates.py` defaults if the YAML
  override is meant to be permanent.
- Document the change rationale in the commit message.
