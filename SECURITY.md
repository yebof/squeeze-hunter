# Security

## Scope

squeeze-hunter is an educational and research project. It is **not** an
investment product and comes with no warranty (see the Apache-2.0 `LICENSE`
and the disclaimer in `README.md`). That said, it can place real orders when
run with `live --confirm-real-money`, so defects in the risk, execution or
broker layers matter.

## Reporting a vulnerability

Please do not open a public issue for anything that could let an attacker
move money, leak credentials or bypass the risk controls. Use GitHub's private
vulnerability reporting on this repository ("Security" tab → "Report a
vulnerability"). Include the affected file, a reproduction and the impact you
believe it has.

Ordinary bugs (wrong signal math, a stop that fires late in backtest, a
documentation mismatch) belong in a normal issue or pull request.

## Operating safely

- Never commit `.env`; it is gitignored and the CLI loads it from the working
  directory. Keep `IBKR_*`, `FINNHUB_KEY`, `REDDIT_*`, `TELEGRAM_*` and
  `SLACK_WEBHOOK_URL` out of shell history and logs.
- Run paper mode (`squeeze-hunter paper`, IBKR port 7497) until Gate 1 and
  Gate 2 have passed. Live mode requires an explicit flag on purpose.
- The `docker/compose.yml` stack uses throwaway local credentials and binds
  ports on the host; do not deploy it unchanged to an internet-reachable
  machine.
- The `/metrics` and `/health` endpoint has no authentication. Bind it to
  localhost (`monitor.http_host`) unless a scraper on another host needs it.
