# squeeze-hunter

Quantitative trading system for short-squeeze events (GME-type and CAR-type).
See `docs/superpowers/specs/2026-05-10-squeeze-hunter-design.md` for the full design.

## Quick start

    uv sync --all-extras
    docker compose -f docker/compose.yml up -d
    uv run squeeze-hunter hello
