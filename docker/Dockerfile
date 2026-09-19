# squeeze-hunter application image (P10).
# Build context is the repository root:  docker build -f docker/Dockerfile .
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --all-extras --frozen --no-dev

COPY config ./config
COPY alembic.ini ./
COPY alembic ./alembic

ENV PATH="/opt/venv/bin:${PATH}"
# Runtime state lives on the mounted volume, never in the image.
VOLUME ["/app/data"]

# The /health endpoint answers 503 while unhealthy or while the killswitch is
# active; the compose healthcheck reads it.
HEALTHCHECK --interval=60s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

CMD ["squeeze-hunter", "paper"]
