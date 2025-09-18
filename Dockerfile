FROM ghcr.io/astral-sh/uv:python3.12-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

# Only copy metadata first to leverage Docker layer caching
COPY pyproject.toml README.md ./

# Install dependencies into a local virtual environment managed by uv
RUN uv sync --no-dev

# Copy application code
COPY app ./app
COPY consts.py ./

# Tunables (override in Fly)
ENV PORT=8000 \
    WEB_CONCURRENCY=1 \
    MAX_CONCURRENCY=200

EXPOSE 8000

# Run via uv to ensure the synced environment is used
CMD ["sh", "-c", "uv run uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --loop uvloop --http httptools --proxy-headers --forwarded-allow-ips='*' --limit-concurrency ${MAX_CONCURRENCY} --workers ${WEB_CONCURRENCY}"]

