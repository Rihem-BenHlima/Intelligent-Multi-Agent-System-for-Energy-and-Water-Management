FROM python:3.11-slim-bookworm

# System deps needed to compile asyncpg and other C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gcc g++ libpq-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (separate layer — cached unless requirements change).
# Uses a BuildKit cache mount (not --no-cache-dir) so that on a flaky network,
# a retried build reuses whatever wheels a previous failed attempt already
# downloaded instead of re-fetching everything (incl. the ~200MB torch wheel)
# from scratch every time.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --default-timeout=180 --retries 10 -r requirements.txt && \
    pip install --default-timeout=180 --retries 10 --no-deps "xgboost>=1.7.0,<3.0.0"

# Copy application source
COPY . .

# Create mount-point directories (populated via volume mounts at runtime)
RUN mkdir -p /app/data /app/ml_models

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--reload-exclude", "ml_models/*"]
