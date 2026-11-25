
FROM python:3.11-slim

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --no-install-project

COPY src/ ./src/
COPY config/ ./config/
COPY artifacts/models/ ./artifacts/models/
COPY README.md ./


RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

EXPOSE 8000

CMD ["uvicorn", "telecom_churn.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
