# Multi-stage build — one image for all services
FROM python:3.11-slim AS base

RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chown -R appuser:appuser /app
USER appuser

# Default: run the chat API (orchestrator)
CMD ["uvicorn", "src.chat_api:app", "--host", "0.0.0.0", "--port", "8000"]
