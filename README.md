# ShopHub Support Agent

Multi-tenant AI support assistant for e-commerce merchants, built with **Python 3.11**, **LangGraph**, and **FastAPI**.

## Quick start

### With Docker (recommended)

```bash
# Clone and run — no API key needed (fake-LLM mode)
docker compose up --build

# Services:
#   http://localhost:8000  — Chat API (orchestrator + SSE)
#   http://localhost:8001  — Mock Store API
#   http://localhost:8002  — Orders Sub-Agent (JSON-RPC)
```

### Without Docker

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .venv\Scripts\activate    # Windows

pip install -r requirements.txt

# Terminal 1: Store API
python -m src.store_api                          # port 8001

# Terminal 2: Orders sub-agent
python -m src.orders_agent                       # port 8002

# Terminal 3: Chat API (orchestrator)
python -m src.chat_api                           # port 8000
```

## Running tests (no API key needed)

```bash
# All tests run in fake-LLM mode
pytest tests/ -v
```

## Running the eval harness

```bash
# Fake-LLM mode (default, no API key)
python -m evals.run_evals

# With a real model
LLM_PROVIDER=anthropic ANTHROPIC_API_KEY=sk-... python -m evals.run_evals
```

## Fake-LLM mode

```bash
# Set LLM_PROVIDER=fake (or leave unset — fake is the default)
export LLM_PROVIDER=fake
docker compose up --build
```

All tests, evals, and the demo client work with no API key.

## Using a real model

```bash
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
docker compose up --build
```

## Demo client (SSE streaming)

```bash
python demo_client.py
```

Demonstrates: order queries, order lookup, refund proposal, and approval — all streamed over SSE.

## Model provider

**Anthropic Claude** (via `langchain-anthropic`). The LLM provider is swapped by a single factory function in `src/llm_factory.py` — set `LLM_PROVIDER` to `fake` or `anthropic`.

## Project structure

```
src/
  store_api.py       — R0: Mock store API (FastAPI, in-memory)
  orders_agent.py    — R2: Orders sub-agent (JSON-RPC 2.0)
  orchestrator.py    — R1/R5: LangGraph agent + human-in-the-loop
  chat_api.py        — R6/R7: SSE streaming + SQLite persistence
  guardrails.py      — R4: Prompt injection + PII redaction
  llm_factory.py     — LLM factory function
  fake_llm.py        — Scripted fake LLM for offline use
tests/
  test_tenant_isolation.py  — R3/R8: Tenant isolation tests
  test_proposal_flow.py     — R5/R8: Proposal + guardrail tests
evals/
  run_evals.py       — R9: Mini evaluation harness (6 cases)
```

See [DESIGN.md](DESIGN.md) for architecture, trade-offs, and what was cut.
