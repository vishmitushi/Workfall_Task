# DESIGN.md — ShopHub Support Agent

## Architecture

```
┌──────────────┐    SSE/HTTP     ┌──────────────────┐   JSON-RPC 2.0   ┌────────────────┐
│              │ ──────────────> │                  │ ───────────────> │                │
│  Client      │                │  Chat API        │                  │  Orders Agent  │
│  (curl/py)   │ <───────────── │  (Orchestrator)  │ <─────────────── │  (Sub-agent)   │
│              │    SSE events  │  Port 8000       │   JSON-RPC resp  │  Port 8002     │
└──────────────┘                └──────────────────┘                  └────────────────┘
                                        │                                     │
                                        │ Bearer token forwarded              │ Bearer token
                                        │ unchanged through chain             │ forwarded
                                        ▼                                     ▼
                                ┌──────────────────┐               ┌────────────────┐
                                │  SQLite          │               │  Store API     │
                                │  (checkpoints)   │               │  (Mock)        │
                                │                  │               │  Port 8001     │
                                └──────────────────┘               └────────────────┘
```

Three processes, one shared Docker image:

| Service        | Port | Role                                               |
|----------------|------|-----------------------------------------------------|
| `store-api`    | 8001 | Mock e-commerce backend (FastAPI, in-memory)        |
| `orders-agent` | 8002 | Domain sub-agent, JSON-RPC 2.0, agent card at `/.well-known/agent.json` |
| `chat-api`     | 8000 | LangGraph orchestrator + SSE streaming endpoint     |

## Flow of one turn

1. **Client** sends `POST /chat` with `{"message": "Show me paid orders"}` and `Authorization: Bearer tok-merchant-a`.
2. **Chat API** validates the token by calling `GET /auth/me` on the store API. If the token has no merchant, it rejects (fail-closed, R4c).
3. **Chat API** feeds the message into the LangGraph graph with the bearer token in state.
4. **Agent node** calls the LLM (real or fake). The LLM decides to call `list_orders(status="PAID")`.
5. **Tool execution node** translates the tool call into a JSON-RPC 2.0 request to the orders sub-agent at `POST /jsonrpc`, forwarding the bearer token unchanged.
6. **Orders sub-agent** calls `GET /orders?status=PAID` on the store API with the same bearer token. The store API filters to the caller's merchant.
7. **Guardrails** sanitise the response: emails/phones redacted, injection-like customer notes neutralised.
8. **Tool result** flows back to the LLM, which generates a human-readable response.
9. **SSE stream** emits `tool_start`, `tool_end`, `token`, and `done` events to the client.

### Refund/cancel flow (R5)

- Steps 1–7 are the same but the LLM issues a `lookup_order` first.
- After seeing the order, the LLM emits a **proposal** (JSON with `type: "proposal"`).
- The graph **interrupts** (LangGraph `interrupt_before=["approval"]`).
- The SSE stream emits a `proposal` event. The client shows it to the user.
- The user sends `POST /chat/approve` with `{"action": "approve"}` or `"reject"`.
- The graph resumes from the **approval node**, which either executes the action via JSON-RPC or returns a rejection message.

## Key trade-offs

| Decision | Why | Downside |
|----------|-----|----------|
| **SQLite** instead of PostgreSQL | Zero infrastructure, single-file, simpler Docker setup, adequate for demo | Not production-grade for concurrent writes; would switch to PostgreSQL for real use |
| **Regex-based injection detection** | Simple, deterministic, fast, no ML model needed | Won't catch novel prompt injections; a production system would use a classifier or LLM-as-judge |
| **Fake LLM with pattern matching** | Runs offline with no API key, deterministic for tests/evals | Only covers scripted paths; can't handle novel queries |
| **Single Docker image** for all services | Less build time, simpler Compose file | Not ideal for independent scaling; would split in production |
| **Hand-rolled JSON-RPC** instead of a2a-sdk | Fewer dependencies, full control, easy to understand | Less standard; would use a2a-sdk in a larger system |

## What was cut and why

- **Web UI** — explicitly out of scope; demo client script covers SSE.
- **PostgreSQL** — SQLite justified above; the switch is a one-line change in the checkpointer.
- **Multiple LLM providers** — only Anthropic + fake; the factory function makes adding more trivial.
- **RAG / vector store** — out of scope per instructions.
- **Token/cost accounting (S2)** — chose not to do a stretch goal to keep core solid.
- **Real prompt-injection classifier** — regex is a known weakness; would add a small classifier in production.

## Rough cost and latency per turn

| Metric | Fake LLM | Claude Sonnet (real) |
|--------|----------|---------------------|
| Latency (one turn, no tools) | ~5 ms | ~800–1500 ms |
| Latency (one turn + 1 tool call) | ~15 ms | ~2–3 s |
| Input tokens per turn | N/A | ~500–800 |
| Output tokens per turn | N/A | ~100–300 |
| Cost per turn | $0 | ~$0.003–$0.008 |

Estimates assume Claude Sonnet with ~700 input tokens (system prompt + history + tool result) and ~200 output tokens.

## What I would do with another week

1. **PostgreSQL checkpointer** with async connection pooling for production-grade persistence.
2. **LLM-based injection detector** — a small classifier fine-tuned on prompt-injection examples, run as a guardrail before the main LLM sees tool output.
3. **Correlation IDs (S3)** — generate at the chat API, carry through JSON-RPC headers, log with structlog end-to-end.
4. **Per-merchant rate limiting (S2)** — token bucket in Redis, surfaced in SSE stream.
5. **Integration tests** — spin up all three services in pytest, run full end-to-end conversations.
6. **OpenTelemetry tracing** — spans for each hop in the chain, exported to Jaeger.
7. **Separate Docker images** per service for independent scaling.

## AI assistant disclosure

This project was built with the assistance of Claude Code (Anthropic). All code was reviewed, understood, and tested by the author.
