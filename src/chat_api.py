"""R6/R7 — Chat API with SSE streaming and SQLite persistence.

Endpoints:
  POST /chat          — send a message, get back SSE stream
  POST /chat/approve  — approve a pending proposal
  POST /chat/reject   — reject a pending proposal
  GET  /health        — healthcheck

The bearer token from the Authorization header is forwarded through the
entire chain: chat API → orchestrator → sub-agent → store API.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Optional

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from src.orchestrator import (
    AgentState,
    build_graph,
    create_checkpointer,
)

app = FastAPI(title="ShopHub Support Chat API")

SUB_AGENT_URL = os.getenv("SUB_AGENT_URL", "http://localhost:8002")

_checkpointer = None
_graph = None


@app.on_event("startup")
async def startup():
    global _checkpointer, _graph
    _checkpointer = await create_checkpointer()
    _graph = build_graph(checkpointer=_checkpointer)


# ---------------------------------------------------------------------------
# Auth helper — resolve merchant from token via store API
# ---------------------------------------------------------------------------

async def _resolve_auth(token: str) -> dict:
    store_url = os.getenv("STORE_API_URL", "http://localhost:8001")
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{store_url}/auth/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid token")
        return resp.json()


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None


class ApprovalRequest(BaseModel):
    thread_id: str
    action: str = "approve"  # "approve" or "reject"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(
    body: ChatRequest,
    authorization: Optional[str] = Header(None),
):
    """Stream a chat response over SSE."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    token = authorization.replace("Bearer ", "")

    auth_info = await _resolve_auth(token)
    merchant = auth_info.get("merchant")

    # R4(c): fail closed — no merchant means reject
    if not merchant:
        raise HTTPException(
            status_code=403,
            detail="Your token is not associated with any merchant.",
        )

    thread_id = body.thread_id or str(uuid.uuid4())

    config = {"configurable": {"thread_id": thread_id}}

    from langchain_core.messages import HumanMessage

    input_state = {
        "messages": [HumanMessage(content=body.message)],
        "bearer_token": token,
        "merchant": merchant,
    }

    async def event_generator():
        yield {"event": "thread_id", "data": json.dumps({"thread_id": thread_id})}

        try:
            async for event in _graph.astream_events(
                input_state, config=config, version="v2"
            ):
                kind = event.get("event", "")
                name = event.get("name", "")

                if kind == "on_tool_start":
                    yield {
                        "event": "tool_start",
                        "data": json.dumps({"tool": name}),
                    }
                elif kind == "on_tool_end":
                    yield {
                        "event": "tool_end",
                        "data": json.dumps({"tool": name}),
                    }
                elif kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield {
                            "event": "token",
                            "data": json.dumps({"content": chunk.content}),
                        }
                elif kind == "on_chain_end" and name == "proposal":
                    # The graph has interrupted for human approval
                    state = await _graph.aget_state(config)
                    proposal = state.values.get("pending_proposal")
                    if proposal:
                        yield {
                            "event": "proposal",
                            "data": json.dumps(proposal),
                        }
        except Exception as exc:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(exc)}),
            }

        yield {"event": "done", "data": json.dumps({"thread_id": thread_id})}

    return EventSourceResponse(event_generator())


@app.post("/chat/approve")
async def approve_or_reject(
    body: ApprovalRequest,
    authorization: Optional[str] = Header(None),
):
    """Resume a paused conversation with approve/reject."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    token = authorization.replace("Bearer ", "")

    auth_info = await _resolve_auth(token)
    merchant = auth_info.get("merchant")
    if not merchant:
        raise HTTPException(status_code=403, detail="Token has no merchant scope")

    config = {"configurable": {"thread_id": body.thread_id}}

    from langchain_core.messages import HumanMessage

    input_state = {
        "messages": [HumanMessage(content=body.action)],
        "bearer_token": token,
        "merchant": merchant,
    }

    async def event_generator():
        try:
            async for event in _graph.astream_events(
                input_state, config=config, version="v2"
            ):
                kind = event.get("event", "")
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield {
                            "event": "token",
                            "data": json.dumps({"content": chunk.content}),
                        }
        except Exception as exc:
            yield {
                "event": "error",
                "data": json.dumps({"error": str(exc)}),
            }
        yield {"event": "done", "data": json.dumps({"thread_id": body.thread_id})}

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
