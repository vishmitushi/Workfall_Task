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
import logging
import os
import traceback
import uuid
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

logger = logging.getLogger(__name__)

from src.orchestrator import (
    AgentState,
    build_graph,
    create_checkpointer,
    discover_sub_agent,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

app = FastAPI(title="ShopHub Support Chat API")
_bearer_scheme = HTTPBearer()

SUB_AGENT_URL = os.getenv("SUB_AGENT_URL", "http://localhost:8002")

_checkpointer = None
_graph = None


@app.on_event("startup")
async def startup():
    global _checkpointer, _graph
    # R2: discover the sub-agent by fetching its agent card
    try:
        await discover_sub_agent()
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(
            "Could not discover sub-agent (will use defaults): %s", exc
        )
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


def _extract_text(content) -> str:
    """Extract plain text from LLM content (handles both str and Anthropic list format)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content) if content else ""


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
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
):
    """Stream a chat response over SSE."""
    token = credentials.credentials

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

        streamed = False  # track whether on_chat_model_stream fired

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
                        text = _extract_text(chunk.content)
                        if text:
                            streamed = True
                            yield {
                                "event": "token",
                                "data": json.dumps({"content": text}),
                            }
                elif kind == "on_chat_model_end":
                    if streamed:
                        streamed = False
                        continue
                    output = event.get("data", {}).get("output")
                    if output and hasattr(output, "content") and output.content:
                        text = _extract_text(output.content)
                        if text:
                            yield {
                                "event": "token",
                                "data": json.dumps({"content": text}),
                            }
                elif kind == "on_chain_end" and name == "proposal":
                    state = await _graph.aget_state(config)
                    proposal = state.values.get("pending_proposal")
                    if proposal:
                        yield {
                            "event": "proposal",
                            "data": json.dumps(proposal),
                        }
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("SSE stream error: %s\n%s", exc, tb)
            yield {
                "event": "error",
                "data": json.dumps({"error": str(exc), "traceback": tb}),
            }

        yield {"event": "done", "data": json.dumps({"thread_id": thread_id})}

    return EventSourceResponse(event_generator())


@app.post("/chat/approve")
async def approve_or_reject(
    body: ApprovalRequest,
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
):
    """Resume a paused conversation with approve/reject.

    Uses Command(resume=...) to tell LangGraph to continue from the
    interrupt_before point (the 'approval' node) instead of restarting
    the graph from scratch.
    """
    token = credentials.credentials

    auth_info = await _resolve_auth(token)
    merchant = auth_info.get("merchant")
    if not merchant:
        raise HTTPException(status_code=403, detail="Token has no merchant scope")

    config = {"configurable": {"thread_id": body.thread_id}}

    from langchain_core.messages import HumanMessage
    from langgraph.types import Command

    resume_cmd = Command(
        resume=body.action,
        update={
            "messages": [HumanMessage(content=body.action)],
            "bearer_token": token,
            "merchant": merchant,
        },
    )

    async def event_generator():
        try:
            async for chunk in _graph.astream(
                resume_cmd, config=config, stream_mode="updates"
            ):
                logger.info("Approve chunk: %s", chunk)
                for node_output in chunk.values():
                    msgs = node_output.get("messages", [])
                    for msg in msgs:
                        if hasattr(msg, "content") and msg.content:
                            yield {
                                "event": "token",
                                "data": json.dumps({"content": msg.content}),
                            }
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("SSE approve error: %s\n%s", exc, tb)
            yield {
                "event": "error",
                "data": json.dumps({"error": str(exc), "traceback": tb}),
            }
        yield {"event": "done", "data": json.dumps({"thread_id": body.thread_id})}

    return EventSourceResponse(event_generator())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
