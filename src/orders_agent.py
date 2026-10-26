"""R2 — Orders sub-agent exposed over JSON-RPC 2.0 with an agent card.

This runs as a separate process (port 8002).  The orchestrator discovers it
via GET /.well-known/agent.json and calls it via POST /jsonrpc.

Tools offered:
  - list_orders(status?)    → calls GET /orders on the store API
  - lookup_order(order_id)  → calls GET /orders/lookup on the store API
  - refund_order(order_id)  → calls POST /orders/refund on the store API
  - cancel_order(order_id)  → calls POST /orders/cancel on the store API

The bearer token is forwarded from the caller through to the store API.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from src.guardrails import sanitise_order, sanitise_orders

STORE_API_URL = os.getenv("STORE_API_URL", "http://localhost:8001")

app = FastAPI(title="ShopHub Orders Sub-Agent")
_bearer_scheme = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------------
# Agent card (JSON document describing name, skills, endpoint)
# ---------------------------------------------------------------------------

AGENT_CARD = {
    "name": "orders-agent",
    "description": "Manages order queries and actions (refund, cancel) for a merchant.",
    "version": "1.0.0",
    "skills": [
        {"name": "list_orders", "description": "List orders, optionally filtered by status."},
        {"name": "lookup_order", "description": "Look up a single order by ID."},
        {"name": "refund_order", "description": "Refund an order (requires approval)."},
        {"name": "cancel_order", "description": "Cancel an order (requires approval)."},
    ],
    "endpoint": "/jsonrpc",
}


@app.get("/.well-known/agent.json")
def agent_card():
    return AGENT_CARD


# ---------------------------------------------------------------------------
# Tool implementations — each calls the store API with forwarded auth
# ---------------------------------------------------------------------------


async def _store_get(path: str, params: dict, token: str) -> Any:
    async with httpx.AsyncClient() as client:
        resp = await client.get(
            f"{STORE_API_URL}{path}",
            params=params,
            headers={"Authorization": f"Bearer {token}"},
        )
        if resp.status_code >= 400:
            return {"error": resp.json().get("detail", resp.text)}
        return resp.json()


async def _store_post(path: str, body: dict, token: str) -> Any:
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{STORE_API_URL}{path}",
            json=body,
            headers={"Authorization": f"Bearer {token}"},
        )
        if resp.status_code >= 400:
            return {"error": resp.json().get("detail", resp.text)}
        return resp.json()


TOOL_HANDLERS = {}


def tool(name: str):
    def decorator(fn):
        TOOL_HANDLERS[name] = fn
        return fn
    return decorator


@tool("list_orders")
async def handle_list_orders(params: dict, token: str) -> Any:
    status = params.get("status")
    query: dict[str, str] = {}
    if status:
        query["status"] = status
    result = await _store_get("/orders", query, token)
    if isinstance(result, list):
        return sanitise_orders(result)
    return result


@tool("lookup_order")
async def handle_lookup_order(params: dict, token: str) -> Any:
    order_id = params.get("order_id")
    if not order_id:
        return {"error": "order_id is required"}
    result = await _store_get("/orders/lookup", {"id": order_id}, token)
    # Handle legacy 200 + status:false
    if isinstance(result, dict) and result.get("status") is False:
        return {"error": result.get("message", "Order not found")}
    if isinstance(result, dict) and "order" in result:
        result["order"] = sanitise_order(result["order"])
    return result


@tool("refund_order")
async def handle_refund_order(params: dict, token: str) -> Any:
    order_id = params.get("order_id")
    if not order_id:
        return {"error": "order_id is required"}
    result = await _store_post("/orders/refund", {"id": order_id}, token)
    if isinstance(result, dict) and "order" in result:
        result["order"] = sanitise_order(result["order"])
    return result


@tool("cancel_order")
async def handle_cancel_order(params: dict, token: str) -> Any:
    order_id = params.get("order_id")
    if not order_id:
        return {"error": "order_id is required"}
    result = await _store_post("/orders/cancel", {"id": order_id}, token)
    if isinstance(result, dict) and "order" in result:
        result["order"] = sanitise_order(result["order"])
    return result


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 endpoint
# ---------------------------------------------------------------------------


class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: dict = {}
    id: Optional[str | int] = None


def _jsonrpc_error(req_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "error": {"code": code, "message": message}, "id": req_id}


def _jsonrpc_result(req_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "result": result, "id": req_id}


@app.post("/jsonrpc")
async def jsonrpc_endpoint(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
):
    """JSON-RPC 2.0 dispatcher — routes method to the right tool handler."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Missing authorization header")
    token = credentials.credentials

    try:
        body = await request.json()
        rpc = JsonRpcRequest(**body)
    except Exception:
        return _jsonrpc_error(None, -32700, "Parse error")

    handler = TOOL_HANDLERS.get(rpc.method)
    if not handler:
        return _jsonrpc_error(rpc.id, -32601, f"Method not found: {rpc.method}")

    try:
        result = await handler(rpc.params, token)
        return _jsonrpc_result(rpc.id, result)
    except Exception as exc:
        return _jsonrpc_error(rpc.id, -32000, str(exc))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
