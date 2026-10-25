"""R3/R8 — Tenant isolation tests.

Merchant A must never see merchant B's orders, even by direct ID lookup.
The no-merchant token must be rejected (fail-closed).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Store API level
# ---------------------------------------------------------------------------


async def test_merchant_a_sees_only_own_orders(store_client):
    """Merchant A token should only return merchant_a orders."""
    resp = await store_client.get(
        "/orders", headers={"Authorization": "Bearer tok-merchant-a"}
    )
    assert resp.status_code == 200
    orders = resp.json()
    assert len(orders) > 0
    for o in orders:
        assert o["merchant"] == "merchant_a"


async def test_merchant_b_sees_only_own_orders(store_client):
    """Merchant B token should only return merchant_b orders."""
    resp = await store_client.get(
        "/orders", headers={"Authorization": "Bearer tok-merchant-b"}
    )
    assert resp.status_code == 200
    orders = resp.json()
    assert len(orders) > 0
    for o in orders:
        assert o["merchant"] == "merchant_b"


async def test_cross_merchant_lookup_returns_not_found(store_client):
    """Looking up merchant_b's order with merchant_a's token returns status:false."""
    resp = await store_client.get(
        "/orders/lookup",
        params={"id": "2001"},  # belongs to merchant_b
        headers={"Authorization": "Bearer tok-merchant-a"},
    )
    assert resp.status_code == 200  # legacy behaviour
    data = resp.json()
    assert data["status"] is False
    assert "not found" in data["message"].lower()


async def test_cross_merchant_refund_returns_404(store_client):
    """Refunding merchant_b's order with merchant_a's token returns 404."""
    resp = await store_client.post(
        "/orders/refund",
        json={"id": "2001"},
        headers={"Authorization": "Bearer tok-merchant-a"},
    )
    assert resp.status_code == 404


async def test_cross_merchant_cancel_returns_404(store_client):
    """Cancelling merchant_b's order with merchant_a's token returns 404."""
    resp = await store_client.post(
        "/orders/cancel",
        json={"id": "2001"},
        headers={"Authorization": "Bearer tok-merchant-a"},
    )
    assert resp.status_code == 404


async def test_no_merchant_token_rejected(store_client):
    """A valid token with no merchant scope must be rejected (fail-closed, R4c)."""
    resp = await store_client.get(
        "/orders", headers={"Authorization": "Bearer tok-no-merchant"}
    )
    assert resp.status_code == 403


async def test_invalid_token_rejected(store_client):
    """An invalid token must be rejected."""
    resp = await store_client.get(
        "/orders", headers={"Authorization": "Bearer tok-invalid"}
    )
    assert resp.status_code == 401


async def test_missing_auth_rejected(store_client):
    """No Authorization header must be rejected."""
    resp = await store_client.get("/orders")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Sub-agent level (JSON-RPC)
# ---------------------------------------------------------------------------


async def test_subagent_cross_merchant_lookup(store_client, subagent_client, monkeypatch):
    """Sub-agent lookup of another merchant's order returns an error."""
    import httpx as httpx_mod
    from httpx import ASGITransport

    import src.orders_agent as oa
    from src.store_api import app as store_app

    monkeypatch.setattr(oa, "STORE_API_URL", "http://testserver-store")

    OriginalClient = httpx_mod.AsyncClient

    class PatchedClient:
        def __init__(self, *args, **kwargs):
            self._client = OriginalClient(
                transport=ASGITransport(app=store_app),
                base_url="http://testserver-store",
            )

        async def __aenter__(self):
            await self._client.__aenter__()
            return self._client

        async def __aexit__(self, *args):
            await self._client.__aexit__(*args)

    monkeypatch.setattr(oa.httpx, "AsyncClient", PatchedClient)

    resp = await subagent_client.post(
        "/jsonrpc",
        json={
            "jsonrpc": "2.0",
            "method": "lookup_order",
            "params": {"order_id": "2001"},
            "id": "test-1",
        },
        headers={"Authorization": "Bearer tok-merchant-a"},
    )
    data = resp.json()
    result = data.get("result", {})
    assert "error" in result, f"Expected error in result, got: {data}"
