"""Shared fixtures for pytest — runs store API and sub-agent in-process."""

from __future__ import annotations

import os

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Force fake LLM
os.environ["LLM_PROVIDER"] = "fake"


@pytest.fixture(scope="session", autouse=True)
def _set_env():
    os.environ["LLM_PROVIDER"] = "fake"
    os.environ["STORE_API_URL"] = "http://testserver-store"
    os.environ["SUB_AGENT_URL"] = "http://testserver-subagent"


@pytest.fixture
def store_client():
    """Async test client for the store API."""
    from src.store_api import app, _reset_db

    _reset_db()
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://testserver-store")


@pytest.fixture
def subagent_client():
    """Async test client for the sub-agent."""
    from src.orders_agent import app

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://testserver-subagent")
