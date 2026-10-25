"""Fake LLM that returns scripted tool calls and responses.

Runs with no API key.  The scripted sequences cover the five eval-harness
scenarios plus basic interactive use.  When no script matches, it returns a
safe fallback message.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolCall
from langchain_core.outputs import ChatGeneration, ChatResult


def _extract_user_text(messages: list[BaseMessage]) -> str:
    """Pull the last human message text for pattern matching."""
    for m in reversed(messages):
        if m.type == "human":
            return m.content if isinstance(m.content, str) else ""
    return ""


def _has_tool_result(messages: list[BaseMessage]) -> bool:
    """Check if the last message is a tool result."""
    return bool(messages) and messages[-1].type == "tool"


def _last_tool_content(messages: list[BaseMessage]) -> str:
    """Get text of the last tool message."""
    for m in reversed(messages):
        if m.type == "tool":
            return m.content if isinstance(m.content, str) else ""
    return ""


class FakeChatModel(BaseChatModel):
    """Deterministic chat model for testing and offline evaluation."""

    model_name: str = "fake"

    @property
    def _llm_type(self) -> str:
        return "fake-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        user_text = _extract_user_text(messages).lower().strip()
        has_tool_res = _has_tool_result(messages)
        tool_content = _last_tool_content(messages)

        # If we just got a tool result back, summarise it
        if has_tool_res:
            return self._summarise_tool_result(tool_content, user_text, messages)

        # Pattern-match the user query to decide which tool to call
        ai_msg = self._route_query(user_text)
        return ChatResult(generations=[ChatGeneration(message=ai_msg)])

    def _route_query(self, user_text: str) -> AIMessage:
        # List orders / status queries
        if re.search(r"(paid|shipped|placed|delivered|cancelled|refunded)\s*(orders?)?", user_text):
            status = "PAID"
            for s in ["SHIPPED", "PLACED", "DELIVERED", "CANCELLED", "REFUNDED"]:
                if s.lower() in user_text:
                    status = s
                    break
            return AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="list_orders",
                        args={"status": status},
                        id="call_list_1",
                    )
                ],
            )

        # Lookup by order id
        id_match = re.search(r"order\s*#?\s*(\d+)", user_text)
        if id_match and not re.search(r"(refund|cancel)", user_text):
            return AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="lookup_order",
                        args={"order_id": id_match.group(1)},
                        id="call_lookup_1",
                    )
                ],
            )

        # Refund
        if "refund" in user_text:
            id_match = re.search(r"(\d{4,})", user_text)
            order_id = id_match.group(1) if id_match else "1001"
            return AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="lookup_order",
                        args={"order_id": order_id},
                        id="call_lookup_refund",
                    )
                ],
            )

        # Cancel
        if "cancel" in user_text:
            id_match = re.search(r"(\d{4,})", user_text)
            order_id = id_match.group(1) if id_match else "1001"
            return AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="lookup_order",
                        args={"order_id": order_id},
                        id="call_lookup_cancel",
                    )
                ],
            )

        # Customer search
        name_match = re.search(r"customer\s+(\w+)", user_text)
        if name_match or "maria" in user_text or "what did" in user_text:
            return AIMessage(
                content="",
                tool_calls=[
                    ToolCall(
                        name="list_orders",
                        args={},
                        id="call_list_cust",
                    )
                ],
            )

        # Fallback
        return AIMessage(
            content="I can help you with order inquiries. You can ask me to look up orders, check order status, or request refunds and cancellations."
        )

    def _summarise_tool_result(
        self, tool_content: str, user_text: str, messages: list[BaseMessage]
    ) -> ChatResult:
        # Check if this was a lookup for refund/cancel
        original_query = _extract_user_text(messages).lower()

        # If tool returned a failure (legacy 200 + status:false)
        if '"status": false' in tool_content or '"status":false' in tool_content:
            msg = AIMessage(content="I could not find that order. It may not exist or may belong to a different merchant.")
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # If the original query was a refund request, produce a proposal
        if "refund" in original_query:
            try:
                data = json.loads(tool_content)
                order = data.get("order", data)
                if isinstance(order, list):
                    order = order[0] if order else {}
                msg = AIMessage(
                    content=json.dumps({
                        "type": "proposal",
                        "action": "refund",
                        "order_id": order.get("id", "unknown"),
                        "merchant": order.get("merchant", "unknown"),
                        "customer_name": order.get("customer_name", "unknown"),
                        "current_status": order.get("status", "unknown"),
                        "amount": order.get("total", 0),
                        "message": f"I'd like to refund order {order.get('id', 'unknown')} for ${order.get('total', 0)}. Please approve or reject this action.",
                    })
                )
            except (json.JSONDecodeError, KeyError):
                msg = AIMessage(content='{"type": "proposal", "action": "refund", "message": "I\'d like to refund this order. Please approve or reject."}')
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # If the original query was a cancel request, produce a proposal
        if "cancel" in original_query:
            try:
                data = json.loads(tool_content)
                order = data.get("order", data)
                if isinstance(order, list):
                    order = order[0] if order else {}
                msg = AIMessage(
                    content=json.dumps({
                        "type": "proposal",
                        "action": "cancel",
                        "order_id": order.get("id", "unknown"),
                        "merchant": order.get("merchant", "unknown"),
                        "customer_name": order.get("customer_name", "unknown"),
                        "current_status": order.get("status", "unknown"),
                        "amount": order.get("total", 0),
                        "message": f"I'd like to cancel order {order.get('id', 'unknown')}. Please approve or reject this action.",
                    })
                )
            except (json.JSONDecodeError, KeyError):
                msg = AIMessage(content='{"type": "proposal", "action": "cancel", "message": "I\'d like to cancel this order. Please approve or reject."}')
            return ChatResult(generations=[ChatGeneration(message=msg)])

        # Normal summary of tool results
        try:
            data = json.loads(tool_content)
            if isinstance(data, list):
                count = len(data)
                msg = AIMessage(content=f"I found {count} order(s). " + self._format_orders(data))
            elif isinstance(data, dict) and "order" in data:
                msg = AIMessage(content=f"Here are the details for the order: " + self._format_orders([data["order"]]))
            else:
                msg = AIMessage(content=f"Here is the result: {tool_content}")
        except json.JSONDecodeError:
            msg = AIMessage(content=f"Here is the result: {tool_content}")

        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _format_orders(self, orders: list[dict]) -> str:
        lines = []
        for o in orders:
            lines.append(
                f"Order {o.get('id')}: {o.get('customer_name', 'N/A')} — "
                f"status={o.get('status')}, total=${o.get('total')}"
            )
        return "\n".join(lines)

    @property
    def _identifying_params(self) -> dict:
        return {"model_name": self.model_name}
