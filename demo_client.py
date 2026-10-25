"""R6 demo — small Python client that demonstrates SSE streaming."""

import json
import sys

import httpx

BASE = "http://localhost:8000"
TOKEN = "tok-merchant-a"


def stream_chat(message: str, thread_id: str | None = None):
    """Send a message and print SSE events as they arrive."""
    payload = {"message": message}
    if thread_id:
        payload["thread_id"] = thread_id

    print(f"\n>>> {message}")
    tid = thread_id

    with httpx.Client(timeout=30.0) as client:
        with client.stream(
            "POST",
            f"{BASE}/chat",
            json=payload,
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    event = line[len("event:"):].strip()
                elif line.startswith("data:"):
                    data = json.loads(line[len("data:"):].strip())
                    if event == "thread_id":
                        tid = data["thread_id"]
                        print(f"[thread: {tid}]")
                    elif event == "tool_start":
                        print(f"  [tool start: {data['tool']}]")
                    elif event == "tool_end":
                        print(f"  [tool end: {data['tool']}]")
                    elif event == "token":
                        print(data["content"], end="", flush=True)
                    elif event == "proposal":
                        print(f"\n  ** PROPOSAL: {json.dumps(data, indent=2)}")
                    elif event == "done":
                        print()

    return tid


def approve(thread_id: str, action: str = "approve"):
    """Send approve/reject and print the SSE events."""
    print(f"\n>>> {action}")
    with httpx.Client(timeout=30.0) as client:
        with client.stream(
            "POST",
            f"{BASE}/chat/approve",
            json={"thread_id": thread_id, "action": action},
            headers={"Authorization": f"Bearer {TOKEN}"},
        ) as resp:
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    data = json.loads(line[len("data:"):].strip())
                    if "content" in data:
                        print(data["content"], end="", flush=True)
            print()


if __name__ == "__main__":
    print("=== ShopHub Support Chat — SSE Demo ===")
    print(f"Token: {TOKEN}\n")

    # 1. Ask about paid orders
    tid = stream_chat("Show me all paid orders")

    # 2. Look up a specific order
    tid = stream_chat("Look up order 1001", thread_id=tid)

    # 3. Request a refund (will produce a proposal)
    tid = stream_chat("Refund order 1001", thread_id=tid)

    # 4. Approve the refund
    if tid:
        approve(tid, "approve")

    print("\n=== Demo complete ===")
