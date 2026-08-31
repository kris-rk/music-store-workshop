"""Create one Studio assistant per simulated logged-in customer.

Each assistant pins a `customer_id` into the graph's runtime context, so in
Studio's Chat mode you pick a persona from the dropdown and start typing — no
config panel needed.

Assistant ids are derived from the persona name, so re-running this script
updates the existing assistants instead of creating duplicates.

    uv run langgraph dev                        # in one terminal
    uv run python scripts/create_assistants.py  # in another
"""

import json
import urllib.error
import urllib.request
import uuid

BASE = "http://127.0.0.1:2024"
GRAPH = "music_store_support"
NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")

PERSONAS = [
    ("Roberto Almeida (customer 12)", 12),
    ("Frantisek Wichterlova (customer 5)", 5),
    ("Signed out (no session)", None),
]


def call(method: str, path: str, payload: dict | None = None) -> dict | None:
    request = urllib.request.Request(
        f"{BASE}{path}",
        method=method,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"content-type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            body = response.read()
            return json.loads(body) if body else None
    except urllib.error.URLError as exc:
        raise SystemExit(f"Could not reach {BASE} — is `langgraph dev` running?\n  {exc}") from exc


for name, customer_id in PERSONAS:
    # Deterministic id => re-running is idempotent rather than duplicating.
    assistant_id = str(uuid.uuid5(NAMESPACE, name))
    call(
        "POST",
        "/assistants",
        {
            "assistant_id": assistant_id,
            "graph_id": GRAPH,
            "name": name,
            "context": {"customer_id": customer_id},
            "if_exists": "do_nothing",
        },
    )
    print(f"  {name:<36} customer_id={customer_id}")

existing = call("POST", "/assistants/search", {"limit": 50}) or []
print(f"\n{len(existing)} assistants on the server:")
for assistant in existing:
    print(f"  {assistant['name']:<36} context={assistant.get('context')}")

print(f"\nOpen https://smith.langchain.com/studio/?baseUrl={BASE}")
print("Switch to Chat mode and pick a persona from the assistant dropdown.")
