"""Music Store Customer Support — Deep Agent.

AUTO-GENERATED from `agent.ipynb`: every cell tagged `# @export`, in order.
Edit the notebook and re-run its final cell rather than editing this file.
"""

from __future__ import annotations

import os
import re
import sqlite3
import threading
import urllib.request
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepagents import create_deep_agent
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    ModelRequest,
    ModelResponse,
    SummarizationMiddleware,
    ToolCallRequest,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime
from langgraph.types import interrupt


# --- Model policy ----------------------------------------------------------
# Cheapest-first. The supervisor is the only place that gets a stronger model,
# because routing/judgement is literally its whole job. Every specialist, every
# guard, and the web-search executor run on Haiku.
ORCHESTRATOR_MODEL = os.environ.get("MS_ORCHESTRATOR_MODEL", "anthropic:claude-sonnet-5")
SPECIALIST_MODEL = os.environ.get("MS_SPECIALIST_MODEL", "anthropic:claude-haiku-4-5-20251001")
GUARD_MODEL = os.environ.get("MS_GUARD_MODEL", "claude-haiku-4-5-20251001")

# --- Data ------------------------------------------------------------------
def _find_project_root() -> Path:
    """Locate the repo root from either a notebook (cwd) or the exported module."""
    env = os.environ.get("MS_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    start = Path(globals()["__file__"]).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd().resolve()


PROJECT_ROOT = _find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = Path(os.environ.get("CHINOOK_DB_PATH", DATA_DIR / "chinook.db"))
CHINOOK_SQL_URL = (
    "https://raw.githubusercontent.com/lerocha/chinook-database/master/"
    "ChinookDatabase/DataSources/Chinook_Sqlite.sql"
)

MAX_ROWS = 40


def ensure_database(force: bool = False) -> Path:
    """Download the Chinook SQL dump and materialise it as a local SQLite file."""
    if DB_PATH.exists() and not force:
        return DB_PATH
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump = DB_PATH.parent / "Chinook_Sqlite.sql"
    if not dump.exists() or force:
        urllib.request.urlretrieve(CHINOOK_SQL_URL, dump)
    DB_PATH.unlink(missing_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(dump.read_text(encoding="utf-8"))
    con.commit()
    con.close()
    return DB_PATH

#SYSTEM SETTINGS
@dataclass
class SupportContext:
    """Per-run session context — the simulated *logged-in user*.

    `customer_id` is set by the application (or by LangSmith Studio, or by the
    eval harness). The model can neither read nor set it, and every customer-data
    tool call is bound to it by the security middleware.
    """

    customer_id: int | None = None
    store_name: str = "Chinook Records"
    audit_id: str | None = None
    """Optional correlation id. When set, every tool call and its result are
    recorded in `AUDIT_LOG[audit_id]` so evaluators can grade the *trajectory*,
    not just the final answer."""


def _session_customer_id(runtime: Runtime[SupportContext] | None) -> int | None:
    """Read the authenticated customer id, failing closed when it is absent."""
    ctx = getattr(runtime, "context", None)
    if ctx is None:
        return None
    value = ctx.get("customer_id") if isinstance(ctx, dict) else getattr(ctx, "customer_id", None)
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _session_audit_id(runtime: Runtime[SupportContext] | None) -> str | None:
    ctx = getattr(runtime, "context", None)
    if ctx is None:
        return None
    value = ctx.get("audit_id") if isinstance(ctx, dict) else getattr(ctx, "audit_id", None)
    return str(value) if value else None


def extract_text(message: Any) -> str:
    """Flatten a message's content to plain text (Claude returns content blocks)."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") for b in content if isinstance(b, dict) and b.get("type") == "text"
        ).strip()
    return str(content)


def final_answer(result: dict[str, Any]) -> str:
    """The customer-facing text of an agent run."""
    messages = result.get("messages") or []
    return extract_text(messages[-1]) if messages else ""


CHINOOK_SCHEMA = """\
Artist(ArtistId, Name)
Album(AlbumId, Title, ArtistId -> Artist)
Track(TrackId, Name, AlbumId -> Album, MediaTypeId -> MediaType, GenreId -> Genre,
      Composer, Milliseconds, Bytes, UnitPrice)
Genre(GenreId, Name)
MediaType(MediaTypeId, Name)
Playlist(PlaylistId, Name)
PlaylistTrack(PlaylistId -> Playlist, TrackId -> Track)
Customer(CustomerId, FirstName, LastName, Company, Address, City, State, Country,
         PostalCode, Phone, Fax, Email, SupportRepId -> Employee)
Invoice(InvoiceId, CustomerId -> Customer, InvoiceDate, BillingAddress, BillingCity,
        BillingState, BillingCountry, BillingPostalCode, Total)
InvoiceLine(InvoiceLineId, InvoiceId -> Invoice, TrackId -> Track, UnitPrice, Quantity)
Employee(EmployeeId, LastName, FirstName, Title, ReportsTo, BirthDate, HireDate,
         Address, City, State, Country, PostalCode, Phone, Fax, Email)

SQLite dialect. A single SELECT or WITH statement.
"""

# Appended to the schema only when the security middleware is installed, because
# it describes guarantees that middleware provides. The prompt-only baseline in
# `evaluators.ipynb` gets the plain schema above — which is the point.
CHINOOK_SCHEMA_SECURITY_NOTES = """
Enforced by the security middleware:
* Customer, Invoice and InvoiceLine are row-level filtered to the signed-in
  customer, so `SELECT * FROM Customer` returns exactly one row: theirs.
* Employee holds staff records and is not customer-accessible. Do not query it.
* Catalogue tables (Artist/Album/Track/Genre/MediaType/Playlist/PlaylistTrack)
  are unrestricted — use them freely for product questions.
"""

_WRITE_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|"
    r"pragma|vacuum|reindex|analyze)\b",
    re.IGNORECASE,
)
_SCHEMA_QUALIFIED = re.compile(r"\b(main|temp)\s*\.", re.IGNORECASE)
_EMPLOYEE_TABLE = re.compile(r"\bemployee\b", re.IGNORECASE)
_RESTRICTED_TABLES = re.compile(r"\b(customer|invoice|invoiceline)\b", re.IGNORECASE)

#OPENING DATABASE
def _open_scoped_connection(customer_scope: int | None) -> sqlite3.Connection:
    """Open a read-only Chinook connection, optionally with row-level security.

    SQLite resolves the `temp` schema before `main`, so these TEMP VIEWs shadow
    the real tables for every unqualified reference in the query. Even
    `WHERE CustomerId = 5 OR 1=1` cannot escape the view.

    With `customer_scope=None` this is just a read-only database client. The
    security middleware is what turns it into a single-customer one.
    """
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    if customer_scope is None:
        return con
    scope = int(customer_scope)
    cur = con.cursor()
    cur.execute(f"CREATE TEMP VIEW Customer AS SELECT * FROM main.Customer WHERE CustomerId = {scope}")
    cur.execute(f"CREATE TEMP VIEW Invoice AS SELECT * FROM main.Invoice WHERE CustomerId = {scope}")
    cur.execute(
        "CREATE TEMP VIEW InvoiceLine AS SELECT il.* FROM main.InvoiceLine il "
        "JOIN main.Invoice i ON i.InvoiceId = il.InvoiceId "
        f"WHERE i.CustomerId = {scope}"
    )
    return con

#TOOL CALLS
@tool(parse_docstring=True)
def search_db(sql: str, customer_scope: int | None = None) -> str:
    """Run a read-only SQL query against the Chinook music-store database.

    Use this for anything in the store's own records: the music catalogue
    (artists, albums, tracks, genres, playlists, prices) and the signed-in
    customer's profile and orders.

    Args:
        sql: A single SQLite SELECT (or WITH ... SELECT) statement.
        customer_scope: Do not set this. The security middleware overwrites it
            with the signed-in customer's id.
    """
    statement = sql.strip().rstrip(";").strip()
    con = _open_scoped_connection(customer_scope)
    try:
        rows = con.execute(statement).fetchmany(MAX_ROWS + 1)
        if not rows:
            return "0 rows."
        truncated = len(rows) > MAX_ROWS
        rows = rows[:MAX_ROWS]
        columns = list(rows[0].keys())
        lines = [" | ".join(columns)]
        lines += [" | ".join("" if r[c] is None else str(r[c]) for c in columns) for r in rows]
        if truncated:
            lines.append(f"... (truncated at {MAX_ROWS} rows — add LIMIT or aggregate)")
        return "\n".join(lines)
    except sqlite3.Error as exc:
        return f"SQL error: {exc}. Check the schema and try again."
    finally:
        con.close()


@tool(parse_docstring=True)
def cust_profile(customer_id: int | None = None, include_orders: bool = True) -> str:
    """Retrieve the signed-in customer's account profile and order history.

    The fast path for "what did I buy", "what's my order status" and "what do you
    have on file for me" style questions.

    Args:
        customer_id: Do not set this. The security middleware overwrites it with
            the signed-in customer's id; naming a different customer is denied.
        include_orders: Whether to include the customer's invoice history.
    """
    if customer_id is None:
        return "ACCESS DENIED: no authenticated customer session."

    con = _open_scoped_connection(customer_id)
    try:
        row = con.execute("SELECT * FROM Customer").fetchone()
        if row is None:
            return f"No customer record found for CustomerId {customer_id}."
        out = ["ACCOUNT PROFILE"]
        out += [f"  {k}: {row[k]}" for k in row.keys() if k != "SupportRepId" and row[k] is not None]
        if include_orders:
            invoices = con.execute(
                "SELECT InvoiceId, date(InvoiceDate) AS InvoiceDate, BillingCity, "
                "BillingCountry, Total FROM Invoice ORDER BY InvoiceDate DESC"
            ).fetchall()
            # Order lines too: without them the agent knows *that* orders exist but
            # not what is in them, and will happily invent the contents.
            lines: dict[int, list[str]] = {}
            for line in con.execute(
                "SELECT il.InvoiceId, t.Name AS Track, ar.Name AS Artist "
                "FROM InvoiceLine il "
                "JOIN Track t ON t.TrackId = il.TrackId "
                "LEFT JOIN Album al ON al.AlbumId = t.AlbumId "
                "LEFT JOIN Artist ar ON ar.ArtistId = al.ArtistId"
            ):
                lines.setdefault(line["InvoiceId"], []).append(
                    f"{line['Track']} — {line['Artist'] or 'Unknown artist'}"
                )

            total = sum(float(i["Total"]) for i in invoices)
            out.append(f"\nORDER HISTORY ({len(invoices)} orders, ${total:.2f} lifetime)")
            for invoice in invoices[:MAX_ROWS]:
                out.append(
                    f"  Order #{invoice['InvoiceId']} | {invoice['InvoiceDate']} | "
                    f"{invoice['BillingCity']}, {invoice['BillingCountry']} | "
                    f"${invoice['Total']:.2f} | status: Delivered"
                )
                out += [f"      - {item}" for item in lines.get(invoice["InvoiceId"], [])]
        return "\n".join(out)
    except sqlite3.Error as exc:
        return f"Database error: {exc}"
    finally:
        con.close()


_anthropic_client = None
_async_anthropic_client = None

#ANTHROPIC POWERED WEB SEARCH
def _anthropic():
    """Lazily create a raw Anthropic client (used inside the sync `web_search` tool)."""
    global _anthropic_client
    if _anthropic_client is None:
        from anthropic import Anthropic

        _anthropic_client = Anthropic()
    return _anthropic_client


def _async_anthropic():
    """Async twin, for guards that run on the event loop and must not block it."""
    global _async_anthropic_client
    if _async_anthropic_client is None:
        from anthropic import AsyncAnthropic

        _async_anthropic_client = AsyncAnthropic()
    return _async_anthropic_client


@tool(parse_docstring=True)
def web_search(query: str) -> str:
    """Search the public web for music information the store database cannot answer.

    Use for artists, albums, release history, band members, cover art, awards and
    tours. Queries unrelated to music or the store are rejected by policy.

    Args:
        query: A focused, music-related search query.
    """
    response = _anthropic().messages.create(
        model=GUARD_MODEL,
        max_tokens=1200,
        system=(
            "You are a music research assistant for a record store. Use web search to "
            "answer the query. Reply with a concise factual summary (<=150 words) and "
            "list the source URLs you used."
        ),
        tools=[
            {
                "type": "web_search_20260209",
                "name": "web_search",
                "max_uses": 3,
                "allowed_callers": ["direct"],
            }
        ],
        messages=[{"role": "user", "content": query}],
    )
    text = "\n".join(b.text for b in response.content if getattr(b, "type", None) == "text")
    return text.strip() or "No web results found."

#HUMAN ESCALATION TOOL
@tool(parse_docstring=True)
def escalate_to_human(reason: str, conversation_summary: str, urgency: str = "normal") -> str:
    """Hand this conversation off to a human support agent.

    Args:
        reason: Why a human is required (customer asked, complaint, refund,
            legal or safety concern, repeated failure to help).
        conversation_summary: Everything the human needs to pick up the thread.
        urgency: One of "low", "normal", "high".
    """
    ticket = f"MS-{abs(hash(conversation_summary)) % 90000 + 10000}"
    return (
        f"Escalation accepted. Ticket {ticket} created (urgency={urgency}, reason={reason}). "
        "A human support agent has joined the conversation and will take it from here."
    )


AUDIT_LOG: dict[str, list[dict[str, Any]]] = defaultdict(list)
_AUDIT_LOCK = threading.Lock()

#TOOL CALL USAGE MIDDLEWARE
@wrap_tool_call
async def tool_audit(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
) -> ToolMessage:
    """Record every tool call and the result the model actually received.

    Placed *outermost* so it observes the post-guard outcome. This is what makes
    trajectory evaluation possible: the final answer can look impeccable while
    the trace shows that another customer's rows were loaded into context.
    """
    result = await handler(request)
    audit_id = _session_audit_id(request.runtime)
    if audit_id:
        content = extract_text(result) if isinstance(result, ToolMessage) else str(result)
        with _AUDIT_LOCK:
            AUDIT_LOG[audit_id].append(
                {
                    "tool": request.tool_call["name"],
                    "args": dict(request.tool_call.get("args") or {}),
                    "result": content[:4000],
                }
            )
    return result


def audit_trail(audit_id: str) -> list[dict[str, Any]]:
    """Every tool call made during the run tagged with `audit_id`."""
    with _AUDIT_LOCK:
        return list(AUDIT_LOG.get(audit_id, []))


CUSTOMER_TOOLS = {"search_db", "cust_profile"}
SECURITY_LOG: list[dict[str, Any]] = []

#LLM GAURDRAILS - CUSTOMER SECURITY
def _deny(request: ToolCallRequest, message: str, rule: str) -> ToolMessage:
    SECURITY_LOG.append({"tool": request.tool_call["name"], "action": "denied", "rule": rule})
    return ToolMessage(
        content=f"ACCESS DENIED ({rule}): {message}",
        tool_call_id=request.tool_call["id"],
        name=request.tool_call["name"],
        status="error",
    )


def _leaks_foreign_customer(content: str, session_id: int) -> bool:
    """True if a rendered result table exposes a CustomerId other than the session's."""
    lines = [line for line in content.splitlines() if line.strip()]
    if not lines:
        return False
    header = [c.strip() for c in lines[0].split("|")]
    lowered = [c.lower() for c in header]
    if "customerid" not in lowered:
        return False
    idx = lowered.index("customerid")
    for line in lines[1:]:
        cells = [c.strip() for c in line.split("|")]
        if len(cells) == len(header) and cells[idx].isdigit() and int(cells[idx]) != session_id:
            return True
    return False


@wrap_tool_call
async def customer_security_guard(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
) -> ToolMessage:
    """Bind every customer-data tool call to the signed-in customer. Five layers:

    1. **Fail closed** — no authenticated session, no customer data. Ever.
    2. **Scope rewriting** — the session's customer id is written into the tool
       arguments. Asking for a *different* customer is denied and logged.
    3. **SQL policy** — read-only, single statement, no staff records, and no
       `main.`/`temp.` qualification (which would sidestep row-level security).
    4. **Transparency** — results from row-filtered tables are labelled, so the
       agent cannot mistake "my average order" for "the average customer".
    5. **Leak check** — any row carrying a foreign CustomerId is discarded before
       the model sees it.
    """
    name = request.tool_call["name"]
    if name not in CUSTOMER_TOOLS:
        return await handler(request)

    session_id = _session_customer_id(request.runtime)
    args = dict(request.tool_call.get("args") or {})

    # 1. fail closed
    if session_id is None:
        return _deny(
            request,
            "there is no authenticated customer session, so no account or order data "
            "can be read. Ask the customer to sign in.",
            "no-session",
        )

    # 2. scope rewriting
    scope_arg = "customer_scope" if name == "search_db" else "customer_id"
    requested = args.get(scope_arg)
    if requested is not None and str(requested).strip() not in ("", "None"):
        try:
            requested_id: int | None = int(requested)
        except (TypeError, ValueError):
            requested_id = None
        if requested_id != session_id:
            return _deny(
                request,
                f"the signed-in customer ({session_id}) may only access their own records, "
                f"not customer {requested}. Tell the customer you can only discuss their "
                "own account.",
                "cross-customer-access",
            )
    args[scope_arg] = session_id

    # 3. SQL policy
    if name == "search_db":
        sql = str(args.get("sql", "")).strip().rstrip(";").strip()
        if not re.match(r"^\s*(select|with)\b", sql, re.IGNORECASE):
            return _deny(request, "only read-only SELECT/WITH queries are permitted.", "read-only")
        if ";" in sql:
            return _deny(request, "only a single SQL statement may be executed.", "single-statement")
        if _WRITE_KEYWORDS.search(sql):
            return _deny(request, "the query contains a write or schema keyword.", "read-only")
        if _SCHEMA_QUALIFIED.search(sql):
            return _deny(
                request,
                "schema-qualified table names (main./temp.) bypass row-level security. "
                "Use bare table names.",
                "rls-bypass",
            )
        if _EMPLOYEE_TABLE.search(sql):
            return _deny(request, "employee and staff records are not customer-accessible.", "staff-pii")
        args["sql"] = sql

    SECURITY_LOG.append({"tool": name, "action": "scoped", "customer_id": session_id})
    result = await handler(request.override(tool_call={**request.tool_call, "args": args}))

    if not isinstance(result, ToolMessage) or not isinstance(result.content, str):
        return result

    # 5. leak check
    if _leaks_foreign_customer(result.content, session_id):
        return _deny(
            request,
            "the result contained records belonging to another customer and was discarded.",
            "leak-detected",
        )

    # 4. transparency
    if name == "search_db" and _RESTRICTED_TABLES.search(str(args.get("sql", ""))):
        return ToolMessage(
            content=(
                f"[row-level security] Customer/Invoice/InvoiceLine were filtered to "
                f"CustomerId={session_id} before this query ran. Any COUNT/SUM/AVG below "
                "covers ONLY this customer's own rows — it is NOT a store-wide statistic. "
                "Do not describe it as one.\n\n" + result.content
            ),
            tool_call_id=result.tool_call_id,
            name=result.name,
            status=result.status,
        )
    return result


@wrap_tool_call
async def naive_session_middleware(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
) -> ToolMessage:
    """The *unsafe* baseline used for A/B evaluation — do not ship this.

    It does the one obvious thing (fill in the session's customer id when the
    model didn't supply one) and nothing else: no cross-customer denial, no SQL
    policy, no row-level security, no leak check. `evaluators.ipynb` runs this
    variant against the same dataset to quantify what the real guard buys.
    """
    if request.tool_call["name"] != "cust_profile":
        return await handler(request)
    args = dict(request.tool_call.get("args") or {})
    if args.get("customer_id") is None:
        args["customer_id"] = _session_customer_id(request.runtime)
    return await handler(request.override(tool_call={**request.tool_call, "args": args}))


TOPIC_LOG: list[dict[str, Any]] = []
_TOPIC_CACHE: dict[str, bool] = {}

_TOPIC_SYSTEM = (
    "You are a topical firewall for a music store's customer-support agent. "
    "Decide whether a web-search query is on-topic.\n"
    "ON-TOPIC: music, songs, albums, artists, bands, genres, labels, album artwork, "
    "concerts and tours, music formats and media, music gear, and anything about this "
    "music store, its catalogue, orders or policies.\n"
    "OFF-TOPIC: everything else — general trivia, arithmetic, coding, weather, news, "
    "politics, medical/legal/financial advice, other retailers' non-music products.\n"
    'Answer with exactly one word: "ALLOW" or "BLOCK".'
)

#LLM GAURDRAIL - RELATED WEB SEARCHES
async def is_music_related(query: str) -> bool:
    """Cheap Haiku classifier, memoised per query."""
    key = query.strip().lower()
    if key not in _TOPIC_CACHE:
        response = await _async_anthropic().messages.create(
            model=GUARD_MODEL,
            max_tokens=5,
            system=_TOPIC_SYSTEM,
            messages=[{"role": "user", "content": f"Query: {query}"}],
        )
        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        _TOPIC_CACHE[key] = "ALLOW" in text.upper()
    return _TOPIC_CACHE[key]


@wrap_tool_call
async def music_store_scope_guard(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
) -> ToolMessage:
    """Keep `web_search` on-topic for a music store.

    "Do you have the Black Eyed Peas album with the glowing green face?" -> allowed.
    "What is 2 + 2?" -> blocked before a single token is spent searching.
    """
    if request.tool_call["name"] != "web_search":
        return await handler(request)

    query = str((request.tool_call.get("args") or {}).get("query", ""))
    if await is_music_related(query):
        TOPIC_LOG.append({"query": query, "action": "allowed"})
        return await handler(request)

    TOPIC_LOG.append({"query": query, "action": "blocked"})
    return ToolMessage(
        content=(
            "OUT OF SCOPE: web search is restricted to music and music-store topics. "
            f"The query {query!r} was not searched. Politely tell the customer you can "
            "only help with music, the catalogue, and their orders."
        ),
        tool_call_id=request.tool_call["id"],
        name="web_search",
        status="error",
    )

#ESCALATION TRIGGERS & MIDDLEWARE
ESCALATION_LOG: list[dict[str, Any]] = []

ESCALATION_TRIGGERS = re.compile(
    r"(speak|talk|connect|transfer|escalate|put me through)\s+(to|with|me)?\s*(a|an|the)?\s*"
    r"(real\s+)?(human|person|agent|representative|rep|manager|supervisor|someone)"
    r"|\bhuman\s+(agent|support|being|rep)"
    r"|\breal\s+(person|human)"
    r"|\b(lawyer|attorney|sue|lawsuit|legal action|small claims)\b"
    r"|\b(fraud|unauthorized charge|unauthorised charge|stolen card|identity theft)\b"
    r"|\b(refund|chargeback|cancel my account|close my account)\b"
    r"|\b(this is (unacceptable|ridiculous)|i want to (complain|file a complaint)|"
    r"terrible service|worst)\b",
    re.IGNORECASE,
)

ESCALATION_DIRECTIVE = """

<escalation_override>
The customer's latest message matched the human-escalation policy. You MUST
delegate to the `escalation-specialist` subagent with the `task` tool on this
turn. Do not try to resolve it yourself and do not answer without escalating.
</escalation_override>"""


@wrap_model_call
async def human_escalation_router(
    request: ModelRequest,
    handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
) -> ModelResponse:
    """Trigger the human-escalation workflow from the customer's own words.

    A prompt alone leaves this to the model's mood. Matching the policy in
    middleware means "I want to talk to a person" always lands the same way,
    and it shows up as a deterministic branch in the LangSmith trace.
    """
    last_human = next((m for m in reversed(request.messages) if isinstance(m, HumanMessage)), None)
    text = ""
    if last_human is not None:
        raw = getattr(last_human, "text", None)
        text = raw if isinstance(raw, str) else str(last_human.content)

    if text and ESCALATION_TRIGGERS.search(text):
        ESCALATION_LOG.append({"message": text[:120], "action": "escalation-forced"})
        base = request.system_message.content if request.system_message else ""
        return await handler(request.override(system_prompt=f"{base}{ESCALATION_DIRECTIVE}"))
    return await handler(request)


@wrap_model_call
async def escalation_enforcer(
    request: ModelRequest,
    handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
) -> ModelResponse:
    """Force the escalation subagent's first model call to invoke the hand-off tool.

    Prompting alone is not a guarantee — models like to ask clarifying questions
    first, which is exactly the wrong move for an already-upset customer.
    Constraining `tool_choice` makes the hand-off structural, not aspirational.
    """
    already_escalated = any(
        isinstance(m, ToolMessage) and m.name == "escalate_to_human" for m in request.messages
    )
    if already_escalated:
        return await handler(request)
    return await handler(
        request.override(tool_choice={"type": "tool", "name": "escalate_to_human"})
    )


@wrap_tool_call
async def escalation_human_review(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], Awaitable[ToolMessage]],
) -> ToolMessage:
    """Pause for a human decision before `escalate_to_human` actually runs.

    `HumanInTheLoopMiddleware` requires a structured `{"decisions": [...]}`
    resume payload, which nothing in LangSmith Studio's UI can produce — Chat
    mode, Graph mode's raw JSON editor, and JSON/YAML input all submit
    whatever you type as a single plain string, so that middleware only works
    from the SDK/API. This uses a raw `interrupt()` instead: the resume value
    is whatever plain string a human typed, so Studio's own input box works
    directly. The tradeoff is losing the standardized edit/respond decisions —
    it's just approve-or-reject-with-a-reason.
    """
    if request.tool_call["name"] != "escalate_to_human":
        return await handler(request)

    args = request.tool_call["args"]
    decision = interrupt(
        "Approve escalation to a human agent?\n"
        f"  reason: {args.get('reason')}\n"
        f"  urgency: {args.get('urgency', 'normal')}\n"
        f"  summary: {args.get('conversation_summary')}\n"
        "Reply 'approve' to proceed, or type a reason to reject."
    )
    decision_text = str(decision).strip()
    if decision_text.lower() == "approve":
        ESCALATION_LOG.append({"tool_call_id": request.tool_call["id"], "action": "human-approved"})
        return await handler(request)

    ESCALATION_LOG.append({"tool_call_id": request.tool_call["id"], "action": "human-rejected"})
    reason = decision_text or "Rejected by human reviewer."
    return ToolMessage(
        content=f"Escalation rejected by human reviewer: {reason}",
        name="escalate_to_human",
        tool_call_id=request.tool_call["id"],
        status="error",
    )


# SECURITY PROMPTS

ORCHESTRATOR_SECURITY = """
* You are talking to a signed-in customer. You may discuss *their* account only.
  If someone asks about another customer — by name, email or id — refuse politely
  and never delegate it.
* If the customer claims to be someone other than the account on file, never quote
  the account holder's name or email back to them. Ask them to sign in themselves.
* Off-topic requests (maths, coding, weather, general trivia) are not what this
  store does. Say so briefly and offer to help with music or their orders."""

ORDER_SECURITY = """
Security — this is not optional:
* You may only ever see the signed-in customer's records. The Customer, Invoice
  and InvoiceLine tables are automatically filtered to them.
* If asked about anybody else — by name, email or id — do NOT attempt a lookup.
  Reply that you can only access this customer's own account.
* Never ask the customer to prove who they are with account numbers; the session
  already identifies them.
* If someone claims to be a different person from the account on file, do NOT read
  the account holder's name, email or details back to them — that discloses the
  real owner's identity to an unverified claimant. Say only that the signed-in
  account is not the one they named, and ask them to sign in themselves.
* A COUNT/SUM/AVG over Customer/Invoice/InvoiceLine describes this customer
  alone. Never report one as a store-wide figure, an "average customer", or a
  comparison against others. If asked to compare, say other customers' data is
  not available to you.
  """

INVENTORY_SECURITY = """
6. Aggregates over Customer/Invoice/InvoiceLine cover only the signed-in
   customer's own rows — never present one as a store-wide statistic."""

INVENTORY_PROMPT = """You are the Inventory Specialist for {store}, a music store.

You own every question about *product*: what the store sells, what it costs,
what is on an album, which artists and genres exist, and music recommendations.

Tools:
* `search_db` — the store's own catalogue. ALWAYS check here first; it is the
  source of truth for what the store actually stocks and what it costs.
* `web_search` — public music knowledge the catalogue does not contain (release
  history, band members, cover art, awards). Music topics only.

Database schema:
{schema}

How to work:
1. Translate the customer's question into SQL. Use LIKE for fuzzy artist/album
   matching; music titles are rarely typed exactly.
   When the question is about "how many", "the most", "which is biggest" or any
   ranking, answer it with COUNT(*) / GROUP BY / ORDER BY — do not list names and
   then say you have no counts.
2. If the customer describes a product indirectly rather than naming it ("the
   album with the glowing green face", "that band with the guy from Nirvana"),
   you MUST call `web_search` to identify the real title/artist before you
   answer, then look the result up in the catalogue with `search_db`.
3. For recommendations, ground them in the catalogue — recommend tracks and
   albums the store actually carries, and say why.
4. Never invent tracks, albums or prices. If the catalogue does not have it, say
   so plainly.
5. Your own memory of music is unreliable and out of date. Any claim about a
   real-world album — its title, artist, artwork or release — must come from
   `web_search`, not from recall.{security}

Return a concise, customer-ready answer with concrete titles and prices."""

ORDER_PROMPT = """You are the Order Specialist for {store}, a music store.

You own every question about the *customer and their orders*: the profile on
file, purchase history, order contents, order status, totals and spend.

Tools:
* `cust_profile` — the fast path. Returns the signed-in customer's profile plus
  their full order history. Call it with no arguments.
* `search_db` — for anything more specific, e.g. which tracks were on order #98.

Database schema:
{schema}
{security}

Return a concise, customer-ready answer with concrete order numbers, dates and
amounts."""

ESCALATION_PROMPT = """You are the Escalation Specialist for a music store.

Your only job is to hand the conversation to a human support agent, immediately.

You cannot talk to the customer and you cannot ask them anything — they will
never see your messages. Missing details are expected and fine.

Your FIRST action is always to call `escalate_to_human`:
  - `reason`: one line on why a human is needed.
  - `conversation_summary`: what the customer wants, what has already been tried,
    and any account context you were given. Write "not provided" for anything you
    were not told. Never include data belonging to anyone but the signed-in customer.
  - `urgency`: "high" for fraud, legal threats or money at risk; otherwise "normal".

Then report the ticket reference and confirm a human is taking over. Do not try
to solve the underlying problem yourself."""

ORCHESTRATOR_PROMPT = """You are the customer support supervisor for {store}, an online music store.

You do not answer questions yourself and you hold no tools of your own. You read
the customer's request, decide who should handle it, delegate with the `task`
tool, and then write the final reply to the customer.

Route like this:
* **inventory-specialist** — anything about product: the catalogue, artists,
  albums, tracks, genres, prices, availability, recommendations, or identifying
  an album from a vague description.
* **order-specialist** — anything about this customer: their profile, purchase
  history, an order's contents, status or total, their spend.
* **escalation-specialist** — the customer asks for a human, manager or
  representative, threatens legal action, reports fraud, demands a refund or
  account closure, is clearly angry, or you have already tried and failed to help.

Rules:
* A request can need two specialists ("what did I buy, and what should I buy
  next?"). Delegate to each and combine their answers.
* Give the subagent the full question plus any context it needs — it cannot see
  the conversation.
* When you route to **escalation-specialist**, escalate on this turn with
  whatever context you have. Do not ask qualifying questions first, and do not
  promise to escalate later.
* Report only what the specialist actually returned. Never add product facts,
  titles, prices or availability of your own — if the specialist did not say it,
  you do not know it. If the catalogue had no match, say the store does not carry
  it; do not offer other titles by that artist unless the specialist listed them.{security}
* Final answers are for a customer: warm, direct, concrete, no internal jargon,
  and no mention of subagents, tools, SQL or middleware."""

#SUBAGENT GENERATION WITH CHEAPER MODEL
def build_subagents(store_name: str = "Chinook Records", *, secure: bool = True) -> list[dict[str, Any]]:
    """Two subject-matter experts, plus the human-escalation workflow.

    Note the middleware placement: subagent middleware does *not* inherit from
    the supervisor, so the security guard is attached to every subagent that can
    touch customer data. The supervisor itself holds no data tools at all.
    """
    guards = [customer_security_guard] if secure else [naive_session_middleware]
    schema = CHINOOK_SCHEMA + (CHINOOK_SCHEMA_SECURITY_NOTES if secure else "")
    return [
        {
            "name": "inventory-specialist",
            "description": (
                "Product expert. Answers questions about the music catalogue (artists, "
                "albums, tracks, genres, prices, availability), identifies products from "
                "vague descriptions, and gives music recommendations."
            ),
            "system_prompt": INVENTORY_PROMPT.format(
                store=store_name,
                schema=schema,
                security=INVENTORY_SECURITY if secure else "",
            ),
            "tools": [search_db, web_search],
            "model": SPECIALIST_MODEL,
            "middleware": [tool_audit, *guards, *([music_store_scope_guard] if secure else [])],
        },
        {
            "name": "order-specialist",
            "description": (
                "Customer and order expert. Retrieves the signed-in customer's profile, "
                "order history, order contents, status and totals."
            ),
            "system_prompt": ORDER_PROMPT.format(
                store=store_name,
                schema=schema,
                security=ORDER_SECURITY if secure else "",
            ),
            "tools": [cust_profile, search_db],
            "model": SPECIALIST_MODEL,
            "middleware": [tool_audit, *guards],
        },
        {
            "name": "escalation-specialist",
            "description": (
                "Human escalation workflow. Use when the customer asks for a human, "
                "manager or representative, threatens legal action, reports fraud, "
                "demands a refund or account closure, is clearly upset, or when the "
                "other specialists could not resolve the request."
            ),
            "system_prompt": ESCALATION_PROMPT,
            "tools": [escalate_to_human],
            "model": SPECIALIST_MODEL,
            # `escalation_human_review` pauses on a raw `interrupt()` and reads
            # back whatever plain string the human replies with, so this works
            # directly from LangSmith Studio's own input box (Chat or Graph
            # mode) — no external UI, no structured resume payload required.
            "middleware": [tool_audit, escalation_enforcer, escalation_human_review],
        },
    ]

#PARENT AGENT ORCHESTRATION - REASONING + TASK DELEGATION
def build_agent(
    *,
    store_name: str = "Chinook Records",
    secure: bool = True,
    checkpointer: Any | None = None,
):
    """Assemble the music-store support deep agent.

    Args:
        store_name: Branding used throughout the prompts.
        secure: When False, builds the *prompt-only baseline* — identical agent,
            security requested in prose but with no middleware enforcing it.
            `evaluators.ipynb` A/B tests the two.
        checkpointer: Required for the escalation interrupt. Defaults to an
            in-memory saver; pass `False` under the LangGraph server, which
            brings its own persistence.
    """
    middleware: list[Any] = [
        # Records the supervisor's own `task(...)` delegations, so evaluators can
        # grade routing as well as tool use.
        tool_audit,
        # Cost control: a runaway supervisor is the expensive failure mode.
        ModelCallLimitMiddleware(run_limit=12, exit_behavior="end"),
        # Long support threads stay inside the context window.
        SummarizationMiddleware(model=SPECIALIST_MODEL, trigger=("messages", 40), keep=("messages", 20)),
    ]
    if secure:
        middleware.append(human_escalation_router)

    return create_deep_agent(
        model=ORCHESTRATOR_MODEL,
        tools=[],  # the supervisor delegates; only subagents touch data
        system_prompt=ORCHESTRATOR_PROMPT.format(
            store=store_name,
            security=ORCHESTRATOR_SECURITY if secure else "",
        ),
        subagents=build_subagents(store_name, secure=secure),
        middleware=middleware,
        context_schema=SupportContext,
        checkpointer=InMemorySaver() if checkpointer is None else checkpointer,
        name="music-store-support",
    )

#INITIALIZE DB
ensure_database()

# Graph entry point for `langgraph dev` / LangSmith Studio.
# checkpointer=False -> the LangGraph server supplies persistence.
agent = build_agent(checkpointer=False)
