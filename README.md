# Music Store Support Agent

A customer-support **deep agent** for a fictional online record store, built on
[LangChain Deep Agents](https://docs.langchain.com/oss/python/deepagents/overview) + LangGraph
over the [Chinook](https://github.com/lerocha/chinook-database) sample database, with a
security-hardened middleware stack and a full [LangSmith](https://smith.langchain.com)
evaluation harness that proves the hardening matters.

This README is written to be read in under 5 minutes: what the pieces are, what `agent.py`
actually does method-by-method, and what the evaluators measure.

---

## Repo Map

| File / Folder | What it is |
|---|---|
| `src/agent/agent.py` | **The agent.** Tools, middleware, subagents, prompts, graph assembly. Detailed below. |
| `agent.ipynb`, `src/agent/agent.ipynb` | Notebook version of `agent.py` — same logic, notebook form. Not detailed separately here. |
| `evaluators.ipynb` | **The evaluation harness.** Dataset, 11 evaluators, two experiments, pairwise comparison. Detailed below. |
| `langgraph.json` | Points the LangGraph CLI/Studio at `src/agent/agent.py:agent`. |
| `data/chinook.db`, `data/Chinook_Sqlite.sql` | The sample music-store database (artists, albums, tracks, customers, invoices, employees). Built on first run. |
| `scripts/create_assistants.py` | Creates one LangSmith Studio "assistant" per demo persona (signed-in customer A, customer B, signed-out), so a persona can be picked from a dropdown instead of hand-editing context. |
| `pyproject.toml`, `uv.lock` | Dependencies (`deepagents`, `langchain`, `langgraph-cli`, `langsmith`, `pandas`, …), managed with `uv`. |
| `src/music_store_workshop/__init__.py` | Boilerplate package stub from `uv init`; not part of the agent. |

---

## The Flow, in 30 Seconds

```
customer message
      │
      ▼
 SUPERVISOR (Sonnet 5, no tools of its own)
      │  reads the request, decides who should handle it,
      │  delegates via the `task` tool, writes the final reply
      │
      ├──► inventory-specialist  (Haiku)  — catalogue, prices, recommendations
      │        tools: search_db, web_search
      │
      ├──► order-specialist      (Haiku)  — this customer's profile & orders
      │        tools: cust_profile, search_db
      │
      └──► escalation-specialist (Haiku)  — hand off to a human
               tool: escalate_to_human → pauses on interrupt() for approval
```

Every tool call from every subagent passes through **middleware** first: an audit logger
(so evaluators can grade the trajectory, not just the final text), and — in the "hardened"
configuration — a security guard that pins every customer-data query to the *signed-in*
customer, regardless of what the model asks for or what the customer claims in chat.

---

## `agent.py` — Method Reference

Every function, grouped by what it's for, described in plain terms.

### Setup & Database

| Method | What it does |
|---|---|
| `_find_project_root()` | Finds the repo root by locating `pyproject.toml` upward. |
| `ensure_database(force=False)` | Downloads the Chinook SQL dump, builds a local SQLite file. |

### Session Context

`SupportContext` is a small data object carrying the *signed-in customer id* for a run — set by
the app/Studio/eval harness, never by the model itself.

| Method | What it does |
|---|---|
| `_session_customer_id(runtime)` | Reads the signed-in customer id from runtime context. |
| `_session_audit_id(runtime)` | Reads the optional audit-trail id from runtime context. |
| `extract_text(message)` | Flattens a message's content blocks into plain text. |
| `final_answer(result)` | Returns the last message's text as the customer reply. |
| `_open_scoped_connection(customer_scope)` | Opens a read-only DB connection, optionally scoped to one customer. |

### Tools (callable by the model)

| Method | What it does |
|---|---|
| `search_db(sql, customer_scope)` | Tool: runs a read-only SQL query against the database. |
| `cust_profile(customer_id, include_orders)` | Tool: returns the customer's profile plus order history. |
| `_anthropic()` | Lazily builds a shared synchronous Anthropic client. |
| `_async_anthropic()` | Lazily builds a shared asynchronous Anthropic client. |
| `web_search(query)` | Tool: answers music questions using Claude's web search. |
| `escalate_to_human(reason, conversation_summary, urgency)` | Tool: hands the conversation off to a human agent. |

### Middleware — Audit & Security

| Method | What it does |
|---|---|
| `tool_audit(request, handler)` | Middleware: logs every tool call and its result. |
| `audit_trail(audit_id)` | Returns every tool call logged for one run. |
| `_deny(request, message, rule)` | Builds a denial message and logs the block. |
| `_leaks_foreign_customer(content, session_id)` | Detects another customer's id inside a result table. |
| `customer_security_guard(request, handler)` | Middleware: enforces single-customer data isolation in five layers. |
| `naive_session_middleware(request, handler)` | Middleware: unsafe baseline — fills in customer id only. |
| `is_music_related(query)` | Classifies whether a search query is music-related. |
| `music_store_scope_guard(request, handler)` | Middleware: blocks web searches that are off-topic. |

### Middleware — Escalation Workflow

| Method | What it does |
|---|---|
| `human_escalation_router(request, handler)` | Middleware: forces escalation when the message matches trigger phrases. |
| `escalation_enforcer(request, handler)` | Middleware: forces the first escalation call to fire. |
| `escalation_human_review(request, handler)` | Middleware: pauses for human approval before escalating. |

### Agent Assembly

| Method | What it does |
|---|---|
| `build_subagents(store_name, secure)` | Builds the inventory, order, and escalation subagent configs. |
| `build_agent(store_name, secure, checkpointer)` | Assembles the full supervisor agent graph and middleware. |

**The `secure` flag is the whole experiment.** `build_agent(secure=True)` installs
`customer_security_guard` + `music_store_scope_guard`; `build_agent(secure=False)` swaps in
`naive_session_middleware` and asks for the same guarantees in prose instead. Same agent,
same prompts otherwise — one variant enforces isolation in code, the other just asks nicely.
`evaluators.ipynb` runs both through the same dataset to measure the difference.

**Model policy:** cheapest-first. Every specialist, every guard classifier, and the web-search
executor run on Haiku 4.5. Only the supervisor (`ORCHESTRATOR_MODEL`, Sonnet 5) uses a stronger
model, because routing/judgement is its whole job.

---

## `evaluators.ipynb` — What Gets Measured

Two agents (`hardened` = middleware on, `prompt-only` = middleware off) are run over the same
**18-example dataset** split into four categories: `product`, `orders`, `security`,
`escalation`. Each example carries a reference: the expected specialist route, facts that must
(or must not) appear, whether escalation is expected, a tool-call budget, and a rubric.

Each run is executed through `run_agent()`, which returns not just the final answer but the
whole **trajectory**: which subagents were routed to, every tool called, every raw tool result,
and any access denials — because a reply can be flawless while its trace shows out-of-scope
data was loaded into context along the way. That distinction is the entire point of the harness.

### Code evaluators (deterministic)

| Evaluator | Axis | Checks |
|---|---|---|
| `no_foreign_data_retrieved` | Security (trajectory) | Did another customer's/staff's data ever enter a tool result? |
| `no_pii_in_answer` | Security (output) | Did out-of-scope data reach the final answer text? |
| `routing_correct` | Reasoning | Did the supervisor delegate to the expected specialist(s)? |
| `escalation_correct` | Reasoning | Did it escalate exactly when — and only when — expected? |
| `trajectory_efficiency` | Reasoning | Did it stay within the example's tool-call budget? |
| `contains_expected_facts` | Quality | Are the required facts present and the forbidden ones absent? |
| `customer_ready` | Quality | Did internal jargon (SQL, tool names, "ACCESS DENIED") leak into the reply? |

### LLM-as-judge evaluators

| Evaluator | Model | Checks |
|---|---|---|
| `security_policy` | Sonnet (strict) | Nuanced policy compliance a regex can't express — e.g. confirming another account exists. |
| `reasoning_quality` | Sonnet (strict) | Does the trajectory actually support the conclusions, without guessing? |
| `groundedness` | Haiku | Does every factual claim trace back to an actual tool result? |
| `response_quality` | Haiku | Is the reply helpful, concrete, and appropriately toned? |

### Summary evaluators (whole-experiment)

| Evaluator | Reports |
|---|---|
| `zero_leak_runs` | All-or-nothing gate — 1.0 only if **no** run in the whole suite leaked. |
| `mean_tool_calls` | Average trajectory length, a cost proxy. |
| `mean_latency_s` | Average wall-clock time per run. |

### Also in the notebook

- **Pairwise comparison** (`prefer_better_answer`) — a judge picks the better of the two
  agents' replies to the same question, head-to-head, independent of absolute scores.
- **Per-split breakdown**, a leak drill-down naming exactly which strings escaped, programmatic
  human feedback, an annotation queue for low-scoring runs, and token/cost analytics.

### The headline result

| metric | prompt-only | hardened | delta |
|---|---|---|---|
| `no_foreign_data_retrieved` | 0.944 | 1.000 | +0.056 |
| `no_pii_in_answer` | 0.944 | 1.000 | +0.056 |
| `contains_expected_facts` | 0.833 | 1.000 | +0.167 |
| `security_policy` | 0.833 | 0.944 | +0.111 |
| `groundedness` | 0.833 | 0.933 | +0.100 |
| `response_quality` | 0.828 | 0.924 | +0.096 |
| `reasoning_quality` | 0.772 | 0.856 | +0.084 |

The prompt-only baseline *sounds* fine — its final answers are almost always polite and
on-topic. But asked for "my support rep's email," it joins the `Employee` table and pulls a
staff address into context, then decides afterward how much to repeat. Grading only the final
text would miss that; grading the trajectory doesn't. The hardened agent never receives the row
in the first place — the guard denies at the tool boundary, not the reply boundary.

Full run: 36 agent invocations, 11 evaluators each, **$0.72** — cheap enough to re-run per prompt change.
