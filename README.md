# Medical Form Chat Agent — Three Approaches

A step-by-step demonstration of building a **structured, deterministic conversational
agent** using three progressively different approaches. The domain is a medical intake
chatbot that fills a standard GP questionnaire by chatting with a patient.

The same architecture is implemented three times so you can compare how the same
problem is solved at different levels of abstraction — and understand what each layer
is actually buying you.

---

## What the Agent Does

The agent conducts a structured patient intake conversation. It collects eight fields
in a fixed order, then produces a completed form:

```
name → date of birth → chief complaint → symptom duration →
pain level → medications → allergies → smoking status
```

The key design constraint is that **the LLM never controls the flow**. It only handles
natural language: phrasing questions, extracting values, and clarifying confusion. All
routing, ordering, and validation is deterministic Python.

```
[Decide next field] → [Generate question] → [Wait for patient]
         ↑                                          ↓
         │               [Sanitize input] ──────────┘
         │              ╱        ╲         ╲
         │        [Reject]    [Clarify]   [Extract]
         │      (injection)  (ReAct loop)  ↙  ↓  ↘
         └──────────────────────── [Next] [Reask] [Conclude]
```

### Security model

Every user message passes through a two-pass sanitiser before any reasoning LLM sees it:

- **Pass 1 — regex**: blocks known prompt injection patterns instantly, at zero cost
- **Pass 2 — LLM classifier**: a fast, cheap model catches novel injection attempts

The ReAct clarification sub-loop is sandboxed: the tool whitelist is enforced via
`bind_tools` (Approaches 1 & 2) or a Python dict check (Approach 3) — not by prompt.
The LLM cannot call anything outside `[explain_medical_term, rephrase_question,
give_example]`, and cannot write directly to the form.

---

## Repository Structure

```
.
├── README.md
├── pyproject.toml                          ← uv-managed deps for all three approaches
├── smoke_test.py                           ← verify LLM provider connectivity
├── .env.example                            ← API key template
│
├── 01_langchain/
│   └── medical_form_agent.py              ← Approach 1: LangChain only
│
├── 02_langgraph/
│   └── medical_form_agent.py              ← Approach 2: LangGraph + StateGraph
│
└── 03_anthropic_sdk/
    └── medical_form_agent.py              ← Approach 3: Anthropic SDK, no framework
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# 1. Clone and enter the repo
git clone https://github.com/shreyas2206/medical_form_chat_agent
cd medical_form_chat_agent

# 2. Install all dependencies
uv sync

# 3. Set up API keys
cp .env.example .env   # then edit .env and add your keys
```

`.env` keys — include only the providers you plan to use:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
GROQ_API_KEY=...
```

```bash
# 4. Verify connectivity
uv run python smoke_test.py                       # all providers with keys present
uv run python smoke_test.py --provider anthropic  # single provider
```

### Running each approach

All three scripts share the same CLI flags:

```bash
uv run python 01_langchain/medical_form_agent.py
uv run python 02_langgraph/medical_form_agent.py
uv run python 03_anthropic_sdk/medical_form_agent.py

# Flags (all three):
#   --provider  anthropic | openai | google | groq   (default: anthropic)
#   --model     override the reasoning model
#   --quiet     suppress step-by-step logging
```

> **Note:** `03_anthropic_sdk` uses the `anthropic` package directly and is
> Claude-only. The `--provider` flag is not available there.

---

## The Three Approaches

### Approach 1 — LangChain only (`01_langchain/`)

The agent is a `MedicalFormAgent` class with two public methods: `start()` returns the
first question; `reply(user_input)` processes one patient message and returns the next
utterance. Internally it owns an `AgentState` dataclass and calls node functions in
sequence.

LangChain is used for exactly two things: a provider-agnostic `BaseChatModel` for all
LLM calls, and `bind_tools` inside the ReAct clarification loop so the whitelist is
enforced at the API level. No graph framework is involved.

Provider switching is first-class: an `AgentConfig` dataclass with a `Provider` enum
(`anthropic`, `openai`, `google`, `groq`) holds sensible model defaults per provider,
and the `--model` CLI flag overrides the reasoning model.

**What you learn here:** the underlying mechanics — what a graph framework is actually
doing for you. Every routing decision, state mutation, and pause-and-resume is visible
as explicit Python inside `reply()`.

**Logging approach:** a `StepLogger` is passed into every node function as a parameter.
Each node calls `log.node_enter()`, `log.llm_call()`, `log.routing()`,
`log.node_exit()` etc. explicitly. The log is a side-effect of the node running.

**Workarounds present — intentional, to motivate Approach 2:**

- `reply()` is a manual dispatcher: it decides which nodes to call after each patient
  message, duplicating the logic that a graph would express declaratively as edges
- `StepLogger` is threaded into every node as a parameter, coupling logging to node
  signatures

---

### Approach 2 — LangGraph (`02_langgraph/`)

Refactors Approach 1 into a `StateGraph`. Node functions become pure — they take state
and return updates, with no injected parameters. Routing is declared once as conditional
edges rather than duplicated in a dispatcher. `MedicalFormAgent` now wraps the compiled
graph rather than calling nodes directly.

`AgentConfig`, all node logic, and `StepLogger` carry over unchanged. The graph
topology makes the flow inspectable and makes it structurally impossible to skip the
sanitiser or extractor — the edges enforce it.

**What you learn here:** how `StateGraph`, `add_conditional_edges`, and shared typed
state work. Also: the gap between "using LangGraph" and "using LangGraph idiomatically."

**Logging approach:** the same `StepLogger`, injected via closures inside
`build_graph()` rather than threaded through node signatures. Nodes remain pure from
LangGraph's perspective; the logger wraps them from outside.

**Workarounds still present — what full LangGraph idiom would replace:**

| Workaround in this approach | LangGraph-native replacement |
|---|---|
| `END` after ask / clarify / reject | `interrupt()` inside a `get_input` node |
| `manual_resume()` dispatcher on each turn | `Command(resume=user_text)` |
| logger injected via closure | `app.stream(stream_mode="updates")` + side-channel keys |
| manual state dict passed per turn | `MemorySaver` checkpointer + `thread_id` |

These native patterns are documented here but not implemented — the goal of this repo
is to understand the building blocks, not to demonstrate LangGraph Cloud deployment.

---

### Approach 3 — Anthropic SDK, no framework (`03_anthropic_sdk/`)

The same agent rebuilt using `import anthropic` directly — no LangChain, no LangGraph,
no framework of any kind. LLM calls go to `client.messages.create()`. State is a plain
Python dataclass. The conversation loop is a `while` loop you can read top to bottom.

The `MedicalFormAgent` class and `start()` / `reply()` interface are kept identical to
Approach 1 so the difference is purely internal. Tool calling uses the Anthropic API's
native `tools` parameter and `tool_use` / `tool_result` message blocks directly.

**What you learn here:** what the frameworks were actually doing. Without LangGraph you
write the dispatcher yourself; without LangChain you write the message dicts yourself.
Neither is much code — which reveals that frameworks earn their keep not at small scale
but when you need checkpointing, streaming, graph visualisation, or multi-agent
coordination.

**Logging approach:** same `StepLogger` threaded into nodes as Approach 1. Without a
streaming infrastructure to observe from outside, the logger is necessarily a
side-effect inside each node — same workaround, same reason.

**What is simpler than Approach 1:**

- Plain `{"role": ..., "content": ...}` dicts instead of `SystemMessage` /
  `HumanMessage` wrapper objects
- No graph to declare or wire; control flow is one readable `reply()` method
- No framework version pinning or abstraction layer to understand

**What is more manual than Approach 1:**

- Tool definitions are raw JSON schemas rather than `@tool`-decorated functions
- Tool results are `{"type": "tool_result", ...}` content blocks rather than
  `ToolMessage` objects
- No `bind_tools` — the whitelist is a plain `if name in ALLOWED_TOOLS` check

---

## Reading the Log Output

All three approaches emit the same structured trace per turn (unless `--quiet` is set):

```
┌─ [sanitize]
│  LLM(injection_classifier): safe=True reason=plain medical answer
└─ [sanitize] done
   → safe

┌─ [extract]
│  LLM(extract): {"extracted": true, "value": "Jane Smith", ...}
│  ✓ name = 'Jane Smith'
└─ [extract] done
   → extracted

┌─ [decide_field]
   → next field = dob
└─ [decide_field] done

┌─ [generate_question]
│  LLM(generate_question): Could you tell me your date of birth?
└─ [generate_question] done

Agent: Could you tell me your date of birth?
```

For a **confusion turn** (e.g. "what do you mean by chief complaint?") the route forks
to `clarify`, the ReAct loop runs up to three steps, and each tool call is logged with
its name, input, and result.

For an **injection attempt** the sanitiser logs which pass caught it (regex or LLM
classifier) and the session is terminated.

---

## Architecture Notes

### Why not a ReAct agent for the whole thing?

A ReAct agent lets the LLM decide which tool to call at every step. That is the right
pattern when the number of steps is unpredictable and the goal is open-ended. Here the
opposite is true: there are exactly eight fields, they must be collected in order, and
every answer must pass validation before the form advances. Giving the LLM control of
the flow would mean it could skip fields, re-order questions, or fail to validate.

### Why keep ReAct at all?

The clarification sub-loop is a good fit for limited ReAct. The number of exchanges
needed to clarify a single question is unpredictable, the LLM needs to choose between
explaining a term, rephrasing, or giving an example, and the consequences of a wrong
tool choice are low. The sandbox — a whitelist enforced in Python and a hard cap of
three steps — keeps it safe.

### Why three approaches and not just the best one?

Because "best" depends on what you're optimising for. Approach 1 is the most readable
when learning the pattern. Approach 2 is most appropriate when the pipeline grows and
you want the graph to enforce the routing. Approach 3 is the most transparent about
what is happening at the API level — useful for debugging, cost analysis, or when
adding a framework dependency is not justified. Reading all three in order shows what
each abstraction layer costs and what it gives back.

---

## Extending the Agent

**Add a field:** add an entry to `FIELDS` and `FIELD_LABELS`. The order in `FIELDS`
determines collection order. No other changes needed.

**Change the LLM provider at the CLI** (Approaches 1 & 2):

```bash
uv run python 01_langchain/medical_form_agent.py --provider openai --model gpt-4o
uv run python 01_langchain/medical_form_agent.py --provider groq
```

**Change the provider in code** (Approaches 1 & 2):

```python
from medical_form_agent import MedicalFormAgent, AgentConfig, Provider

agent = MedicalFormAgent(config=AgentConfig(provider=Provider.OPENAI))
```

**Deploy as an API:** replace the `input()` loop with a FastAPI endpoint. Receive
`user_input`, call `agent.reply(user_input)`, and return the result. Because
`MedicalFormAgent` holds state in memory, store one instance per session — keyed by
session ID in a dict, or serialise `AgentState` to a database between requests.