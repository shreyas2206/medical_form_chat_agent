# Medical Form Chat Agent — Three Approaches

A step-by-step demonstration of building a **structured, deterministic conversational agent** using three progressively more idiomatic approaches. The domain is a medical intake chatbot that fills a standard GP questionnaire by chatting with a patient.

The same architecture is implemented three times, each with full sequential logging so you can watch every node, state transition, routing decision, LLM call, and security check as they happen.

---

## What the Agent Does

The agent conducts a structured patient intake conversation. It collects eight fields — name, date of birth, chief complaint, symptom duration, pain level, medications, allergies, and smoking status — in a fixed order, then produces a completed form.

The key design constraint is that **the LLM never controls the flow**. It only handles natural language: phrasing questions, extracting values, and clarifying confusion. All routing, ordering, and validation is deterministic Python.

```
[Decide next field] → [Generate question] → [Wait for patient]
         ↑                                          ↓
         │               [Sanitize input] ──────────┘
         │              ╱        ╲         ╲
         │        [Reject]    [Clarify]  [Extract & validate]
         │      (injection)  (ReAct sandbox)   ↙  ↓  ↘
         └────────────────────────────── [Decide] [Re-ask] [Conclude]
```

### Security model

Every user message passes through a two-pass sanitiser before any reasoning LLM sees it:

- **Pass 1 — regex**: blocks known prompt injection patterns instantly, at zero cost
- **Pass 2 — LLM classifier**: a fast, cheap model catches novel injection attempts

The ReAct clarification sub-loop (for when a patient doesn't understand a question) is sandboxed: the tool whitelist is enforced in Python, not by prompt. The LLM cannot call any tool outside `[explain_medical_term, rephrase_question, give_example]`, and it cannot write directly to the form.

---

## Repository Structure

```
.
├── README.md                          ← you are here
│
├── 01_langchain/
│   └── medical_form_agent.py          ← Approach 1: LangChain only
│
├── 02_langgraph_debug/
│   └── medical_form_agent_debug.py    ← Approach 2: LangGraph + manual logging
│
└── 03_langgraph_native/
    └── medical_form_agent_langgraph.py ← Approach 3: LangGraph native patterns
```

> The Python files will be added to this repository after the README is in place.

---

## Getting Started

### Prerequisites

- **Python 3.11+** (installed on your system)
- **`uv` package manager** (lightweight, fast Python project manager)

### Setup

1. **Install `uv` if you don't have it:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
   Or visit [uv docs](https://docs.astral.sh/uv/) for other installation methods.

2. **Clone this repository and navigate to it:**
   ```bash
   git clone <repo-url>
   cd medical_form_chat_agent
   ```

3. **Sync dependencies and set up the virtual environment:**
   ```bash
   uv sync
   ```
   This creates a `.venv/` directory with all required packages (LangChain, LangGraph, provider SDKs, etc.).

4. **Create a `.env` file in the project root with your API keys:**
   ```bash
   cp .env.example .env  # or create it manually
   ```
   Then edit `.env` and add the keys you'll use. Example:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   OPENAI_API_KEY=sk-...
   GOOGLE_API_KEY=...
   GROQ_API_KEY=...
   ```
   Only include keys for providers you plan to use. The smoke test will skip any provider with a missing key.

5. **Run the smoke test to verify setup:**
   ```bash
   uv run python smoke_test.py
   ```
   Or test a single provider:
   ```bash
   uv run python smoke_test.py --provider anthropic
   ```

---

## The Three Approaches

### Approach 1 — LangChain only (`01_langchain/`)

Implements the agent as a set of plain Python functions and classes, wired together manually. Uses LangChain solely for LLM calls and tool decoration — no graph framework.

The conversation loop, routing, and state management are all explicit code in a `run_agent()` function. A `manual_resume()` dispatcher re-calls nodes in the correct order each time the user sends a message.

**What you learn here:** the underlying mechanics — what a graph framework is actually doing for you. Every routing decision, every state mutation, every pause-and-resume is visible as plain Python.

**Logging approach:** a `StepLogger` class is threaded into every node function as a parameter. Each node calls `log.node_enter()`, `log.llm_call()`, `log.node_exit()` etc. explicitly. The log is a side-effect of the node running.

**Workarounds required:**
- `END` after `ask`/`clarify`/`reject` nodes to simulate pausing
- `manual_resume()` to re-enter the pipeline at the right point
- `log` parameter added to every node, breaking the pure `(state) → state` contract

---

### Approach 2 — LangGraph + debug logging (`02_langgraph_debug/`)

Refactors Approach 1 into a proper `StateGraph`. Nodes become pure `(state) → state` functions. Conditional routing is declared as graph edges rather than if-statements in a dispatcher.

Still uses `END` as a pause mechanism (the LangGraph-compatible workaround before `interrupt()` was available), and still falls back to `manual_resume()` for re-entry — but now the graph structure documents the flow clearly, and the routing is declared once rather than duplicated.

**What you learn here:** how `StateGraph`, `add_conditional_edges`, and shared state work. The gap between "using LangGraph" and "using LangGraph idiomatically."

**Logging approach:** same `StepLogger` class, still threaded into nodes. Nodes are wrapped in closures inside `build_graph()` to inject the logger without changing the LangGraph-required signature.

**Remaining workarounds:**
- `END` pause + `manual_resume()` dispatcher still present
- Logger injected via closure rather than being truly decoupled

---

### Approach 3 — LangGraph native (`03_langgraph_native/`)

Eliminates all workarounds. Uses three LangGraph-native patterns that the previous approaches worked around:

| Workaround (Approaches 1 & 2) | LangGraph-native replacement |
|---|---|
| `END` after ask/clarify/reject | `interrupt()` inside `node_get_input` |
| `manual_resume()` dispatcher | `Command(resume=user_text)` |
| `log` threaded into every node | `app.stream(stream_mode="updates")` |
| Manual state dict passed per turn | `MemorySaver` checkpointer + `thread_id` |

Nodes are pure again. The `StreamLogger` is completely decoupled — it reads `_log_*` side-channel keys from stream chunks, so zero changes to node logic are needed to add or remove logging.

**What you learn here:** the idiomatic LangGraph pattern for human-in-the-loop agents. This is the approach you would use in production or when deploying with LangGraph Cloud.

**Logging approach:** `StreamLogger.on_chunk()` consumes `{node_name: {changed_keys}}` chunks from `app.stream()`. Nodes write structured debug info into `_log_*` state keys; the logger renders them. Neither side knows about the other.

---
