# Medical Form Chat Agent

A structured deterministic medical intake chatbot implemented three ways, each
progressively more idiomatic with LangGraph.

## Commands

```bash
uv sync                                              # install / refresh deps
uv run python smoke_test.py                          # verify all LLM providers
uv run python smoke_test.py --provider anthropic     # single provider
```

## Project structure

Three implementations, built in order:

```
01_langchain/medical_form_agent.py              # plain Python + LangChain LLM calls
02_langgraph_debug/medical_form_agent_debug.py  # StateGraph + manual StepLogger
03_langgraph_native/medical_form_agent_langgraph.py  # interrupt(), Command, stream
```

None of these files exist yet — they are to be built.

## Architecture invariants

- **LLM never controls flow.** The LLM only does NL work: phrasing questions,
  extracting values, clarifying confusion. All routing and ordering is deterministic Python.
- **Node functions are pure** `(state) → state`. No injected parameters, no side effects.
- **Fixed field order**: name → dob → chief_complaint → symptom_duration → pain_level
  → medications → allergies → smoking_status
- **Two-pass sanitiser** runs before any reasoning LLM sees user input:
  - Pass 1 — regex: blocks known prompt injection patterns instantly
  - Pass 2 — cheap LLM classifier: catches novel injection attempts
- **ReAct sub-loop is sandboxed in Python**, not by prompt. Allowed tools:
  `explain_medical_term`, `rephrase_question`, `give_example`. The LLM cannot
  write directly to the form.

## Logging patterns (per approach)

| Approach | Pattern |
|---|---|
| 01_langchain | `StepLogger` passed as param; nodes call `log.node_enter()`, `log.llm_call()`, etc. |
| 02_langgraph_debug | Same `StepLogger`, injected via closure inside `build_graph()` |
| 03_langgraph_native | `StreamLogger.on_chunk()` reads `_log_*` side-channel keys from `app.stream()` — fully decoupled from node logic |

## Key LangGraph patterns per approach

| Workaround (01 & 02) | Native replacement (03) |
|---|---|
| `END` after ask/clarify/reject | `interrupt()` inside `node_get_input` |
| `manual_resume()` dispatcher | `Command(resume=user_text)` |
| logger threaded into nodes | `app.stream(stream_mode="updates")` |
| manual state dict per turn | `MemorySaver` checkpointer + `thread_id` |

## Environment

- Python 3.11+, managed with `uv`
- `.env` at project root holds API keys (gitignored)
- Supported providers: `anthropic`, `openai`, `google`, `groq`
- Smoke test skips any provider with a missing key
