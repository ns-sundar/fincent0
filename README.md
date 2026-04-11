# Fincent

Fincent is a **framework-style** multi-agent financial assistant: a **hub-and-spoke** LangGraph workflow with a **Streamlit** UI and **OpenAI** models (default **`gpt-4o-mini`**). The codebase is split into small modules with **type hints** and clear function boundaries.

## What it does

- **Central hub (router / planner)** reads the conversation and classifies intent. It either:
  - routes to the **Q&A spoke** for **generic educational** questions about finance, markets, regulations, and financial documents (answered from **pretrained knowledge only**), or
  - **declines** requests that imply live market data, personal portfolio guidance, trading or security-specific advice, or individualized financial / tax / legal planning (when in doubt, it declines).
- **Financial documents Q&A agent** is the only spoke today: it answers **without RAG** (no document retrieval) using general knowledge and a dedicated system prompt.
- **Multi-turn memory** is the chat transcript stored in **Streamlit session state**; each turn invokes the graph with the full message list so context is preserved for the hub and the Q&A agent.

```mermaid
flowchart LR
  START([START]) --> Hub[hub_route]
  Hub -->|qa| QA[qa_financial_docs]
  Hub -->|decline| Decline[hub_decline]
  QA --> END([END])
  Decline --> END
```

## Tech stack

| Piece | Role |
|--------|------|
| [LangGraph](https://github.com/langchain-ai/langgraph) | `StateGraph`: route → spoke or decline → end |
| [LangChain OpenAI](https://python.langchain.com/) | `ChatOpenAI`, structured output for routing |
| [Streamlit](https://streamlit.io/) | Sidebar (API key, model, clear chat), chat UI |
| OpenAI API | Default model `gpt-4o-mini` (configurable) |

## Project layout

| Path | Purpose |
|------|---------|
| `streamlit_app.py` | UI: session transcript, cached compiled graph, one turn per user message |
| `graph/workflow.py` | Builds and compiles the hub-and-spoke graph |
| `agents/hub/` | Routing (`planner`, `nodes`), decline copy (`decline`, `prompts`) |
| `agents/qa/` | Q&A spoke prompts and LangGraph node factory |
| `state/` | `FincentState` (messages + route), UI ↔ LangChain message adapters |
| `config/` | `AppSettings`, env-backed defaults, `make_chat_model()` |
| `requirements.txt` | Python dependencies |

## Install

Use Python 3.11+ (3.12 recommended). From this directory:

```bash
cd fincent
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

Dependencies include `streamlit`, `langgraph`, `langchain-openai`, `langchain-core`, and `pydantic`.

## Usage

1. Set your API key (either is fine):

   ```bash
   export OPENAI_API_KEY="your_key_here"
   ```

   Or paste the key in the app sidebar (stored only for that browser session).

2. Run the app **from the `fincent` directory** so package imports (`config`, `agents`, `graph`, `state`) resolve:

   ```bash
   cd fincent
   streamlit run streamlit_app.py
   ```

3. Open the local URL Streamlit prints (usually [http://localhost:8501](http://localhost:8501)).

4. Use **Clear conversation** in the sidebar to reset session memory.

### Optional environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model for hub and Q&A spoke |
| `FINCENT_ROUTER_TEMPERATURE` | `0` | Sampling temperature for the router LLM |
| `FINCENT_QA_TEMPERATURE` | `0.2` | Sampling temperature for the Q&A spoke |

Example:

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-4o-mini"
export FINCENT_QA_TEMPERATURE="0.2"
streamlit run streamlit_app.py
```

## Design notes

- **Hub-and-spoke orchestration** is implemented as explicit graph nodes and conditional edges in `graph/workflow.py`, not as ad-hoc `if` chains in the UI.
- **Typing**: shared state uses `FincentState` (extends LangGraph `MessagesState` with a `route` field); Pydantic models support structured routing decisions in `agents/hub/planner.py`.
- **Extending**: add a new folder under `agents/` for another spoke, register a node in `graph/workflow.py`, and extend the router schema/prompt in `agents/hub/` so the hub can branch to the new path.

## Limitations (by design)

- No retrieval / RAG and no access to private filings unless the user pastes text in chat.
- No personalized investment, tax, or legal advice; out-of-scope topics receive a fixed-style decline from the hub path.
- Memory is **session-scoped** (Streamlit); persisting threads across sessions would require a checkpointer or external store (not included here).
