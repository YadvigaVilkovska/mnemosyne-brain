# Mnemosyne Brain

Clean-room Brain Orchestrator MVP v0.4.2.

## Setup

Use Python 3.11+.

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Run Tests

```bash
python3 -m unittest discover -s tests -v
```

## Run Demo

```bash
python3 -m mnemosyne_brain.app.run_demo
```

## Send One Local Message

```bash
python3 -m mnemosyne_brain.app.cli "Remember that Pav loves architecture diagrams"
```

## Debug LLM Context

```bash
python3 -m mnemosyne_brain.app.context_debug stage1 --track-id <track_id> --message "test message"
python3 -m mnemosyne_brain.app.context_debug stage2 --track-id <track_id> --message "test message" --memory-id mem_1 --memory-id mem_2
```

## LLM Context Policy v0.4.3

Stage 1 context is deterministic: current raw message, up to the last 12 messages from the current active track, previous track analysis, pinned exact messages, and a memory manifest.

Stage 2 uses the same base context, then adds full content only for selected validated `MemoryItems`.

Stage 1 returns a structured decision: answer directly or request selected memory ids.

Stage 2 returns a structured final decision with `final_answer` and `used_memory_ids`.

There is no free-form summary in the context. Closed tracks do not leak their dialogue tail into new active tracks. All `dialogue_turns` remain stored in SQLite.

## Real LLM Provider Adapter

The isolated provider adapter uses OpenAI-compatible chat completions and is configured only by `MNEMOSYNE_LLM_BASE_URL`, `MNEMOSYNE_LLM_API_KEY`, and `MNEMOSYNE_LLM_MODEL`.

Tests use fake HTTP transports only, and secrets must never be committed to git.

## DB Path

The demo reads `MNEMOSYNE_DB_PATH`. If it is not set, it uses `./mnemosyne_brain.sqlite3`.

```bash
MNEMOSYNE_DB_PATH=/tmp/mnemosyne_brain.sqlite3 python3 -m mnemosyne_brain.app.run_demo
```

## Architecture Rules

LangGraph state is runtime/checkpoint state only. Durable business truth lives in SQLite.

Executor callbacks are persisted in `executor_events` before `graph.invoke`. The callback graph invocation passes only `input_source`, `event_id`, `capsule_id`, and `thread_id`.

Memory writes pass through dedupe, conflict decision, and atomic apply. `save_immediately` does not bypass `ConflictDecision`.

## Limitations

This MVP has no FastAPI/Flask server, no external queues, no cloud storage, no browser executor, no shell executor, and no Graphiti integration. Hermes is a local adapter that creates durable task capsules; no network call is made.
