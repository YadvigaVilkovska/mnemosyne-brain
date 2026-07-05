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
