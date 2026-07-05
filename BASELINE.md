# Mnemosyne Brain Baseline

## Version

v0.4.2

## Accepted Date

2026-07-05

## Verification

### Test Command

python3 -m unittest discover -s tests -v

### Test Result

Ran 19 tests
OK

### Demo Command

MNEMOSYNE_DB_PATH=/tmp/mnemosyne_brain_external_review.sqlite3 python3 -m mnemosyne_brain.app.run_demo

### Demo Result

Local answer: Executor result received: {"answer":"Context collected locally"}
DB path: /tmp/mnemosyne_brain_external_review.sqlite3
Track: trk_b790fb18965a42e4abbe98551e773b13
Capsule: cap_9eae86c6411745cd84bf3b588a1af939

## Known Limitations

- No FastAPI or Flask service.
- No external queues.
- No cloud storage.
- No Graphiti integration.
- No browser executor.
- No shell executor.
- Hermes is local-only and does not perform real network dispatch.

## Forbidden Changes Without Approval

- Do not use or import Jeeves MVP code.
- Do not modify /Users/cinema/projects/jeeves-mvp.
- Do not add Graphiti.
- Do not add browser executor or shell executor.
- Do not add FastAPI, Flask, pytest, external queues, or cloud storage.
- Do not change approved table names or add unapproved replacement tables.
- Do not store business truth in LangGraph checkpoint state.
- Do not process ExecutorEvent as a user turn.
- Do not call graph.invoke inside a DB transaction.
- Do not bypass ConflictDecision, provenance, identity validation, or idempotency checks.
