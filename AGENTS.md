# AGENTS.md

You are working on Brain Orchestrator v0.4.2.

## Non-negotiable rules

1. Do not redesign the architecture.
2. Do not add Graphiti.
3. Do not add multi-agent marketplace.
4. Do not add browser_executor or shell_executor.
5. Do not implement automatic memory approval.
6. Do not delete existing files unless explicitly instructed.
7. Do not run destructive commands.
8. Do not use git reset --hard.
9. Do not change public contracts unless required by this task.
10. Prefer small, reviewable changes.

## Implementation target

Build the MVP implementation baseline:

user → track → candidate → staging → memory → Hermes task → callback → resume → answer

## Runtime rule

LangGraph checkpoint is runtime state only.
Business truth lives in SQL tables.

## Required coding style

- Use Pydantic v2.
- Keep JSON serialization inside repository functions.
- Use explicit transactions for memory and identity writes.
- No external calls inside DB transactions.
- Callback handler must be idempotent.
- ExecutorEvent is persisted at HTTP callback boundary.
- LangGraph executor ingress loads persisted event by event_id.
- save_immediately still passes through ConflictDecision.

## Required output after work

Provide:
1. Changed files list.
2. Migration summary.
3. New/updated Pydantic models.
4. LangGraph node changes.
5. Transaction functions added.
6. Tests added.
7. Test command output.
8. Known limitations.
9. Rollback instructions.
