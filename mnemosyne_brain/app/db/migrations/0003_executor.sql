BEGIN;

CREATE TABLE IF NOT EXISTS executor_tasks (
  capsule_id TEXT PRIMARY KEY,
  schema_version TEXT NOT NULL DEFAULT '0.4.2',
  source_track_id TEXT NOT NULL,
  thread_id TEXT NOT NULL,
  executor TEXT NOT NULL,
  status TEXT NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE,
  attempt_count INTEGER NOT NULL DEFAULT 0,
  locked_by TEXT,
  locked_until TEXT,
  capsule_json TEXT NOT NULL,
  result_json TEXT,
  last_error_json TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_executor_tasks_status
ON executor_tasks(status, updated_at);
CREATE INDEX IF NOT EXISTS idx_executor_tasks_track
ON executor_tasks(source_track_id);

CREATE TABLE IF NOT EXISTS executor_events (
  event_id TEXT PRIMARY KEY,
  schema_version TEXT NOT NULL DEFAULT '0.4.2',
  capsule_id TEXT NOT NULL,
  correlation_id TEXT NOT NULL,
  executor TEXT NOT NULL,
  status TEXT NOT NULL,
  attempt INTEGER NOT NULL,
  is_final INTEGER NOT NULL,
  applied INTEGER NOT NULL DEFAULT 0,
  stale INTEGER NOT NULL DEFAULT 0,
  applied_at TEXT,
  payload_json TEXT NOT NULL,
  error_json TEXT,
  artifacts_json TEXT,
  created_at TEXT NOT NULL,
  received_at TEXT NOT NULL,
  FOREIGN KEY(capsule_id) REFERENCES executor_tasks(capsule_id)
);
CREATE INDEX IF NOT EXISTS idx_executor_events_capsule
ON executor_events(capsule_id, attempt, created_at);
CREATE INDEX IF NOT EXISTS idx_executor_events_correlation
ON executor_events(correlation_id, capsule_id, attempt);

COMMIT;
