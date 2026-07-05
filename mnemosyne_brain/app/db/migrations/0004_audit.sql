BEGIN;

CREATE TABLE IF NOT EXISTS audit_events (
  audit_event_id TEXT PRIMARY KEY,
  schema_version TEXT NOT NULL DEFAULT '0.4.2',
  event_type TEXT NOT NULL,
  actor_type TEXT NOT NULL,
  actor_id TEXT,
  dialogue_id TEXT,
  track_id TEXT,
  turn_id TEXT,
  target_type TEXT NOT NULL,
  target_id TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_audit_target
ON audit_events(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_audit_track
ON audit_events(track_id, created_at);

COMMIT;
