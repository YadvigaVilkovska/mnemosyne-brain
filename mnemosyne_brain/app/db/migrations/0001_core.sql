BEGIN;

CREATE TABLE IF NOT EXISTS dialogue_threads (
  dialogue_id TEXT NOT NULL,
  track_id TEXT NOT NULL,
  thread_id TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  PRIMARY KEY (dialogue_id, track_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS uniq_dialogue_threads_thread
ON dialogue_threads(thread_id);
CREATE INDEX IF NOT EXISTS idx_dialogue_threads_dialogue_status
ON dialogue_threads(dialogue_id, status);

CREATE TABLE IF NOT EXISTS dialogue_turns (
  turn_id TEXT PRIMARY KEY,
  schema_version TEXT NOT NULL DEFAULT '0.4.2',
  dialogue_id TEXT NOT NULL,
  track_id TEXT,
  thread_id TEXT,
  input_source TEXT NOT NULL,
  role TEXT NOT NULL,
  external_message_id TEXT,
  content_text TEXT,
  content_json TEXT,
  created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS uniq_dialogue_turn_external_message
ON dialogue_turns(external_message_id)
WHERE external_message_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_dialogue_turns_dialogue_created
ON dialogue_turns(dialogue_id, created_at);
CREATE INDEX IF NOT EXISTS idx_dialogue_turns_track_created
ON dialogue_turns(track_id, created_at);

CREATE TABLE IF NOT EXISTS dialogue_tracks_temp (
  track_id TEXT PRIMARY KEY,
  schema_version TEXT NOT NULL DEFAULT '0.4.2',
  dialogue_id TEXT NOT NULL,
  thread_id TEXT NOT NULL,
  owner_user_id TEXT NOT NULL,
  status TEXT NOT NULL,
  summary TEXT,
  track_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  last_turn_id TEXT
);
CREATE INDEX IF NOT EXISTS idx_tracks_dialogue_status
ON dialogue_tracks_temp(dialogue_id, status);
CREATE INDEX IF NOT EXISTS idx_tracks_thread
ON dialogue_tracks_temp(thread_id);

CREATE TABLE IF NOT EXISTS memory_candidates (
  candidate_id TEXT PRIMARY KEY,
  schema_version TEXT NOT NULL DEFAULT '0.4.2',
  dialogue_id TEXT NOT NULL,
  track_id TEXT NOT NULL,
  turn_id TEXT NOT NULL,
  candidate_type TEXT NOT NULL,
  recommended_action TEXT NOT NULL,
  confidence REAL NOT NULL,
  dedupe_key TEXT NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE,
  content_json TEXT NOT NULL,
  provenance_json TEXT NOT NULL,
  merge_target_memory_id TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_candidates_track
ON memory_candidates(track_id);
CREATE INDEX IF NOT EXISTS idx_candidates_turn
ON memory_candidates(turn_id);
CREATE INDEX IF NOT EXISTS idx_candidates_dedupe
ON memory_candidates(dedupe_key);

CREATE TABLE IF NOT EXISTS memory_staging (
  staging_id TEXT PRIMARY KEY,
  schema_version TEXT NOT NULL DEFAULT '0.4.2',
  candidate_id TEXT NOT NULL,
  candidate_type TEXT NOT NULL,
  status TEXT NOT NULL,
  recommended_action TEXT NOT NULL,
  confidence REAL NOT NULL,
  dedupe_key TEXT NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE,
  merge_target_memory_id TEXT,
  conflict_memory_ids TEXT NOT NULL DEFAULT '[]',
  content_json TEXT NOT NULL,
  provenance_json TEXT NOT NULL,
  review_reason TEXT NOT NULL,
  reviewed_by TEXT,
  reviewed_at TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(candidate_id) REFERENCES memory_candidates(candidate_id)
);
CREATE INDEX IF NOT EXISTS idx_staging_status
ON memory_staging(status);
CREATE INDEX IF NOT EXISTS idx_staging_dedupe
ON memory_staging(dedupe_key);

CREATE TABLE IF NOT EXISTS memory_items (
  memory_id TEXT PRIMARY KEY,
  schema_version TEXT NOT NULL DEFAULT '0.4.2',
  memory_type TEXT NOT NULL,
  status TEXT NOT NULL,
  stability TEXT NOT NULL,
  content_json TEXT NOT NULL,
  intent_tags TEXT NOT NULL,
  entity_keys TEXT NOT NULL,
  provenance_json TEXT NOT NULL,
  dedupe_key TEXT NOT NULL,
  source_track_id TEXT,
  source_turn_id TEXT,
  valid_from TEXT,
  valid_to TEXT,
  observed_at TEXT,
  confidence REAL NOT NULL,
  privacy_level TEXT NOT NULL DEFAULT 'normal',
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_type_status
ON memory_items(memory_type, status);
CREATE INDEX IF NOT EXISTS idx_memory_updated_at
ON memory_items(updated_at);
CREATE INDEX IF NOT EXISTS idx_memory_dedupe_key
ON memory_items(dedupe_key);
CREATE UNIQUE INDEX IF NOT EXISTS uniq_active_memory_dedupe
ON memory_items(dedupe_key)
WHERE status IN ('active', 'needs_confirmation');

COMMIT;
