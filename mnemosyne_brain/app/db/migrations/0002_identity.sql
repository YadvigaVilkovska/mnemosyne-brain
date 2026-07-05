BEGIN;

CREATE TABLE IF NOT EXISTS persons (
  person_id TEXT PRIMARY KEY,
  schema_version TEXT NOT NULL DEFAULT '0.4.2',
  display_name TEXT NOT NULL,
  status TEXT NOT NULL,
  merged_into TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS personas (
  persona_id TEXT PRIMARY KEY,
  schema_version TEXT NOT NULL DEFAULT '0.4.2',
  person_id TEXT NOT NULL,
  persona_type TEXT NOT NULL,
  display_name TEXT NOT NULL,
  persona_context_json TEXT NOT NULL,
  status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(person_id) REFERENCES persons(person_id)
);

CREATE TABLE IF NOT EXISTS identifiers (
  identifier_id TEXT PRIMARY KEY,
  schema_version TEXT NOT NULL DEFAULT '0.4.2',
  identifier_type TEXT NOT NULL,
  raw_value TEXT NOT NULL,
  normalized_value TEXT NOT NULL,
  identifier_key TEXT NOT NULL UNIQUE,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS identifier_assignments (
  assignment_id TEXT PRIMARY KEY,
  schema_version TEXT NOT NULL DEFAULT '0.4.2',
  identifier_key TEXT NOT NULL,
  person_id TEXT,
  persona_id TEXT,
  resolution_status TEXT NOT NULL,
  candidate_person_ids TEXT NOT NULL DEFAULT '[]',
  assignment_scope TEXT NOT NULL DEFAULT 'individual',
  status TEXT NOT NULL,
  valid_from TEXT,
  valid_to TEXT,
  confidence REAL NOT NULL,
  provenance_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(identifier_key) REFERENCES identifiers(identifier_key),
  FOREIGN KEY(person_id) REFERENCES persons(person_id),
  FOREIGN KEY(persona_id) REFERENCES personas(persona_id)
);
CREATE INDEX IF NOT EXISTS idx_identifier_reverse_lookup
ON identifier_assignments(identifier_key, status);
CREATE INDEX IF NOT EXISTS idx_identifier_current_lookup
ON identifier_assignments(person_id, persona_id, status, valid_to);

CREATE TABLE IF NOT EXISTS name_aliases (
  alias_id TEXT PRIMARY KEY,
  schema_version TEXT NOT NULL DEFAULT '0.4.2',
  raw_name TEXT NOT NULL,
  normalized_name TEXT NOT NULL,
  person_id TEXT,
  persona_id TEXT,
  locale TEXT,
  status TEXT NOT NULL,
  confidence REAL NOT NULL,
  provenance_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(person_id) REFERENCES persons(person_id),
  FOREIGN KEY(persona_id) REFERENCES personas(persona_id)
);

COMMIT;
