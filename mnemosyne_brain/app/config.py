"""Runtime configuration for the local MVP."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DB_FILENAME = "mnemosyne_brain.sqlite3"
DB_PATH_ENV = "MNEMOSYNE_DB_PATH"
PROJECT_ENV_FILENAME = ".env"


@dataclass(frozen=True)
class AppConfig:
    """Configuration values injected into application entrypoints."""

    db_path: Path


def load_project_env(env_path: Path | None = None) -> None:
    """Load simple KEY=value pairs from the project .env without overriding process env."""

    path = env_path or Path.cwd() / PROJECT_ENV_FILENAME
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip()


def load_config() -> AppConfig:
    """Load configuration from environment with local deterministic defaults."""

    load_project_env()
    db_path = Path(os.environ.get(DB_PATH_ENV, DEFAULT_DB_FILENAME))
    return AppConfig(db_path=db_path)
