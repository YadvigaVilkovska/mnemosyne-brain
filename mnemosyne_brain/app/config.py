"""Runtime configuration for the local MVP."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_DB_FILENAME = "mnemosyne_brain.sqlite3"
DB_PATH_ENV = "MNEMOSYNE_DB_PATH"


@dataclass(frozen=True)
class AppConfig:
    """Configuration values injected into application entrypoints."""

    db_path: Path


def load_config() -> AppConfig:
    """Load configuration from environment with local deterministic defaults."""

    db_path = Path(os.environ.get(DB_PATH_ENV, DEFAULT_DB_FILENAME))
    return AppConfig(db_path=db_path)
