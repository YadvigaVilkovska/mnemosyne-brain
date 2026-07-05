"""Routing constants and graph route contracts."""

from __future__ import annotations

from enum import StrEnum


class InputSource(StrEnum):
    """Supported ingress sources for the graph."""

    USER = "user"
    SYSTEM = "system"
    EXECUTOR = "executor"


class GraphRoute(StrEnum):
    """Terminal graph routes."""

    LOCAL_ANSWER = "local_answer"
    CALL_EXECUTOR = "call_executor"
    ASK_CLARIFICATION = "ask_clarification"
    CLOSE_TRACK = "close_track"
    ERROR_HANDLER = "error_handler"
