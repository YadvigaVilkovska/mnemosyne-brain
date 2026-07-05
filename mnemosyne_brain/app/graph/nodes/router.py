"""Routing helpers for the graph."""

from __future__ import annotations

from ...contracts.routing import GraphRoute, InputSource
from ..state import BrainGraphState


def route_ingress(state: BrainGraphState) -> str:
    """Route graph ingress based on input_source."""

    if state["input_source"] == InputSource.EXECUTOR.value:
        return "load_persisted_executor_event"
    return "persist_dialogue_turn"


def route_terminal(state: BrainGraphState) -> str:
    """Route to a terminal action node."""

    return state.get("route", GraphRoute.LOCAL_ANSWER.value)
