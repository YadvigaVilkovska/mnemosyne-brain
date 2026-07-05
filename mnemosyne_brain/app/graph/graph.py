"""LangGraph assembly for Mnemosyne Brain v0.4.2."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from ..db.repository import SqliteRepository
from ..executors.hermes import HermesExecutor
from ..memory.conflicts import ConflictResolver
from ..memory.dedupe import MemoryDeduper
from ..memory.staging import MemoryStagingService
from ..memory.write import MemoryWriter
from .nodes.executor import (
    AskClarificationNode,
    CallExecutorNode,
    CloseTrackNode,
    ErrorHandlerNode,
    LocalAnswerNode,
    RouterNode,
)
from .nodes.executor_ingress import (
    ExecutorFeedbackAnalyzerNode,
    ExecutorFeedbackHandlerNode,
    LoadPersistedExecutorEventNode,
    LoadTrackByCapsuleNode,
    ValidateExecutorEventNode,
)
from .nodes.memory import ApplyMemoryCandidatesNode
from .nodes.router import route_ingress, route_terminal
from .nodes.user_ingress import (
    BootstrapOrLoadTrackNode,
    MemoryRetrievalNode,
    PersistDialogueTurnNode,
    PersistMemoryCandidatesNode,
    TrackUpdaterNode,
    TurnAnalyzerNode,
)
from .state import BrainGraphState


def build_graph(repository: SqliteRepository):
    """Build the approved graph shape with reference-only state."""

    deduper = MemoryDeduper(repository)
    conflict_resolver = ConflictResolver(deduper)
    staging_service = MemoryStagingService(repository)
    writer = MemoryWriter(repository, conflict_resolver, staging_service)
    hermes = HermesExecutor(repository)

    graph = StateGraph(BrainGraphState)
    graph.add_node("ingress_router", lambda state: {})
    graph.add_node("persist_dialogue_turn", PersistDialogueTurnNode(repository))
    graph.add_node("bootstrap_or_load_track", BootstrapOrLoadTrackNode(repository))
    graph.add_node("memory_retrieval", MemoryRetrievalNode())
    graph.add_node("turn_analyzer", TurnAnalyzerNode())
    graph.add_node("persist_memory_candidates", PersistMemoryCandidatesNode(repository))
    graph.add_node("apply_memory_candidates", ApplyMemoryCandidatesNode(repository, writer))
    graph.add_node("track_updater", TrackUpdaterNode(repository))
    graph.add_node("load_persisted_executor_event", LoadPersistedExecutorEventNode(repository))
    graph.add_node("load_track_by_capsule", LoadTrackByCapsuleNode(repository))
    graph.add_node("validate_executor_event", ValidateExecutorEventNode(repository))
    graph.add_node("executor_feedback_analyzer", ExecutorFeedbackAnalyzerNode(repository))
    graph.add_node("executor_feedback_handler", ExecutorFeedbackHandlerNode(repository))
    graph.add_node("router", RouterNode())
    graph.add_node("local_answer", LocalAnswerNode())
    graph.add_node("call_executor", CallExecutorNode(repository, hermes))
    graph.add_node("ask_clarification", AskClarificationNode())
    graph.add_node("close_track", CloseTrackNode(repository))
    graph.add_node("error_handler", ErrorHandlerNode())

    graph.add_edge(START, "ingress_router")
    graph.add_conditional_edges(
        "ingress_router",
        route_ingress,
        {
            "persist_dialogue_turn": "persist_dialogue_turn",
            "load_persisted_executor_event": "load_persisted_executor_event",
        },
    )
    graph.add_edge("persist_dialogue_turn", "bootstrap_or_load_track")
    graph.add_edge("bootstrap_or_load_track", "memory_retrieval")
    graph.add_edge("memory_retrieval", "turn_analyzer")
    graph.add_edge("turn_analyzer", "persist_memory_candidates")
    graph.add_edge("persist_memory_candidates", "apply_memory_candidates")
    graph.add_edge("apply_memory_candidates", "track_updater")
    graph.add_edge("track_updater", "router")

    graph.add_edge("load_persisted_executor_event", "load_track_by_capsule")
    graph.add_edge("load_track_by_capsule", "validate_executor_event")
    graph.add_edge("validate_executor_event", "executor_feedback_analyzer")
    graph.add_edge("executor_feedback_analyzer", "executor_feedback_handler")
    graph.add_edge("executor_feedback_handler", "router")

    graph.add_conditional_edges(
        "router",
        route_terminal,
        {
            "local_answer": "local_answer",
            "call_executor": "call_executor",
            "ask_clarification": "ask_clarification",
            "close_track": "close_track",
            "error_handler": "error_handler",
        },
    )
    graph.add_edge("local_answer", END)
    graph.add_edge("call_executor", END)
    graph.add_edge("ask_clarification", END)
    graph.add_edge("close_track", END)
    graph.add_edge("error_handler", END)
    return graph.compile()
