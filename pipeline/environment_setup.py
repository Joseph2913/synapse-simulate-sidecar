from __future__ import annotations

import os
from models.seed_graph import SeedGraph

MAX_ROUNDS = int(os.getenv("MAX_SIMULATION_ROUNDS", 10))


class SimulationEnvironment:
    def __init__(self):
        self.prediction_question: str = ""
        self.what_if_variables: list[str] = []
        self.world_context: str = ""
        self.max_rounds: int = MAX_ROUNDS
        self.source_context: str = ""


def build_environment(
    seed_graph: SeedGraph,
    prediction_question: str,
    what_if_variables: list[str] | None = None,
) -> SimulationEnvironment:
    """
    Assemble the simulation environment.

    World context is constructed from:
    1. Source chunks (the richest signal — raw knowledge)
    2. Key entity descriptions (structured signal)
    3. Prediction question (directional frame)
    4. What-if variables (injected conditions)
    """
    if what_if_variables is None:
        what_if_variables = []

    env = SimulationEnvironment()
    env.prediction_question = prediction_question
    env.what_if_variables = what_if_variables
    env.max_rounds = MAX_ROUNDS

    # Build source context from chunks (most valuable signal)
    chunks_by_source: dict[str, list[str]] = {}
    for chunk in seed_graph.source_chunks:
        chunks_by_source.setdefault(chunk.source_id, []).append(chunk.content)

    sampled_chunks: list[str] = []
    for source_id, chunks in chunks_by_source.items():
        sampled_chunks.extend(chunks[:3])

    source_text = "\n\n---\n\n".join(sampled_chunks)
    if len(source_text) > 32000:
        source_text = source_text[:32000] + "\n\n[Context truncated]"

    env.source_context = source_text

    what_if_text = (
        "\n".join(f"- {v}" for v in what_if_variables)
        if what_if_variables
        else "None specified."
    )

    anchor_nodes = [n for n in seed_graph.nodes if n.is_anchor]
    anchor_summary = ", ".join(n.label for n in anchor_nodes[:10])

    env.world_context = f"""SIMULATION CONTEXT
==================
Prediction question: {prediction_question}

Assumed conditions (what-if variables):
{what_if_text}

Key entities in scope: {anchor_summary}

Background knowledge:
{source_text}
"""

    return env
