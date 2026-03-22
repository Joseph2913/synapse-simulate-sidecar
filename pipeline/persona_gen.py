import os
from openai import AsyncOpenAI
from models.agent import SimulationAgent
from models.seed_graph import SimulationNode
from pipeline.graph_import import InternalGraph

MAX_AGENTS = int(os.getenv("MAX_AGENTS", 30))

AGENT_ENTITY_TYPES = {"Person", "Organization", "Team"}

_client = None

def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
        )
    return _client

def compute_influence(node: SimulationNode) -> str:
    """Derive influence tier from centrality + anchor status."""
    if node.is_anchor:
        return "high"
    if node.centrality >= 5:
        return "high"
    if node.centrality >= 2:
        return "medium"
    return "low"

def summarise_relationships(node_id: str, graph: InternalGraph) -> list[str]:
    """Build human-readable relationship summaries for a node's edges."""
    summaries = []
    for edge in graph.edge_index.get(node_id, []):
        other_id = (
            edge.target_node_id
            if edge.source_node_id == node_id
            else edge.source_node_id
        )
        other_node = graph.nodes.get(other_id)
        if other_node:
            direction = "→" if edge.source_node_id == node_id else "←"
            summaries.append(
                f"{node_id} {direction} [{edge.relation_type}] {other_node.label}"
                + (f": {edge.evidence[:100]}" if edge.evidence else "")
            )
    return summaries[:10]  # cap to avoid oversized prompts

async def generate_personality_prompt(
    node: SimulationNode,
    relationships: list[str],
    prediction_question: str,
) -> str:
    """
    Ask the LLM to write a behavioural persona for this agent.
    This prompt is passed to OASIS and governs how the agent behaves in simulation.
    """
    relationship_text = "\n".join(relationships) if relationships else "No direct relationships in scope."

    system = (
        "You write concise behavioural personas for AI simulation agents. "
        "Each persona describes how the entity thinks, what motivates it, "
        "and how it typically acts. Be specific and grounded. 3–5 sentences max."
    )

    user = f"""Entity: {node.label}
Type: {node.entity_type}
Description: {node.description}
Known relationships:
{relationship_text}

Prediction context: {prediction_question}

Write a behavioural persona for this entity as a simulation agent."""

    response = await _get_client().chat.completions.create(
        model=os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=200,
        temperature=0.7,
    )
    return response.choices[0].message.content or f"{node.label} is a key actor in this domain."

async def generate_personas(
    graph: InternalGraph,
    prediction_question: str,
) -> list[SimulationAgent]:
    """
    Build agent list from eligible nodes, sorted by influence descending.
    Caps at MAX_AGENTS to avoid runaway simulation cost.
    """
    eligible_nodes = [
        node for node in graph.nodes.values()
        if node.entity_type in AGENT_ENTITY_TYPES
    ]

    # Sort: anchors first, then by centrality
    eligible_nodes.sort(
        key=lambda n: (not n.is_anchor, -n.centrality)
    )

    # Cap
    eligible_nodes = eligible_nodes[:MAX_AGENTS]

    agents: list[SimulationAgent] = []

    for node in eligible_nodes:
        relationships = summarise_relationships(node.id, graph)
        personality = await generate_personality_prompt(node, relationships, prediction_question)

        agents.append(SimulationAgent(
            node_id=node.id,
            label=node.label,
            entity_type=node.entity_type,
            description=node.description,
            is_anchor=node.is_anchor,
            influence=compute_influence(node),
            centrality=node.centrality,
            relationships=relationships,
            personality_prompt=personality,
        ))

    return agents
