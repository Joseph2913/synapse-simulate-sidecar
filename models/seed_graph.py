from pydantic import BaseModel, Field
from typing import Optional, Literal, Union

class SimulationNode(BaseModel):
    id: str
    label: str
    entity_type: str
    description: str
    is_anchor: bool
    confidence: float
    centrality: int
    source_id: Optional[str] = None
    tags: list[str] = []

class SimulationEdge(BaseModel):
    id: str
    source_node_id: str
    target_node_id: str
    relation_type: str
    evidence: str
    weight: float

class SimulationChunk(BaseModel):
    id: str
    source_id: str
    content: str
    chunk_index: int

class SeedGraphMetadata(BaseModel):
    exported_at: str
    anchor_ids: list[str]
    time_window_days: int

class SeedGraph(BaseModel):
    nodes: list[SimulationNode]
    edges: list[SimulationEdge]
    source_chunks: list[SimulationChunk] = []
    metadata: SeedGraphMetadata


# ── SimulationConfig (from PRD-Simulate-C) ────────────────────────────

class SimulationConfig(BaseModel):
    """Simulation configuration — depth tier, mode, and question."""
    question: str
    mode: str = 'prediction'   # prediction | hypothesis_test | contrarian_scan | optimisation | consensus_mapping
    depth: str = 'standard'    # quick_scan | standard | deep_dive | exhaustive
    what_if_variables: list[str] = []


# ── SimulationPersona stub (from PRD-Simulate-D) ─────────────────────
# PRD-Simulate-D may not be merged yet. This is a permissive stub that
# accepts whatever fields arrive. The simulation_runner and oasis_adapter
# access fields by key with safe .get() defaults.

class SimulationPersona(BaseModel):
    """
    Permissive persona model — accepts all fields from PRD-Simulate-D.
    Falls back gracefully if fields are missing.
    """
    agent_id: str = ''
    label: str = ''
    entity_type: str = ''
    description: str = ''
    behavioural_prompt: str = ''
    influence_tier: str = 'medium'
    epistemic_style: str = 'cautious'
    question_specific_stance: str = ''
    stance_category: str = 'neutral'
    update_conditions: Union[str, list[str]] = ''
    grounding_chunk_ids: list[str] = []
    inter_agent_relationships: Union[list[str], list[dict]] = []
    # Legacy fields (for backward compat with existing sidecar contract)
    node_id: str = ''
    is_anchor: bool = False
    influence: str = 'medium'
    centrality: int = 0
    relationships: list[str] = []
    personality_prompt: str = ''

    class Config:
        extra = 'allow'  # Accept unknown fields without failing


# ── Request model ─────────────────────────────────────────────────────

class SimulateRequest(BaseModel):
    job_id: str
    seed_graph: SeedGraph
    config: SimulationConfig = Field(default_factory=lambda: SimulationConfig(question=''))
    personas: list[SimulationPersona] = []
    # Legacy fields (backward compat — ignored if config/personas present)
    prediction_question: str = ''
    what_if_variables: list[str] = []
