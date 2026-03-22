from pydantic import BaseModel
from typing import Literal

class SimulationAgent(BaseModel):
    node_id: str
    label: str
    entity_type: str
    description: str
    is_anchor: bool
    influence: Literal['low', 'medium', 'high']   # derived from centrality
    centrality: int
    relationships: list[str]                       # edge summaries for context
    personality_prompt: str                        # LLM-generated, used by OASIS
