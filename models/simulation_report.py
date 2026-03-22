from pydantic import BaseModel
from typing import Literal

class SimulationForecast(BaseModel):
    direction: str
    rationale: str
    timeframe: str
    confidence: Literal['low', 'medium', 'high']

class SimulationAgentMove(BaseModel):
    agent_label: str
    entity_type: str
    likely_action: str
    rationale: str
    influence: Literal['low', 'medium', 'high']

class SimulationReport(BaseModel):
    headline: str
    summary: str
    forecasts: list[SimulationForecast]
    agent_moves: list[SimulationAgentMove]
    surprises: list[str]
    confidence_level: Literal['low', 'medium', 'high']
    confidence_rationale: str
    simulation_rounds: int
    agent_count: int
    generated_at: str
