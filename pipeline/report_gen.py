import os
import json
from datetime import datetime, timezone
from openai import AsyncOpenAI
from models.simulation_report import SimulationReport, SimulationForecast, SimulationAgentMove
from pipeline.simulation_runner import SimulationResult
from pipeline.environment_setup import SimulationEnvironment

client = AsyncOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)

REPORT_SYSTEM_PROMPT = """You are a prediction analyst synthesising the results of a multi-agent social simulation.
You will receive a log of agent interactions, final agent states, and the original prediction question.
Produce a structured JSON prediction report. Be specific, grounded, and honest about confidence levels.
Respond ONLY with valid JSON matching the exact schema provided. No preamble, no markdown fences."""

async def generate_report(
    result: SimulationResult,
    env: SimulationEnvironment,
) -> SimulationReport:
    """
    Synthesise a structured prediction report from the simulation output.
    Uses a single LLM call with a structured JSON output schema.
    """
    # Prefer structured log if available; fall back to raw interaction log
    if result.structured_log:
        interaction_summary = _summarise_structured_log(result.structured_log, limit=50)
    else:
        interaction_summary = _summarise_interactions(result.interaction_log, limit=50)

    agent_state_summary = "\n".join([
        f"- {state.get('label', state.get('name', 'Unknown'))}: "
        f"{state.get('final_position', 'no data')[:200]}"
        for state in result.agent_states[:20]
    ])

    what_if_text = (
        "\n".join(f"- {v}" for v in env.what_if_variables)
        if env.what_if_variables else "None."
    )

    emergent_text = "\n".join(f"- {s}" for s in result.emergent_signals) or "None detected."

    fallback_note = ""
    if result.used_fallback:
        fallback_note = "\n\nNOTE: This simulation used a fallback debate engine (OASIS was unavailable). Results may be less structured than a full OASIS simulation."

    schema = {
        "headline": "string — one sentence core prediction",
        "summary": "string — 2–3 sentence overview",
        "forecasts": [
            {
                "direction": "string",
                "rationale": "string",
                "timeframe": "string (e.g. 'Next 3 months', '6–12 months')",
                "confidence": "low | medium | high"
            }
        ],
        "agent_moves": [
            {
                "agent_label": "string",
                "entity_type": "string",
                "likely_action": "string",
                "rationale": "string",
                "influence": "low | medium | high"
            }
        ],
        "surprises": ["string — unexpected emergent behaviours"],
        "confidence_level": "low | medium | high",
        "confidence_rationale": "string — why this confidence level"
    }

    user_prompt = f"""PREDICTION QUESTION: {env.prediction_question}

WHAT-IF CONDITIONS:
{what_if_text}

AGENT INTERACTION SUMMARY:
{interaction_summary}

FINAL AGENT STATES:
{agent_state_summary}

EMERGENT SIGNALS DETECTED:
{emergent_text}{fallback_note}

Produce a prediction report in this exact JSON schema:
{json.dumps(schema, indent=2)}"""

    response = await client.chat.completions.create(
        model=os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash"),
        messages=[
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=2000,
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content or "{}"

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "headline": "Simulation completed but report generation encountered an error.",
            "summary": "The simulation ran successfully but the report could not be parsed. Check sidecar logs.",
            "forecasts": [],
            "agent_moves": [],
            "surprises": [],
            "confidence_level": "low",
            "confidence_rationale": "Report parsing failed.",
        }

    return SimulationReport(
        headline=data.get("headline", ""),
        summary=data.get("summary", ""),
        forecasts=[SimulationForecast(**f) for f in data.get("forecasts", [])],
        agent_moves=[SimulationAgentMove(**m) for m in data.get("agent_moves", [])],
        surprises=data.get("surprises", []),
        confidence_level=data.get("confidence_level", "low"),
        confidence_rationale=data.get("confidence_rationale", ""),
        simulation_rounds=result.rounds_completed,
        agent_count=len(result.agent_states),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def _summarise_structured_log(log: list[dict], limit: int) -> str:
    """Format structured InteractionLogEntry records for the report prompt."""
    recent = log[-limit:] if len(log) > limit else log
    lines = []
    for entry in recent:
        agent = entry.get("agent_label", entry.get("agent_id", "Unknown"))
        rtype = entry.get("round_type", "")
        rnum = entry.get("round_number", "?")
        summary = entry.get("position_summary", "")
        delta = entry.get("position_delta", "unchanged")
        confidence = entry.get("confidence", 0.5)

        line = f"[R{rnum}/{rtype}] {agent} (Δ={delta}, conf={confidence:.1f}): {summary}"
        lines.append(line)
    return "\n".join(lines) if lines else "No interactions recorded."


def _summarise_interactions(log: list[dict], limit: int) -> str:
    """Format raw interaction log entries for the report prompt."""
    recent = log[-limit:] if len(log) > limit else log
    lines = []
    for entry in recent:
        agent = entry.get("agent_name", entry.get("agent_id", "Unknown"))
        action = entry.get("action", "posted")
        content = str(entry.get("content", entry.get("response_text", "")))[:200]
        lines.append(f"[{agent}] {action}: {content}")
    return "\n".join(lines) if lines else "No interactions recorded."
