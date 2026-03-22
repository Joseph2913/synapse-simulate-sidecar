"""
OASIS Adapter — translates between Synapse persona format and OASIS profile format.

Three responsibilities:
  1. SimulationPersona → OASIS agent profile
  2. Inter-agent relationships → OASIS follower/influence graph
  3. OASIS raw interaction log → Synapse structured InteractionLogEntry format
"""

import os
import re
import asyncio
from openai import AsyncOpenAI

# Concurrency limit for post-simulation LLM calls
SUMMARY_CONCURRENCY = int(os.getenv("SUMMARY_CONCURRENCY", 5))

_client = None

def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
        )
    return _client

def _get_model() -> str:
    return os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash")


# ── Persona → OASIS profile ──────────────────────────────────────────

def persona_to_oasis_profile(persona: dict) -> dict:
    """
    Translate a SimulationPersona dict into an OASIS-compatible agent profile.

    Expects persona keys from PRD-Simulate-D:
      agent_id, label, entity_type, behavioural_prompt, influence_tier,
      epistemic_style, question_specific_stance, update_conditions,
      grounding_chunk_ids, inter_agent_relationships
    """
    activity_map = {'high': 0.9, 'medium': 0.6, 'low': 0.3}

    susceptibility_map = {
        'empirical': 0.7,
        'ideological': 0.2,
        'opportunistic': 0.8,
        'contrarian': 0.4,
        'cautious': 0.6,
        'structural': 0.5,
    }

    influence_tier = persona.get('influence_tier', 'medium')

    return {
        'id': persona.get('agent_id', ''),
        'name': persona.get('label', ''),
        'type': persona.get('entity_type', ''),
        'system_prompt': persona.get('behavioural_prompt', ''),
        'activity_level': activity_map.get(influence_tier, 0.6),
        'opinion_susceptibility': susceptibility_map.get(
            persona.get('epistemic_style', 'cautious'), 0.6
        ),
        'influence_weight': (
            1.0 if influence_tier == 'high'
            else 0.6 if influence_tier == 'medium'
            else 0.3
        ),
        'initial_stance': persona.get('question_specific_stance', ''),
        'update_conditions': persona.get('update_conditions', []),
        'grounding_chunks': persona.get('grounding_chunk_ids', []),
    }


# ── Inter-agent influence graph ───────────────────────────────────────

def build_influence_graph(personas: list[dict]) -> dict:
    """
    Build the OASIS follower graph from inter_agent_relationships.

    Returns a dict of { (source_id, target_id): weight } representing
    directional influence exposure.

    Rules:
      - Positive relationship (supports, collaborates, enables) → mutual 0.8
      - Negative relationship (blocks, contradicts, competes) → mutual 0.9
      - No relationship → default 0.3
    """
    POSITIVE_RELATIONS = {'supports', 'collaborates', 'enables', 'allies', 'funds', 'advises'}
    NEGATIVE_RELATIONS = {'blocks', 'contradicts', 'competes', 'opposes', 'undermines', 'challenges'}

    graph: dict[tuple[str, str], float] = {}
    agent_ids = [p.get('agent_id', '') for p in personas]

    # Set defaults
    for i, a_id in enumerate(agent_ids):
        for j, b_id in enumerate(agent_ids):
            if i != j:
                graph[(a_id, b_id)] = 0.3

    # Override with explicit relationships
    for persona in personas:
        agent_id = persona.get('agent_id', '')
        relationships = persona.get('inter_agent_relationships', [])

        for rel in relationships:
            target_id = rel.get('target_agent_id', '')
            relation_type = rel.get('relation_type', '').lower()

            if target_id not in agent_ids:
                continue

            if relation_type in POSITIVE_RELATIONS:
                weight = 0.8
            elif relation_type in NEGATIVE_RELATIONS:
                weight = 0.9  # high exposure to adversaries drives challenge
            else:
                weight = 0.5  # unknown relation type — moderate exposure

            # Mutual
            graph[(agent_id, target_id)] = weight
            graph[(target_id, agent_id)] = weight

    return graph


# ── OASIS raw log → Synapse structured log ────────────────────────────

async def transform_interaction_log(
    oasis_log: list[dict],
    personas: list[dict],
) -> list[dict]:
    """
    Transform the raw interaction log (collected during simulation rounds)
    into the structured InteractionLogEntry format for the report generator.

    Post-simulation enrichment:
      - position_summary: one-sentence summary per entry (batched LLM calls)
      - position_delta: unchanged / updated / reversed (heuristic comparison)
    """
    if not oasis_log:
        return []

    # Build persona label lookup
    label_map = {p.get('agent_id', ''): p.get('label', '') for p in personas}

    # Batch extract position summaries
    summaries = await _batch_extract_summaries(oasis_log)

    structured = []
    for i, entry in enumerate(oasis_log):
        agent_id = entry.get('agent_id', '')
        round_number = entry.get('round_number', 0)

        position_summary = summaries[i] if i < len(summaries) else ''
        position_delta = _compute_position_delta(
            agent_id, round_number, i, oasis_log, summaries
        )

        structured.append({
            'agent_id': agent_id,
            'agent_label': label_map.get(agent_id, agent_id),
            'round_number': round_number,
            'round_type': entry.get('round_type', 'opening'),
            'prompt_issued': entry.get('prompt_issued', ''),
            'response_text': entry.get('response_text', ''),
            'position_summary': position_summary,
            'position_delta': position_delta,
            'confidence': entry.get('confidence', 0.5),
            'agents_addressed': entry.get('agents_addressed', []),
            'grounding_chunks': entry.get('grounding_chunks', []),
            'stance_category': entry.get('stance_category', 'unknown'),
        })

    return structured


async def _batch_extract_summaries(oasis_log: list[dict]) -> list[str]:
    """
    Extract one-sentence position summaries for all log entries.
    Batched with concurrency control to avoid rate limits.
    """
    sem = asyncio.Semaphore(SUMMARY_CONCURRENCY)

    async def _summarise_one(entry: dict) -> str:
        response_text = entry.get('response_text', '')
        if not response_text or response_text.startswith('[rate limit'):
            return response_text[:100] if response_text else ''

        async with sem:
            try:
                response = await _get_client().chat.completions.create(
                    model=_get_model(),
                    messages=[
                        {
                            "role": "system",
                            "content": "Summarise this agent's current position in one sentence.",
                        },
                        {"role": "user", "content": response_text},
                    ],
                    max_tokens=80,
                    temperature=0.1,
                )
                return response.choices[0].message.content or ''
            except Exception:
                # Fallback: first sentence of the response
                sentences = re.split(r'[.!?]', response_text)
                return (sentences[0].strip() + '.') if sentences else ''

    tasks = [_summarise_one(entry) for entry in oasis_log]
    return await asyncio.gather(*tasks)


def _compute_position_delta(
    agent_id: str,
    round_number: int,
    current_index: int,
    oasis_log: list[dict],
    summaries: list[str],
) -> str:
    """
    Compare this entry's position summary to the same agent's previous round.
    Uses simple word-overlap heuristic (avoids embedding dependency).

    Returns: 'unchanged' | 'updated' | 'reversed'
    """
    # Find the agent's most recent prior entry
    prev_summary = ''
    for j in range(current_index - 1, -1, -1):
        if oasis_log[j].get('agent_id') == agent_id:
            prev_summary = summaries[j] if j < len(summaries) else ''
            break

    if not prev_summary:
        return 'unchanged'  # first round for this agent

    current_summary = summaries[current_index] if current_index < len(summaries) else ''
    if not current_summary:
        return 'unchanged'

    # Word-overlap similarity (lightweight alternative to embedding cosine)
    prev_words = set(prev_summary.lower().split())
    curr_words = set(current_summary.lower().split())

    if not prev_words or not curr_words:
        return 'unchanged'

    intersection = prev_words & curr_words
    union = prev_words | curr_words
    similarity = len(intersection) / len(union) if union else 1.0

    if similarity > 0.75:
        return 'unchanged'
    elif similarity < 0.3:
        return 'reversed'
    else:
        return 'updated'
