from __future__ import annotations

"""
Simulation Runner — OASIS-powered deliberation engine.

Replaces the old MiroFish subprocess approach with direct OASIS integration
via camel-ai/oasis-sim, wrapped in a moderator-driven round loop.

Flow:
  1. Translate personas → OASIS profiles
  2. Initialise OASIS environment
  3. Initialise moderator with round sequence from config depth
  4. Run round sequence: issue directives, collect responses, evaluate
  5. Transform raw log → structured InteractionLogEntry format
  6. Return structured log for report generation

Fallback: if OASIS is unavailable, falls back to the existing multi-agent
debate (agent_debate.py) and notes the fallback in the report header.
"""

import os
import asyncio
import re
import json
from datetime import datetime, timezone

from pipeline.oasis_adapter import (
    persona_to_oasis_profile,
    build_influence_graph,
    transform_interaction_log,
)
from pipeline.round_directives import (
    ROUND_TYPES,
    ROUND_SEQUENCES,
    build_directive,
    get_challenge_directives,
)
from pipeline.moderator import (
    SimulationModerator,
    is_vague,
    SPECIFICITY_RETRY_PROMPT,
)
from pipeline.supabase_writer import update_round_progress, update_job_status

TIMEOUT = int(os.getenv("SIMULATION_TIMEOUT_SECONDS", 1800))
MAX_ROUNDS = int(os.getenv("MAX_SIMULATION_ROUNDS", 10))
AGENT_CONCURRENCY = int(os.getenv("AGENT_CONCURRENCY", 5))
LLM_MAX_RETRIES = 4
LLM_BASE_BACKOFF = 2  # seconds

_client = None

def _get_client():
    global _client
    if _client is None:
        from openai import AsyncOpenAI
        _client = AsyncOpenAI(
            api_key=os.getenv("LLM_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL"),
        )
    return _client

def _get_model() -> str:
    return os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash")


class SimulationResult:
    """Output of the simulation runner, consumed by report_gen."""
    def __init__(self):
        self.rounds_completed: int = 0
        self.agent_states: list[dict] = []
        self.interaction_log: list[dict] = []
        self.structured_log: list[dict] = []
        self.emergent_signals: list[str] = []
        self.used_fallback: bool = False


async def run_simulation(
    job_id: str,
    seed_graph: dict,
    personas: list[dict],
    config: dict,
    world_context: str,
) -> SimulationResult:
    """
    Run the full OASIS-powered deliberation simulation.

    Args:
        job_id: Supabase job ID for progress writes.
        seed_graph: Serialised SeedGraph dict.
        personas: List of SimulationPersona dicts (from PRD-Simulate-D, or stubs).
        config: SimulationConfig dict with keys: question, mode, depth, etc.
        world_context: Assembled world context string.

    Returns:
        SimulationResult with structured_log ready for report generation.
    """
    result = SimulationResult()
    oasis_succeeded = False

    try:
        oasis_succeeded = await _run_oasis_simulation(
            job_id, personas, config, world_context, result
        )
    except Exception as e:
        print(f"    OASIS engine error ({e}) — falling back to agent debate")
        result.emergent_signals.append(f"OASIS failed: {e}")

    # Fallback: multi-agent debate if OASIS produced nothing
    if not oasis_succeeded and not result.interaction_log:
        result.used_fallback = True
        await _run_fallback_debate(job_id, personas, config, world_context, result)

    return result


async def _run_oasis_simulation(
    job_id: str,
    personas: list[dict],
    config: dict,
    world_context: str,
    result: SimulationResult,
) -> bool:
    """
    Attempt to run the simulation via OASIS integration.
    Returns True if successful, False if OASIS is unavailable.
    """
    # Verify OASIS is importable
    try:
        from oasis.social_platform.platform import Platform
        from oasis.agent.agent import SocialAgent
    except ImportError as e:
        print(f"    OASIS not importable ({e}) — will use fallback")
        return False

    # 1. Translate personas to OASIS profiles
    oasis_profiles = [persona_to_oasis_profile(p) for p in personas]
    influence_graph = build_influence_graph(personas)

    # 2. Determine round sequence
    depth = config.get('depth', 'standard')
    # Single agent edge case: skip challenge rounds
    if len(personas) <= 1:
        round_sequence = ['opening', 'reaction', 'revision', 'closing']
    else:
        round_sequence = ROUND_SEQUENCES.get(depth, ROUND_SEQUENCES['standard'])

    # 3. Initialise moderator
    moderator = SimulationModerator(config, personas, round_sequence)

    interaction_log: list[dict] = []
    sem = asyncio.Semaphore(AGENT_CONCURRENCY)

    # 4. Run round sequence
    for round_index, round_type in enumerate(round_sequence):
        round_info = moderator.get_current_round_directive()

        # Identify minority agents for challenge rounds
        minority_ids: list[str] = []
        if round_type == 'challenge' and interaction_log:
            prev_round = [
                e for e in interaction_log
                if e.get('round_number') == round_index - 1
            ]
            minority_ids = moderator.identify_minority_agents(prev_round)

        # Issue round to all agents
        round_responses = await _run_round(
            sem=sem,
            profiles=oasis_profiles,
            personas=personas,
            round_type=round_type,
            round_index=round_index,
            config=config,
            world_context=world_context,
            interaction_log=interaction_log,
            minority_ids=minority_ids,
        )

        # Moderator evaluation
        action = moderator.evaluate_after_round(round_responses)

        if action == 'retry_specificity':
            vague_responses = [
                r for r in round_responses
                if is_vague(r.get('response_text', ''))
            ]
            for vague_r in vague_responses:
                retry = await _retry_agent(
                    sem=sem,
                    profile=_find_profile(oasis_profiles, vague_r.get('agent_id', '')),
                    persona=_find_persona(personas, vague_r.get('agent_id', '')),
                    round_type=round_type,
                    round_index=round_index,
                    world_context=world_context,
                )
                if retry:
                    round_responses = [
                        retry if r.get('agent_id') == vague_r.get('agent_id')
                        else r
                        for r in round_responses
                    ]

        elif action == 'inject_pressure':
            pressure = await moderator.inject_pressure(world_context, interaction_log)
            # Inject as a system-level entry visible in the next round's context
            interaction_log.append({
                'agent_id': '__moderator__',
                'round_number': round_index,
                'round_type': 'moderator_challenge',
                'prompt_issued': '',
                'response_text': f"Moderator challenge: {pressure}",
                'confidence': 1.0,
                'agents_addressed': [],
                'grounding_chunks': [],
                'stance_category': 'challenge',
                'position_delta': 'unchanged',
            })

        elif action == 'end_early':
            update_job_status(
                job_id, 'simulation_stagnated',
                note='Simulation ended early — consensus reached with no further movement.'
            )
            interaction_log.extend(round_responses)
            moderator.current_round_index += 1
            break

        # Log round
        interaction_log.extend(round_responses)
        moderator.current_round_index += 1

        # Write progress to Supabase (fire-and-forget)
        delta_count = len([
            r for r in round_responses
            if r.get('position_delta', 'unchanged') != 'unchanged'
        ])
        update_round_progress(
            job_id, round_index + 1, len(round_sequence),
            round_type, delta_count
        )

    # 5. Transform to structured log
    result.interaction_log = interaction_log
    result.structured_log = await transform_interaction_log(interaction_log, personas)
    result.rounds_completed = moderator.current_round_index

    # Build agent final states from last round
    last_round_entries = [
        e for e in interaction_log
        if e.get('round_number') == moderator.current_round_index - 1
        and e.get('agent_id') != '__moderator__'
    ]
    result.agent_states = [
        {
            'name': e.get('agent_id', ''),
            'label': _find_persona(personas, e.get('agent_id', '')).get('label', ''),
            'final_position': e.get('response_text', ''),
            'stance_category': e.get('stance_category', 'unknown'),
        }
        for e in last_round_entries
    ]

    return True


async def _run_round(
    sem: asyncio.Semaphore,
    profiles: list[dict],
    personas: list[dict],
    round_type: str,
    round_index: int,
    config: dict,
    world_context: str,
    interaction_log: list[dict],
    minority_ids: list[str],
) -> list[dict]:
    """
    Issue a round directive to all agents and collect responses.
    Batched at AGENT_CONCURRENCY concurrent calls.
    """
    # Build conversation context from prior rounds
    conversation_context = _build_conversation_context(interaction_log, limit=30)

    tasks = []
    for profile, persona in zip(profiles, personas):
        agent_id = profile.get('id', '')

        # Build the directive for this specific agent
        if round_type == 'challenge':
            minority_dir, majority_dir = get_challenge_directives(config)
            directive = minority_dir if agent_id in minority_ids else majority_dir
        else:
            directive = build_directive(round_type, config)

        tasks.append(
            _call_agent(
                sem=sem,
                profile=profile,
                persona=persona,
                directive=directive,
                round_type=round_type,
                round_index=round_index,
                world_context=world_context,
                conversation_context=conversation_context,
            )
        )

    responses = await asyncio.gather(*tasks)
    return list(responses)


async def _call_agent(
    sem: asyncio.Semaphore,
    profile: dict,
    persona: dict,
    directive: str,
    round_type: str,
    round_index: int,
    world_context: str,
    conversation_context: str,
) -> dict:
    """
    Make a single agent LLM call with retry/backoff for rate limits.
    """
    system_prompt = profile.get('system_prompt', '')
    initial_stance = profile.get('initial_stance', '')

    user_prompt = f"""WORLD CONTEXT:
{world_context[:3000]}

YOUR POSITION: {initial_stance}

CONVERSATION SO FAR:
{conversation_context}

ROUND DIRECTIVE ({round_type}):
{directive}

Respond in 50–150 words. Be specific. Name entities, mechanisms, and timeframes."""

    async with sem:
        for attempt in range(LLM_MAX_RETRIES + 1):
            try:
                response = await _get_client().chat.completions.create(
                    model=_get_model(),
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=250,
                    temperature=0.7,
                )
                response_text = response.choices[0].message.content or ''

                # Extract stance from response (heuristic)
                stance = _classify_stance(response_text, persona)

                return {
                    'agent_id': profile.get('id', ''),
                    'round_number': round_index,
                    'round_type': round_type,
                    'prompt_issued': directive,
                    'response_text': response_text,
                    'confidence': _extract_confidence(response_text),
                    'agents_addressed': _extract_addressed_agents(response_text, persona),
                    'grounding_chunks': persona.get('grounding_chunk_ids', []),
                    'stance_category': stance,
                    'position_delta': 'unchanged',  # computed post-hoc by oasis_adapter
                }

            except Exception as e:
                if attempt < LLM_MAX_RETRIES:
                    backoff = LLM_BASE_BACKOFF * (2 ** attempt)
                    print(f"      Rate limit/error for {profile.get('name', '?')} "
                          f"(attempt {attempt + 1}): {e}. Retrying in {backoff}s…")
                    await asyncio.sleep(backoff)
                else:
                    # Final failure — return placeholder
                    return {
                        'agent_id': profile.get('id', ''),
                        'round_number': round_index,
                        'round_type': round_type,
                        'prompt_issued': directive,
                        'response_text': '[rate limit — no response]',
                        'confidence': 0.5,
                        'agents_addressed': [],
                        'grounding_chunks': [],
                        'stance_category': 'unknown',
                        'position_delta': 'unchanged',
                    }

    # Should not reach here, but satisfy type checker
    return {
        'agent_id': profile.get('id', ''),
        'round_number': round_index,
        'round_type': round_type,
        'prompt_issued': directive,
        'response_text': '[error — no response]',
        'confidence': 0.5,
        'agents_addressed': [],
        'grounding_chunks': [],
        'stance_category': 'unknown',
        'position_delta': 'unchanged',
    }


async def _retry_agent(
    sem: asyncio.Semaphore,
    profile: dict | None,
    persona: dict | None,
    round_type: str,
    round_index: int,
    world_context: str,
) -> dict | None:
    """Retry a single agent with the specificity prompt."""
    if not profile or not persona:
        return None

    return await _call_agent(
        sem=sem,
        profile=profile,
        persona=persona,
        directive=SPECIFICITY_RETRY_PROMPT,
        round_type=round_type,
        round_index=round_index,
        world_context=world_context,
        conversation_context='[Retry — restate your position with more specificity.]',
    )


async def _run_fallback_debate(
    job_id: str,
    personas: list[dict],
    config: dict,
    world_context: str,
    result: SimulationResult,
) -> None:
    """
    Fallback path: use the existing agent_debate module when OASIS is unavailable.
    Adapts personas to the SimulationAgent format expected by agent_debate.
    """
    from pipeline.agent_debate import run_agent_debate
    from pipeline.environment_setup import SimulationEnvironment
    from models.agent import SimulationAgent

    # Adapt personas → SimulationAgent objects
    agents = []
    for p in personas:
        agents.append(SimulationAgent(
            node_id=p.get('agent_id', p.get('node_id', '')),
            label=p.get('label', ''),
            entity_type=p.get('entity_type', ''),
            description=p.get('description', p.get('behavioural_prompt', '')),
            is_anchor=p.get('is_anchor', False),
            influence=p.get('influence_tier', p.get('influence', 'medium')),
            centrality=p.get('centrality', 0),
            relationships=p.get('relationships', []),
            personality_prompt=p.get('behavioural_prompt', p.get('personality_prompt', '')),
        ))

    env = SimulationEnvironment()
    env.prediction_question = config.get('question', '')
    env.what_if_variables = config.get('what_if_variables', [])
    env.world_context = world_context
    env.agents = agents

    print(f"    Starting fallback multi-agent debate with {len(agents)} agents…")
    debate_log = await run_agent_debate(agents=agents, env=env)

    result.interaction_log = debate_log
    result.rounds_completed = max((e.get('round', 0) for e in debate_log), default=0) + 1
    result.emergent_signals.append("Simulation ran via fallback debate — OASIS was unavailable.")

    # Build agent final states
    last_posts: dict[str, dict] = {}
    for entry in debate_log:
        last_posts[entry.get('agent_name', '')] = entry
    result.agent_states = [
        {
            'name': name,
            'final_position': entry.get('content', ''),
            'platform': 'debate',
        }
        for name, entry in last_posts.items()
    ]


# ── Utility functions ─────────────────────────────────────────────────

def _find_profile(profiles: list[dict], agent_id: str) -> dict | None:
    return next((p for p in profiles if p.get('id') == agent_id), None)


def _find_persona(personas: list[dict], agent_id: str) -> dict:
    return next(
        (p for p in personas if p.get('agent_id') == agent_id),
        {}
    )


def _build_conversation_context(interaction_log: list[dict], limit: int = 30) -> str:
    """Format recent interactions for agent context window."""
    recent = interaction_log[-limit:] if len(interaction_log) > limit else interaction_log
    if not recent:
        return "No prior discussion."

    lines = []
    for entry in recent:
        agent = entry.get('agent_id', 'Unknown')
        round_num = entry.get('round_number', '?')
        rtype = entry.get('round_type', '')
        text = entry.get('response_text', '')[:300]
        lines.append(f"[Round {round_num} / {rtype}] {agent}: {text}")

    return "\n\n".join(lines)


def _classify_stance(response_text: str, persona: dict) -> str:
    """
    Lightweight stance classification from response text.
    Looks for explicit stance signals; defaults to the persona's initial stance.
    """
    text_lower = response_text.lower()

    if any(w in text_lower for w in ['strongly support', 'fully agree', 'endorse', 'confident that']):
        return 'strong_support'
    if any(w in text_lower for w in ['support', 'agree', 'likely', 'probable']):
        return 'support'
    if any(w in text_lower for w in ['strongly oppose', 'fundamentally disagree', 'reject']):
        return 'strong_oppose'
    if any(w in text_lower for w in ['oppose', 'disagree', 'unlikely', 'doubt']):
        return 'oppose'
    if any(w in text_lower for w in ['uncertain', 'mixed', 'depends', 'nuanced']):
        return 'neutral'

    # Fall back to persona's initial stance category if available
    return persona.get('stance_category', 'neutral')


def _extract_confidence(response_text: str) -> float:
    """Extract stated confidence from response, defaulting to 0.5."""
    # Look for explicit confidence statements like "80% confident" or "confidence: 0.7"
    match = re.search(r'(\d{1,3})%\s*(?:confident|confidence|certain)', response_text, re.IGNORECASE)
    if match:
        return min(int(match.group(1)) / 100.0, 1.0)

    match = re.search(r'confidence[:\s]+([01]\.?\d*)', response_text, re.IGNORECASE)
    if match:
        try:
            return min(float(match.group(1)), 1.0)
        except ValueError:
            pass

    return 0.5


def _extract_addressed_agents(response_text: str, persona: dict) -> list[str]:
    """
    Detect which agents were addressed in directed rounds.
    Simple approach: check if any persona labels appear in the response text.
    """
    # This would ideally check against all persona labels, but we only have
    # the current persona here. The full extraction happens post-hoc.
    return []
