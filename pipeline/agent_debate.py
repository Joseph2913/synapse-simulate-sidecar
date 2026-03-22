"""
Multi-Round Agent Debate System

Replaces the OASIS dependency with a lightweight LLM-driven debate.
Each round, a subset of agents reads the conversation so far and contributes:
  - Initial takes (Round 0)
  - Replies, rebuttals, agreements, new angles (Rounds 1–N)

Produces a structured interaction log identical in shape to what the
report generator expects.
"""

import os
import asyncio
import random
import json
from datetime import datetime, timezone
from openai import AsyncOpenAI
from models.agent import SimulationAgent
from pipeline.environment_setup import SimulationEnvironment

client = AsyncOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
)

MODEL = os.getenv("LLM_MODEL_NAME", "gemini-2.0-flash")

# Concurrency limit to avoid rate-limiting
SEMAPHORE_LIMIT = int(os.getenv("DEBATE_CONCURRENCY", 5))

NUM_ROUNDS = int(os.getenv("DEBATE_ROUNDS", 5))
INITIAL_SPEAKERS = int(os.getenv("DEBATE_INITIAL_SPEAKERS", 8))
AGENTS_PER_ROUND = int(os.getenv("DEBATE_AGENTS_PER_ROUND", 10))


# ── Prompt templates ───────────────────────────────────────────────

INITIAL_POST_SYSTEM = """You are playing the role of a specific entity in a multi-agent prediction debate.
Stay completely in character. Post your genuine initial position on the prediction question.
Be specific, opinionated, and grounded in your domain expertise.
Keep your post to 2–4 sentences. Do NOT use any JSON formatting — write plain text only."""

REPLY_SYSTEM = """You are playing the role of a specific entity in a multi-agent prediction debate.
You have read the recent discussion and must now contribute.
You may: agree with someone (explain why), disagree (explain why), add a new angle others missed,
challenge an assumption, or shift your position based on compelling arguments.
Be specific and reference other participants by name when responding to their points.
Keep your response to 2–4 sentences. Do NOT use any JSON formatting — write plain text only."""


async def run_agent_debate(
    agents: list[SimulationAgent],
    env: SimulationEnvironment,
) -> list[dict]:
    """
    Run a multi-round debate among agents. Returns a flat interaction log.

    Each entry: {
        "round": int,
        "agent_name": str,
        "entity_type": str,
        "action": "post" | "reply",
        "content": str,
        "reply_to": str | None,
        "timestamp": str,
    }
    """
    sem = asyncio.Semaphore(SEMAPHORE_LIMIT)
    interaction_log: list[dict] = []

    question_context = _build_question_context(env)

    # ── Round 0: Initial positions ────────────────────────────────
    initial_agents = _select_diverse_agents(agents, count=INITIAL_SPEAKERS)
    print(f"    [Debate] Round 0: {len(initial_agents)} agents posting initial positions")

    initial_tasks = [
        _generate_initial_post(sem, agent, question_context)
        for agent in initial_agents
    ]
    initial_posts = await asyncio.gather(*initial_tasks)

    for agent, content in zip(initial_agents, initial_posts):
        entry = {
            "round": 0,
            "agent_name": agent.label,
            "entity_type": agent.entity_type,
            "action": "post",
            "content": content,
            "reply_to": None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        interaction_log.append(entry)

    # ── Rounds 1–N: Debate ────────────────────────────────────────
    for round_num in range(1, NUM_ROUNDS + 1):
        active_agents = _select_agents_for_round(agents, AGENTS_PER_ROUND, round_num)
        print(f"    [Debate] Round {round_num}/{NUM_ROUNDS}: {len(active_agents)} agents responding")

        # Build a digest of the conversation so far
        conversation_digest = _build_conversation_digest(interaction_log, max_entries=20)

        reply_tasks = [
            _generate_reply(sem, agent, question_context, conversation_digest, interaction_log)
            for agent in active_agents
        ]
        replies = await asyncio.gather(*reply_tasks)

        for agent, (content, reply_to) in zip(active_agents, replies):
            entry = {
                "round": round_num,
                "agent_name": agent.label,
                "entity_type": agent.entity_type,
                "action": "reply",
                "content": content,
                "reply_to": reply_to,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            interaction_log.append(entry)

    print(f"    [Debate] Complete. {len(interaction_log)} total interactions across {NUM_ROUNDS + 1} rounds.")
    return interaction_log


# ── Internal helpers ───────────────────────────────────────────────

def _build_question_context(env: SimulationEnvironment) -> str:
    what_if = "\n".join(f"- {v}" for v in env.what_if_variables) if env.what_if_variables else "None."
    return f"""PREDICTION QUESTION: {env.prediction_question}

WHAT-IF CONDITIONS:
{what_if}"""


def _select_diverse_agents(agents: list[SimulationAgent], count: int) -> list[SimulationAgent]:
    """Pick initial speakers: prioritise high-influence + diverse entity types."""
    by_type: dict[str, list[SimulationAgent]] = {}
    for a in agents:
        by_type.setdefault(a.entity_type, []).append(a)

    selected: list[SimulationAgent] = []
    # Round-robin across entity types, high influence first
    for entity_type in by_type:
        by_type[entity_type].sort(key=lambda a: (
            {"high": 0, "medium": 1, "low": 2}.get(a.influence, 2),
            -a.centrality,
        ))

    type_keys = list(by_type.keys())
    idx = 0
    while len(selected) < count and any(by_type.values()):
        key = type_keys[idx % len(type_keys)]
        if by_type[key]:
            selected.append(by_type[key].pop(0))
        idx += 1

    return selected


def _select_agents_for_round(
    agents: list[SimulationAgent],
    count: int,
    round_num: int,
) -> list[SimulationAgent]:
    """Weighted random selection — high-influence agents speak more often."""
    weights = []
    for a in agents:
        w = {"high": 3.0, "medium": 2.0, "low": 1.0}.get(a.influence, 1.0)
        weights.append(w)

    count = min(count, len(agents))
    selected_indices = set()
    # Weighted sampling without replacement
    pool = list(range(len(agents)))
    pool_weights = list(weights)
    while len(selected_indices) < count and pool:
        chosen = random.choices(pool, weights=pool_weights, k=1)[0]
        pool_idx = pool.index(chosen)
        selected_indices.add(chosen)
        pool.pop(pool_idx)
        pool_weights.pop(pool_idx)

    return [agents[i] for i in selected_indices]


def _build_conversation_digest(log: list[dict], max_entries: int = 20) -> str:
    """Format recent conversation for the reply prompt."""
    recent = log[-max_entries:]
    lines = []
    for entry in recent:
        prefix = f"[Round {entry['round']}] {entry['agent_name']} ({entry['entity_type']})"
        if entry.get("reply_to"):
            prefix += f" → replying to {entry['reply_to']}"
        lines.append(f"{prefix}:\n{entry['content']}")
    return "\n\n".join(lines)


async def _generate_initial_post(
    sem: asyncio.Semaphore,
    agent: SimulationAgent,
    question_context: str,
) -> str:
    async with sem:
        user_prompt = f"""{question_context}

YOUR IDENTITY:
Name: {agent.label}
Type: {agent.entity_type}
Background: {agent.description}
Persona: {agent.personality_prompt}

Post your initial position on this prediction question. What do you believe will happen and why?"""

        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": INITIAL_POST_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=200,
            temperature=0.8,
        )
        return response.choices[0].message.content or f"{agent.label} has no comment at this time."


async def _generate_reply(
    sem: asyncio.Semaphore,
    agent: SimulationAgent,
    question_context: str,
    conversation_digest: str,
    full_log: list[dict],
) -> tuple[str, str | None]:
    """Returns (content, reply_to_agent_name_or_none)."""
    # Pick a recent post to potentially reply to
    recent_others = [e for e in full_log[-15:] if e["agent_name"] != agent.label]
    reply_target = random.choice(recent_others)["agent_name"] if recent_others else None

    async with sem:
        user_prompt = f"""{question_context}

YOUR IDENTITY:
Name: {agent.label}
Type: {agent.entity_type}
Background: {agent.description}
Persona: {agent.personality_prompt}

RECENT DISCUSSION:
{conversation_digest}

Respond to the discussion. You may agree, disagree, challenge, or add a new perspective.
Reference specific participants by name when engaging with their points."""

        response = await client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": REPLY_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=200,
            temperature=0.8,
        )
        content = response.choices[0].message.content or f"{agent.label} observes the discussion."

        # Try to detect who they actually replied to from content
        detected_target = reply_target
        for entry in recent_others:
            if entry["agent_name"] in content:
                detected_target = entry["agent_name"]
                break

        return content, detected_target
