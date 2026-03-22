"""
Simulation Moderator — runs between OASIS timesteps.

The moderator is NOT an OASIS agent. It is a Python wrapper that intercepts the
simulation loop, evaluates state, and decides what happens next:

  - Round sequencing (advance to next round type)
  - Stagnation detection (2 consecutive zero-delta rounds)
  - Pressure injection (one LLM call to generate a destabilising challenge)
  - Specificity retry (vague responses get one retry with a sharpened prompt)
"""

import os
import re
from collections import Counter
from openai import AsyncOpenAI

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


class SimulationModerator:
    def __init__(self, config: dict, personas: list[dict], round_sequence: list[str]):
        self.config = config
        self.personas = personas
        self.round_sequence = round_sequence
        self.current_round_index = 0
        self.stagnation_count = 0
        self.position_history: dict[str, list[str]] = {}  # agent_id → summaries

    def get_current_round_directive(self) -> dict:
        """Return the current round's type and directive."""
        from pipeline.round_directives import build_directive
        round_type = self.round_sequence[self.current_round_index]
        directive = build_directive(round_type, self.config)
        return {'round_type': round_type, 'directive': directive}

    def evaluate_after_round(self, round_responses: list[dict]) -> str:
        """
        Evaluate round results and decide next action.

        Returns one of:
          'advance'           — proceed to next round
          'inject_pressure'   — insert a destabilising moderator challenge
          'end_early'         — stop simulation (stagnation near end)
          'retry_specificity' — vague responses need one retry
        """
        # Check for vague responses first
        vague_agents = [r for r in round_responses if is_vague(r.get('response_text', ''))]
        if vague_agents:
            return 'retry_specificity'

        # Check for stagnation (no position deltas)
        deltas = [r for r in round_responses if r.get('position_delta', 'unchanged') != 'unchanged']
        if len(deltas) == 0:
            self.stagnation_count += 1
        else:
            self.stagnation_count = 0

        if self.stagnation_count >= 2:
            # Two consecutive rounds with no movement
            if self.current_round_index < len(self.round_sequence) - 2:
                return 'inject_pressure'
            else:
                return 'end_early'

        return 'advance'

    def identify_minority_agents(self, round_responses: list[dict]) -> list[str]:
        """
        Identify agents holding a minority position in the current round.
        Uses stance_category to count alignment; agents in stances held by
        < 40% of participants are classified as minority.
        """
        if not round_responses:
            return []

        stance_counts = Counter(
            r.get('stance_category', 'unknown') for r in round_responses
        )
        total = len(round_responses)
        majority_stances = {
            s for s, c in stance_counts.items()
            if c >= total * 0.6
        }
        minority_agents = [
            r.get('agent_id', '')
            for r in round_responses
            if r.get('stance_category', 'unknown') not in majority_stances
        ]
        return minority_agents

    async def inject_pressure(self, world_context: str, interaction_log: list[dict]) -> str:
        """
        Generate a destabilising challenge when consensus forms too quickly.
        Single LLM call — the one moment of moderator intelligence.
        """
        consensus_summary = _summarise_consensus(interaction_log)

        prompt = f"""You are a simulation moderator. The agents in this deliberation have reached \
an unusually quick consensus. Your job is to inject a destabilising challenge.

Question being deliberated: {self.config.get('question', '')}
Current consensus position: {consensus_summary}
World context summary: {world_context[:2000]}

Generate one specific, evidence-grounded challenge that the consensus position \
has not adequately addressed. This should be a genuine weakness or overlooked \
consideration — not a generic devil's advocate position.

Return one sentence only."""

        try:
            response = await _get_client().chat.completions.create(
                model=_get_model(),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.7,
            )
            return (response.choices[0].message.content or '').strip()
        except Exception as e:
            return f"Consider: what would need to be true for the current consensus to be wrong? [Moderator auto-challenge — LLM call failed: {e}]"


# ── Specificity check ─────────────────────────────────────────────────

def is_vague(response_text: str) -> bool:
    """
    Heuristic check for vague responses.
    Vague if: under 50 words, contains no named entities, or is all hedging.
    """
    if not response_text:
        return True

    word_count = len(response_text.split())

    # Too short
    if word_count < 50:
        return True

    # No named entities (capitalised words beyond sentence start)
    has_named_entity = bool(re.search(r'(?<!\. )\b[A-Z][a-z]+\b', response_text))
    if not has_named_entity:
        return True

    # All hedging with no specificity
    hedge_phrases = ['might', 'could', 'possibly', 'perhaps', 'maybe']
    hedge_count = sum(1 for p in hedge_phrases if p in response_text.lower())
    if hedge_count >= 3 and word_count < 80:
        return True

    return False


SPECIFICITY_RETRY_PROMPT = (
    "Your previous response was too general. Restate your position with "
    "specific named entities, a timeframe, and a concrete mechanism."
)


# ── Internal helpers ──────────────────────────────────────────────────

def _summarise_consensus(interaction_log: list[dict]) -> str:
    """
    Build a quick summary of the current consensus from the most recent
    round's responses. Takes the most common stance and a sample response.
    """
    if not interaction_log:
        return "No interactions recorded yet."

    # Get the latest round
    max_round = max(e.get('round_number', 0) for e in interaction_log)
    latest = [e for e in interaction_log if e.get('round_number', 0) == max_round]

    if not latest:
        return "No interactions in latest round."

    # Count stances
    stance_counts = Counter(e.get('stance_category', 'unknown') for e in latest)
    most_common = stance_counts.most_common(1)
    dominant_stance = most_common[0][0] if most_common else 'unknown'
    dominant_count = most_common[0][1] if most_common else 0

    # Sample response from dominant stance
    sample = next(
        (e.get('response_text', '')[:200] for e in latest
         if e.get('stance_category') == dominant_stance),
        ''
    )

    return (
        f"{dominant_count}/{len(latest)} agents hold stance '{dominant_stance}'. "
        f"Sample: {sample}"
    )
