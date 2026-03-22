"""
Round Directives — prompt templates for each round type and simulation mode.

Five round types form the deliberation structure:
  opening   → agents state initial positions (broadcast)
  reaction  → agents respond to each other (directed)
  challenge → minority position advocacy (directed, split directives)
  revision  → agents update positions based on discussion (broadcast)
  closing   → agents lock final falsifiable claims (broadcast)

Round sequences are indexed by depth tier (from SimulationConfig).
Mode overlays append focus instructions per simulation mode.
"""

# ── Round type definitions ────────────────────────────────────────────

ROUND_TYPES = {
    'opening': {
        'label': 'Initial positions',
        'interaction_mode': 'broadcast',
        'directive': (
            "Given the world context and your known position, state your assessment "
            "of the following question: {question}. Be specific. Name entities, "
            "mechanisms, and timeframes. Ground your position in the evidence you "
            "have been given."
        ),
    },
    'reaction': {
        'label': 'Response to others',
        'interaction_mode': 'directed',
        'directive': (
            "You have read the positions of other participants. Identify the two "
            "positions you most agree or disagree with and explain why. Address "
            "those agents directly. Be specific about what in their argument you "
            "are responding to."
        ),
    },
    'challenge': {
        'label': 'Minority position advocacy',
        'interaction_mode': 'directed',
        'directive_minority': (
            "You hold a minority position. Make your strongest case to the group. "
            "What are they missing? What evidence supports your view that they "
            "have not engaged with?"
        ),
        'directive_majority': (
            "A minority position has been presented. Engage with it seriously. "
            "What is the strongest element of their argument? Does it cause you "
            "to revise anything?"
        ),
    },
    'revision': {
        'label': 'Position update',
        'interaction_mode': 'broadcast',
        'directive': (
            "Has anything in this deliberation caused you to revise your position? "
            "If yes: state your updated position and exactly what argument or "
            "evidence changed your mind. If no: state what argument would have had "
            "to be made to move you, and why none of the arguments presented "
            "reached that bar."
        ),
    },
    'closing': {
        'label': 'Final falsifiable position',
        'interaction_mode': 'broadcast',
        'directive': (
            "State your final position as a specific, falsifiable claim. Include: "
            "what you predict will happen, by when, triggered by what condition, "
            "and how someone would verify you were right or wrong. Do not hedge. "
            "If you have uncertainty, quantify it."
        ),
    },
}

# ── Round sequences by depth tier ────────────────────────────────────

ROUND_SEQUENCES = {
    'quick_scan': ['opening', 'closing'],
    'standard': ['opening', 'reaction', 'reaction', 'revision', 'closing'],
    'deep_dive': [
        'opening', 'reaction', 'reaction', 'challenge',
        'reaction', 'revision', 'revision', 'closing',
    ],
    'exhaustive': [
        'opening', 'reaction', 'reaction', 'reaction',
        'challenge', 'reaction', 'revision', 'challenge',
        'revision', 'revision', 'reaction', 'closing',
    ],
}

# ── Mode-specific directive overlays ─────────────────────────────────

MODE_OVERLAYS = {
    'prediction': "Focus on probability. What is most likely to happen?",
    'hypothesis_test': "Evaluate the hypothesis: {question}. Support or refute it with evidence.",
    'contrarian_scan': "Surface what the consensus is missing. Prioritise minority signals.",
    'optimisation': "Evaluate available paths. What course of action best serves your interests?",
    'consensus_mapping': "Find common ground. What can you agree on with others regardless of other disagreements?",
}

# ── User-facing round type labels (for frontend activity feed) ───────

ROUND_TYPE_LABELS = {
    'opening': 'Initial positions established',
    'reaction': 'Agents responding to each other',
    'challenge': 'Minority position presented',
    'revision': 'Position revision round',
    'closing': 'Final positions locked',
}


def build_directive(round_type: str, config: dict) -> str:
    """
    Build the full directive for a round, combining the base template
    with the mode overlay.

    Args:
        round_type: One of the ROUND_TYPES keys.
        config: SimulationConfig dict — must contain 'question' and 'mode'.

    Returns:
        Complete directive string ready to issue to agents.
    """
    round_def = ROUND_TYPES[round_type]
    question = config.get('question', '')
    mode = config.get('mode', 'prediction')

    # Base directive (challenge rounds have split directives handled by caller)
    base = round_def.get('directive', '')
    if not base and round_type == 'challenge':
        # Default to majority directive; caller splits minority/majority
        base = round_def.get('directive_majority', '')

    directive = base.format(question=question)

    # Append mode overlay
    overlay = MODE_OVERLAYS.get(mode, '')
    if overlay:
        overlay = overlay.format(question=question)
        directive = f"{directive}\n\n{overlay}"

    return directive


def get_challenge_directives(config: dict) -> tuple[str, str]:
    """
    Return (minority_directive, majority_directive) for challenge rounds.
    """
    question = config.get('question', '')
    mode = config.get('mode', 'prediction')
    overlay = MODE_OVERLAYS.get(mode, '').format(question=question)

    minority = ROUND_TYPES['challenge']['directive_minority'].format(question=question)
    majority = ROUND_TYPES['challenge']['directive_majority'].format(question=question)

    if overlay:
        minority = f"{minority}\n\n{overlay}"
        majority = f"{majority}\n\n{overlay}"

    return minority, majority
