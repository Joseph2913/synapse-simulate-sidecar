"""
Re-run simulation using existing personas + config from last run.
Skips persona generation and config generation — goes straight to
OASIS (or agent debate fallback) → report → Supabase write.
"""

import os
import sys
import json
import asyncio
from datetime import datetime, timezone

# Ensure we can import project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from pipeline.simulation_runner import SimulationResult
from pipeline.report_gen import generate_report
from pipeline.supabase_writer import write_job_result, write_job_error
from pipeline.environment_setup import SimulationEnvironment
from models.agent import SimulationAgent

# ── Config: reuse from last run ──────────────────────────────────
SIMULATION_DIR = "uploads/simulations/sim_a835fc7e46ab"
LAST_JOB_ID = "34a38333-48ba-4b6e-8a26-e84b551f1cdf"

PREDICTION_QUESTION = (
    "As AI coding and reasoning capabilities cross the threshold of autonomous "
    "software creation, how will the skills required of knowledge workers transform, "
    "how will enterprise software vendors like SAP adapt or decline, and will "
    "organisations converge on standardised AI-native platforms or diverge into a "
    "world of fully custom per-organisation software stacks — and what does this "
    "mean for how job descriptions are written, hired against, and made obsolete?"
)

WHAT_IF_VARIABLES = [
    "AI coding agents reach 90% task autonomy for enterprise software development "
    "by end of 2026 — a non-technical founder can ship a full ERP system without a single engineer",
    "A major global employer publicly eliminates the software engineer job title entirely, "
    "replacing it with AI Systems Director — triggering a wave of job description rewrites "
    "across the industry",
]


def load_agents_from_profiles(profiles_path: str) -> list[SimulationAgent]:
    """Load agent data from the existing reddit_profiles.json."""
    with open(profiles_path, 'r', encoding='utf-8') as f:
        profiles = json.load(f)

    agents = []
    for p in profiles:
        agents.append(SimulationAgent(
            node_id=p.get("source_entity_uuid", str(p.get("user_id", ""))),
            label=p.get("name", p.get("username", "Unknown")),
            entity_type=p.get("source_entity_type", "Person"),
            description=p.get("bio", ""),
            is_anchor=False,
            influence="medium",
            centrality=0,
            relationships=[],
            personality_prompt=p.get("persona", p.get("bio", "")),
        ))
    return agents


async def main():
    profiles_path = os.path.join(SIMULATION_DIR, "reddit_profiles.json")
    if not os.path.exists(profiles_path):
        print(f"ERROR: Profiles not found at {profiles_path}")
        sys.exit(1)

    # ── Load existing agents ──────────────────────────────────────
    print(f"Loading agents from {profiles_path}…")
    agents = load_agents_from_profiles(profiles_path)
    print(f"Loaded {len(agents)} agents")

    # Cap to top 30 for debate (matching MAX_AGENTS from persona_gen)
    agents = agents[:30]

    # ── Build environment ─────────────────────────────────────────
    env = SimulationEnvironment()
    env.prediction_question = PREDICTION_QUESTION
    env.what_if_variables = WHAT_IF_VARIABLES
    env.agents = agents
    env.max_rounds = 10
    env.source_context = ""
    env.world_context = f"PREDICTION QUESTION: {PREDICTION_QUESTION}"

    # ── Create a new Supabase job row ────────────────────────────
    from pipeline.supabase_writer import _get_client
    client = _get_client()
    if client:
        job_row = client.table("simulation_jobs").insert({
            "user_id": "b9264b41-bee4-49a7-a141-c37764f60216",
            "status": "running",
            "title": f"Simulation (with debate) — {datetime.now().strftime('%d %b %Y %H:%M')}",
            "scope_anchor_ids": [
                "a6dd0cef-22a1-4dd8-90ab-eeecbedc87c5",
                "44f71afc-5a06-4b46-bf36-d5d1d93d9f7f",
                "157dec5f-7079-4ed8-98d7-3fff88020160",
                "732e3a07-a3a3-4883-9081-dd4167e0b1e0",
                "77594553-b556-446f-a498-d64feec2fbce",
                "da927a25-4ac3-48e7-8401-03e89b74652e",
                "d6fab628-336d-487c-8e0f-56aa5de7ede2",
                "8fd47d7a-3ad7-4d4f-99d0-a6629a907b71",
            ],
            "scope_time_window_days": 90,
            "scope_node_count": 150,
            "scope_edge_count": 0,
            "scope_source_count": 10,
            "prediction_question": PREDICTION_QUESTION,
            "what_if_variables": WHAT_IF_VARIABLES,
            "excluded_node_ids": [],
            "progress": 20,
            "progress_message": "Running multi-agent debate…",
        }).execute()
        job_id = job_row.data[0]["id"]
        print(f"Created Supabase job: {job_id}")
    else:
        job_id = "local-rerun"
        print("Supabase unavailable — running locally only")

    # ── Run the agent debate ─────────────────────────────────────
    try:
        from pipeline.agent_debate import run_agent_debate

        print(f"\n{'='*60}")
        print(f"Starting multi-agent debate with {len(agents)} agents…")
        print(f"{'='*60}\n")

        debate_log = await run_agent_debate(agents=agents, env=env)

        # Build result
        result = SimulationResult()
        result.interaction_log = debate_log
        result.rounds_completed = max((e["round"] for e in debate_log), default=0) + 1

        last_posts: dict[str, dict] = {}
        for entry in debate_log:
            last_posts[entry["agent_name"]] = entry
        result.agent_states = [
            {"name": name, "final_position": entry["content"], "platform": "debate"}
            for name, entry in last_posts.items()
        ]

        disagreements = [e for e in debate_log if "disagree" in e["content"].lower()]
        if disagreements:
            result.emergent_signals.append(
                f"{len(disagreements)} points of disagreement emerged during debate."
            )

        print(f"\n{'='*60}")
        print(f"Debate complete: {len(debate_log)} interactions, {result.rounds_completed} rounds")
        print(f"{'='*60}\n")

        # ── Generate report ───────────────────────────────────────
        print("Generating prediction report…")
        report = await generate_report(result, env, agents)
        print(f"Report generated: {report.headline}")

        # ── Write to Supabase ─────────────────────────────────────
        await write_job_result(job_id, report)
        print(f"\n✓ Done! Job {job_id} — check Synapse UI.")

    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        await write_job_error(job_id, str(e))


if __name__ == "__main__":
    asyncio.run(main())
