"""
Re-run the agent debate and save full interaction log + profiles to JSON for the viewer.
"""
import os, sys, json, asyncio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from pipeline.agent_debate import run_agent_debate
from pipeline.environment_setup import SimulationEnvironment
from models.agent import SimulationAgent

SIMULATION_DIR = "uploads/simulations/sim_a835fc7e46ab"
OUTPUT_DIR = "debate_viewer"

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

def load_agents(path):
    with open(path, 'r', encoding='utf-8') as f:
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
    return agents, profiles

async def main():
    agents, raw_profiles = load_agents(os.path.join(SIMULATION_DIR, "reddit_profiles.json"))
    agents = agents[:30]
    raw_profiles = raw_profiles[:30]

    env = SimulationEnvironment()
    env.prediction_question = PREDICTION_QUESTION
    env.what_if_variables = WHAT_IF_VARIABLES
    env.agents = agents
    env.max_rounds = 10

    print(f"Running debate with {len(agents)} agents…")
    debate_log = await run_agent_debate(agents=agents, env=env)

    # Build output data
    personas = []
    for agent, profile in zip(agents, raw_profiles):
        personas.append({
            "name": agent.label,
            "entity_type": agent.entity_type,
            "bio": agent.description,
            "personality": agent.personality_prompt,
            "influence": agent.influence,
            "age": profile.get("age"),
            "gender": profile.get("gender"),
            "mbti": profile.get("mbti"),
            "profession": profile.get("profession"),
            "country": profile.get("country"),
            "interested_topics": profile.get("interested_topics", []),
            "username": profile.get("username", ""),
        })

    output = {
        "prediction_question": PREDICTION_QUESTION,
        "what_if_variables": WHAT_IF_VARIABLES,
        "personas": personas,
        "interactions": debate_log,
        "rounds": max((e["round"] for e in debate_log), default=0) + 1,
        "total_interactions": len(debate_log),
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "debate_data.json"), 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved {len(debate_log)} interactions to {OUTPUT_DIR}/debate_data.json")

if __name__ == "__main__":
    asyncio.run(main())
