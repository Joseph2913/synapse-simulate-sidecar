import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from fastapi import Request
from models.seed_graph import SimulateRequest
from pipeline.graph_import import import_graph
from pipeline.environment_setup import build_environment
from pipeline.simulation_runner import run_simulation
from pipeline.report_gen import generate_report
from pipeline.supabase_writer import (
    write_job_result,
    write_job_error,
    write_interaction_log,
)

from dotenv import load_dotenv

load_dotenv()

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")

@asynccontextmanager
async def lifespan(app: FastAPI):
    env = os.getenv("RAILWAY_ENVIRONMENT", "local")
    print(f"✓ Synapse Simulate Sidecar starting… (env={env})")
    print(f"  LLM: {os.getenv('LLM_MODEL_NAME')} via {os.getenv('LLM_BASE_URL')}")
    print(f"  Supabase: {os.getenv('SUPABASE_URL')}")
    yield
    print("Sidecar shutting down.")

app = FastAPI(title="Synapse Simulate Sidecar", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ─── HEALTH ──────────────────────────────────────────────────────────

_ready = True

@app.get("/health")
async def health():
    """Called by Synapse on page load to confirm sidecar is running."""
    if not _ready:
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=503, content={"status": "starting"})
    return {
        "status": "ok",
        "model": os.getenv("LLM_MODEL_NAME"),
        "version": "2.0.0",
    }

# ─── SIMULATE ────────────────────────────────────────────────────────

@app.post("/simulate")
async def simulate(raw_request: Request, background_tasks: BackgroundTasks):
    """
    Accepts a seed graph + config + personas from Synapse and starts
    a simulation in the background. Returns immediately — result is
    written to Supabase when done.
    """
    body = await raw_request.body()
    raw_text = body.decode('utf-8')[:2000]
    print(f"[DEBUG] Raw request body:\n{raw_text}")

    try:
        request = SimulateRequest.model_validate_json(body)
    except Exception as e:
        print(f"[DEBUG] Pydantic validation error:\n{e}")
        raise HTTPException(status_code=422, detail=str(e))

    if not request.seed_graph.nodes:
        raise HTTPException(status_code=422, detail="Seed graph contains no nodes.")

    print(f"[INCOMING] job_id={request.job_id}")
    print(f"[INCOMING] nodes={len(request.seed_graph.nodes)}, edges={len(request.seed_graph.edges)}")
    print(f"[INCOMING] question={request.config.question[:100]}")
    print(f"[INCOMING] depth={request.config.depth}, mode={request.config.mode}")
    print(f"[INCOMING] personas={len(request.personas)}")

    background_tasks.add_task(run_simulation_pipeline, request)
    return {"status": "accepted", "job_id": request.job_id}

# ─── PIPELINE ORCHESTRATION ──────────────────────────────────────────

async def run_simulation_pipeline(request: SimulateRequest) -> None:
    """
    Full pipeline — runs as a FastAPI background task.
    Stages: graph import → environment build → simulation → report → Supabase write.
    """
    job_id = request.job_id

    try:
        # ── Stage 1: Import graph ──────────────────────────────
        print(f"  [{job_id}] Importing knowledge graph…")
        graph = import_graph(request.seed_graph)

        # ── Stage 2: Build environment ─────────────────────────
        # Personas arrive pre-built from PRD-Simulate-D (or as stubs)
        personas = [p.model_dump() if hasattr(p, 'model_dump') else p for p in request.personas]
        config = request.config.model_dump()

        env = build_environment(
            seed_graph=request.seed_graph,
            prediction_question=config['question'],
            what_if_variables=config.get('what_if_variables', []),
        )

        # ── Stage 3: Run simulation ────────────────────────────
        print(f"  [{job_id}] Starting simulation (depth={config['depth']}, "
              f"mode={config['mode']}, agents={len(personas)})…")

        sim_result = await run_simulation(
            job_id=job_id,
            seed_graph=request.seed_graph.model_dump(),
            personas=personas,
            config=config,
            world_context=env.world_context,
        )

        print(f"  [{job_id}] Simulation done. rounds={sim_result.rounds_completed}, "
              f"agents={len(sim_result.agent_states)}, fallback={sim_result.used_fallback}")

        # ── Stage 4: Store interaction log ──────────────────────
        if sim_result.structured_log:
            await write_interaction_log(job_id, sim_result.structured_log)
        elif sim_result.interaction_log:
            await write_interaction_log(job_id, sim_result.interaction_log)

        # ── Stage 5: Generate report ────────────────────────────
        print(f"  [{job_id}] Generating prediction report…")
        report = await generate_report(sim_result, env)

        # ── Stage 6: Write result to Supabase ───────────────────
        await write_job_result(job_id, report)
        print(f"✓ [{job_id}] Completed. headline={report.headline}")

    except Exception as e:
        import traceback
        print(f"[PIPELINE ERROR] {str(e)}")
        print(traceback.format_exc())
        await write_job_error(job_id, str(e))
