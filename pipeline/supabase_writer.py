import os
import threading
from datetime import datetime, timezone
from models.simulation_report import SimulationReport

_client = None
_client_failed = False


def _get_client():
    global _client, _client_failed
    if _client is not None:
        return _client
    if _client_failed:
        return None

    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        print("⚠ SUPABASE_URL or SUPABASE_SERVICE_KEY not set — Supabase writes disabled.")
        _client_failed = True
        return None

    try:
        from supabase import create_client
        _client = create_client(url, key)
        return _client
    except Exception as e:
        print(f"⚠ Supabase client creation failed: {e}")
        _client_failed = True
        return None


async def write_job_result(job_id: str, report: SimulationReport) -> None:
    """Write the completed simulation report to Supabase."""
    client = _get_client()
    if not client:
        print(f"  [{job_id}] Supabase unavailable — report printed locally only:")
        print(f"    headline: {report.headline}")
        print(f"    confidence: {report.confidence_level}")
        return
    try:
        result = client.table("simulation_jobs").update({
            "status": "completed",
            "progress": 100,
            "progress_message": "Simulation complete.",
            "result": report.model_dump(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", job_id).execute()
        print(f"[SUPABASE] write_job_result rows affected: {len(result.data) if result.data else 0}")
    except Exception as e:
        print(f"  ⚠ Supabase write_job_result failed: {e}")
        print(f"    headline: {report.headline}")
        print(f"    confidence: {report.confidence_level}")


async def write_job_error(job_id: str, error_message: str) -> None:
    """Mark a simulation as failed in Supabase."""
    client = _get_client()
    if not client:
        return
    try:
        client.table("simulation_jobs").update({
            "status": "failed",
            "progress": 0,
            "progress_message": None,
            "error_message": error_message,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", job_id).execute()
    except Exception as e:
        print(f"  ⚠ Supabase write_job_error failed: {e}")


async def write_interaction_log(job_id: str, interaction_log: list[dict]) -> None:
    """Write the structured interaction log to simulation_jobs.interaction_log (JSONB)."""
    client = _get_client()
    if not client:
        print(f"  [{job_id}] Supabase unavailable — interaction log not persisted.")
        return
    try:
        client.table("simulation_jobs").update({
            "interaction_log": interaction_log,
        }).eq("id", job_id).execute()
        print(f"  [{job_id}] Interaction log written ({len(interaction_log)} entries)")
    except Exception as e:
        print(f"  ⚠ Supabase write_interaction_log failed: {e}")


def update_round_progress(
    job_id: str,
    current_round: int,
    total_rounds: int,
    round_type: str,
    delta_count: int,
) -> None:
    """
    Fire-and-forget progress update after each round.
    Runs in a daemon thread to avoid blocking the simulation loop.
    """
    def _write():
        client = _get_client()
        if not client:
            return
        try:
            client.table("simulation_jobs").update({
                "progress": {
                    "current_round": current_round,
                    "total_rounds": total_rounds,
                    "round_type": round_type,
                    "delta_count": delta_count,
                    "pct": round((current_round / total_rounds) * 100),
                },
                "status": "running",
            }).eq("id", job_id).execute()
        except Exception as e:
            print(f"  ⚠ Round progress write failed: {e}")

    thread = threading.Thread(target=_write, daemon=True)
    thread.start()


def update_job_status(job_id: str, status: str, note: str = '') -> None:
    """
    Update job status with optional note (e.g. stagnation, timeout).
    Fire-and-forget via daemon thread.
    """
    def _write():
        client = _get_client()
        if not client:
            return
        try:
            update = {"status": status}
            if note:
                update["progress_message"] = note
            client.table("simulation_jobs").update(update).eq("id", job_id).execute()
        except Exception as e:
            print(f"  ⚠ Job status update failed: {e}")

    thread = threading.Thread(target=_write, daemon=True)
    thread.start()
