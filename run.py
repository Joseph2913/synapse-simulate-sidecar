import uvicorn
import os
import sys

# Disable stdout buffering so background task logs appear immediately
os.environ["PYTHONUNBUFFERED"] = "1"

from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    # Railway sets RAILWAY_ENVIRONMENT; any non-local env disables reload
    is_local = not os.getenv("RAILWAY_ENVIRONMENT") and not os.getenv("RAILWAY_PROJECT_ID")
    print(f"Starting Synapse Simulate Sidecar on http://localhost:{port}")
    if is_local:
        print("Press Ctrl+C to stop.\n")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=is_local,
        log_level="info",
    )
