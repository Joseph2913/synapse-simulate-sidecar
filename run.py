import uvicorn
import os
import sys

# Disable stdout buffering so background task logs appear immediately
os.environ["PYTHONUNBUFFERED"] = "1"

from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    is_production = os.getenv("RAILWAY_ENVIRONMENT") == "production"
    print(f"Starting Synapse Simulate Sidecar on http://localhost:{port}")
    if not is_production:
        print("Press Ctrl+C to stop.\n")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=not is_production,
        log_level="info",
    )
