# ── Stage 1: Build ────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    gcc g++ make git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=120 -r requirements.txt
RUN pip install --no-cache-dir --no-deps \
    "camel-oasis @ git+https://github.com/camel-ai/oasis.git@v0.2.5"

# ── Stage 2: Runtime (no gcc/g++/make/git, no pip cache, no .pyc) ─────
FROM python:3.11-slim

WORKDIR /app

# Copy only the installed packages from the builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Remove .pyc and __pycache__ to slim down further
RUN find /usr/local/lib/python3.11/site-packages -name '__pycache__' -exec rm -rf {} + 2>/dev/null; true

COPY . .

EXPOSE 8000
CMD ["python", "run.py"]
