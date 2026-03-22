FROM python:3.11-slim

WORKDIR /app

# Build tools needed by igraph, cairocffi C extensions
RUN apt-get update && apt-get install -y \
    gcc g++ make git \
    && rm -rf /var/lib/apt/lists/*

# 1. Install core sidecar deps (lightweight — fastapi, supabase, openai, camel-ai)
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=120 -r requirements.txt

# 2. Install camel-oasis from GitHub WITHOUT its heavy transitive deps
#    (torch, sentence-transformers, neo4j, pandas, igraph are only used in
#     code paths we don't call — recsys, agent_graph, agents_generator)
RUN pip install --no-cache-dir --no-deps \
    "camel-oasis @ git+https://github.com/camel-ai/oasis.git@v0.2.5"

COPY . .

EXPOSE 8000
CMD ["python", "run.py"]
