FROM python:3.11-slim

WORKDIR /app

# Install build dependencies required by camel-oasis transitive deps
# (igraph, cairocffi, sentence-transformers need C compilation toolchain;
#  git needed for pip install from GitHub)
RUN apt-get update && apt-get install -y \
    gcc g++ make git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=120 -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["python", "run.py"]
