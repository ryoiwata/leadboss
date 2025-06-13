# docker/dev.Dockerfile  (only the pieces you need for ingest)
FROM python:3.11-slim
RUN apt-get update && apt-get install -y git wget && rm -rf /var/lib/apt/lists/*
COPY requirements_ingest.txt .
RUN pip install -r requirements_ingest.txt
WORKDIR /app