FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/opt/hf FASTEMBED_CACHE_PATH=/opt/fastembed
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# Bake the embedding model, Silero VAD and the turn-detector files into the image.
RUN python -c "from voicedesk.rag import embed_query; embed_query('warm up')" \
    && python -m agent.main download-files

EXPOSE 8000 50051 9464
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
