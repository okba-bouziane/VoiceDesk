# Hosted demo worker for LiveKit Cloud Agents: the agent plus the booking gRPC service in one
# container. Storage is SQLite and the vector index is in memory (no Postgres in this mode).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/opt/hf FASTEMBED_CACHE_PATH=/opt/fastembed \
    BOOKING_STORE=sqlite RAG_BACKEND=memory BOOKING_GRPC_ADDR=127.0.0.1:50051 \
    LLM_PROVIDER=groq STT_PROVIDER=groq TTS_PROVIDER=groq \
    LLM_MODEL=openai/gpt-oss-20b STT_MODEL=whisper-large-v3-turbo \
    TTS_MODEL=canopylabs/orpheus-v1-english \
    MAX_CALL_SECONDS=150 DEMO_DAILY_SESSION_CAP=40 AGENT_IDLE_PROCESSES=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python -c "from voicedesk.rag import embed_query; embed_query('warm up')" \
    && python -m agent.main download-files

CMD ["bash", "deploy/livekit-agent/start.sh"]
