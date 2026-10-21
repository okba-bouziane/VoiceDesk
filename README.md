# VoiceDesk

A voice receptionist for a fictional clinic. A caller talks to it in the browser; it books, lists and cancels appointments, answers questions about the clinic from a knowledge base, and hands over to a human when asked or when the situation sounds urgent.

> **All clinic data is sample data.** The clinic, doctors, prices, policies and phone numbers are invented. Nothing here is medical advice.

I built it to show a production-shaped voice-agent stack end to end, not a notebook demo:

- **LiveKit Agents (Python 1.x)** with Silero VAD, a turn-detection model, interruptions and a swappable STT, LLM, TTS pipeline.
- **LangGraph** conversation logic (intent routing, slot filling, RAG answer, handoff) plugged into LiveKit through its LangGraph adapter.
- A separate **gRPC booking microservice** (Protobuf contract, Postgres, validation, status codes, health check, reflection).
- **Hybrid RAG** on Postgres + **pgvector**: local `BAAI/bge-small-en-v1.5` embeddings (fastembed), full-text search, fused with reciprocal rank fusion.
- **Structured outputs** validated with Pydantic, with retry on invalid model output.
- **MLOps basics**: prompts and model choices versioned in one YAML file, per-turn latency metrics (JSON logs and Prometheus), an eval gate in CI.

![Web client talking to the agent](docs/screenshots/web-client.png)

## Architecture

```mermaid
flowchart LR
    subgraph Browser
        WEB["Web client<br/>livekit-client JS"]
    end
    subgraph Compose["docker compose"]
        LK["livekit-server<br/>(--dev, self-hosted)"]
        API["FastAPI server<br/>tokens, static client,<br/>/healthz, /metrics"]
        AG["Agent worker<br/>LiveKit Agents"]
        BK["Booking service<br/>gRPC + Protobuf"]
        PG[("Postgres 16 + pgvector<br/>appointments, doc_chunks")]
    end
    subgraph Agent["inside the agent"]
        VAD["Silero VAD +<br/>turn detector"] --> STT["STT"] --> G["LangGraph graph"] --> TTS["TTS"]
        G -->|"tool: hybrid search"| RAG["fastembed + pgvector"]
        G -->|"tools: check / book /<br/>cancel / list"| BKC["gRPC client"]
    end
    WEB -- "POST /api/token" --> API
    WEB <-- "WebRTC audio, transcript,<br/>agent state" --> LK
    LK <--> AG
    AG --- Agent
    RAG --> PG
    BKC --> BK --> PG
    LLM["LLM / STT / TTS provider<br/>(Groq by default)"] <--> Agent
```

### One voice turn

```mermaid
sequenceDiagram
    participant U as Caller (browser)
    participant L as livekit-server
    participant A as Agent (VAD, turn detector)
    participant S as STT
    participant G as LangGraph
    participant M as LLM
    participant R as pgvector (RAG)
    participant B as Booking gRPC
    participant T as TTS
    U->>L: microphone audio
    L->>A: audio frames
    A->>A: VAD: speech start / end, turn detector: is the turn over?
    A->>S: utterance audio
    S-->>A: transcript
    A->>G: chat context (last user message)
    G->>M: route intent (JSON, validated, retried if invalid)
    alt clinic question
        G->>R: embed question, hybrid search (vector + full text, RRF)
        R-->>G: top 3 chunks
        G->>M: answer from context only (streamed)
    else booking, cancel or list
        G->>M: extract slots (JSON, validated)
        G->>B: CheckAvailability / BookAppointment / CancelAppointment / ListAppointments
        B-->>G: result or gRPC status
    end
    G-->>A: reply tokens (custom stream)
    A->>T: sentences as they complete
    T-->>L: audio
    L-->>U: speech + transcript
    Note over U,A: if the caller talks over the agent, VAD interrupts playback and the graph is cancelled
```

## How the conversation graph works

`voicedesk/graph.py` is one LangGraph `StateGraph` compiled with a checkpointer keyed by room, so slots survive across turns.

| Node | Job |
|---|---|
| `route` | Structured intent classification (`IntentResult`), aware of the active flow and of a pending yes/no. |
| `booking` | Extract slots (`SlotUpdate`), merge into `BookingSlots`, call `CheckAvailability` once specialty and date are known, ask for what is missing, read back for confirmation. |
| `book` | After a yes: `BookAppointment`. A taken slot or a closed day is explained and only the affected slot is asked again. |
| `manage` / `do_cancel` | Look up appointments by phone, pick one, confirm, cancel. Also answers "what are my appointments". |
| `faq` | Hybrid search, a relevance gate, then an answer generated from the retrieved context only, streamed token by token. Off-topic or unknown questions are refused with an offer of a human. |
| `handoff` | Staff handoff flag plus an emergency notice. |
| `decline_flow`, `smalltalk` | Drop the current flow, greet. |

The routing function `decide(state, intent)` is pure and table-tested. Every leaf node builds an `AgentAction` Pydantic model (kind, speech, sources, handoff flag), which is validated before it is spoken. Any exception inside a node (provider outage, rate limit) becomes a spoken apology, so the session survives.

The LiveKit side is thin (`agent/`): `LLMAdapter(graph, stream_mode="custom")` feeds the chat context to the graph and streams its replies to TTS. The graph's own prompts are used; the LiveKit tool-calling mechanism is not (tool calls happen inside the graph).

## Booking microservice

`proto/voicedesk/booking/v1/booking.proto` defines `CheckAvailability`, `BookAppointment`, `CancelAppointment`, `ListAppointments`. Stubs are built by `grpcio-tools` into `voicedesk/booking/v1/` and CI fails if they are stale.

| Situation | Status |
|---|---|
| Bad date, past date, closed day (Fri, Sat), time off the 30-minute grid or in the lunch break, bad phone or name | `INVALID_ARGUMENT` |
| Unknown specialty, provider or appointment | `NOT_FOUND` |
| Slot taken (also under a race, enforced by a partial unique index) | `ALREADY_EXISTS` |
| Cancelling with a phone number that does not own the appointment | `PERMISSION_DENIED` |
| Cancelling twice or cancelling a past appointment | `FAILED_PRECONDITION` |
| Anything unexpected | `INTERNAL` (details logged, not leaked) |

It also serves the standard `grpc.health.v1.Health` service and gRPC reflection, so `grpcurl -plaintext localhost:50051 list` works.

## Evaluation

All numbers below are from runs in this repository. Nothing is estimated.

**Retrieval** (`data/eval/rag_questions.json`: 32 labelled questions plus 6 out-of-scope ones; 32 chunks from 6 sample documents; local bge-small embeddings):

| Search | recall@3 | MRR |
|---|---|---|
| hybrid (vector + full text, RRF) | 1.000 | 0.932 |
| vector only | 1.000 | 0.953 |
| full text only | 0.969 | 0.896 |

Hybrid recall@1 is 0.875. The "answer or abstain" gate classifies 35 of 38 questions correctly (0.921). On this small, clean corpus plain vector search is as good as hybrid (slightly better MRR); I report that as it is. Hybrid is kept because keyword matches (names, "CHIFA", prices) are where embeddings are weakest on larger corpora, but this set does not prove that.

**Live model checks** (`python -m scripts.eval`, Groq free tier; 28 intent utterances, 6 extraction cases, 15 slot fields):

| Model | Intent accuracy | Slot field accuracy | Valid JSON first try | Valid after retry |
|---|---|---|---|---|
| `openai/gpt-oss-120b` (run before the 14-day calendar was added to the extraction prompt) | 0.964 | 1.000 | 1.000 | 1.000 |
| `openai/gpt-oss-20b` (current `docs/eval-results.json`) | 0.929 | 1.000 | 1.000 | 1.000 |

The sets are small, so treat these as smoke-level numbers, not benchmarks. An earlier 20b run resolved "next Sunday" to the wrong date, which is why the extraction prompt now carries a 14-day calendar.

**Tests** (`pytest`): 74 tests, 71 run offline, 3 live ones need an LLM key.

| File | What it covers |
|---|---|
| `test_booking_service.py` | In-process gRPC server on real Postgres: availability, booking, double booking, validation status codes, cancel rules, listing, health check, reflection. |
| `test_rag.py` | Chunking, RRF fusion, recall@3, MRR and abstain thresholds for hybrid, sanity floor for the baselines. |
| `test_structured.py` | Structured output validity, fenced or wrapped JSON, retry with the validation error shown, retry budget, slot and action validation. |
| `test_graph.py` | Routing table, full booking with slot filling, slot taken, closed day, decline, cancel, list, grounded FAQ with token streaming, refusal, handoff, provider outage. Uses a scripted model (a test double that exists only in `tests/fakes.py`). |
| `test_agent_text.py` | LiveKit `AgentSession` text-mode runs (`session.run(user_input=...)`) driving the real adapter and graph. |
| `test_live.py` | Same style, real model and LiveKit's LLM judge (`.judge(...)`). Skipped without a key. |
| `test_server.py`, `test_metrics.py` | Token endpoint and metrics, per-turn latency aggregation. |

Thresholds live in `config/voicedesk.yaml` (`eval_gate`) and are enforced by pytest and by `scripts/eval.py`, which CI runs.

![pytest and eval output](docs/screenshots/eval-output.png)

## Latency

Every turn is logged as one JSON line (`turn_latency {...}`), published to the web client over the data channel, and exported as the Prometheus histogram `voicedesk_turn_stage_seconds{stage,prompt_version}` (stages: `stt`, `endpointing`, `llm_ttft`, `tts_ttfb`, `end_to_end`) on the agent's `:9464/metrics`.

![Prometheus metrics](docs/screenshots/metrics.png)

Budget for the time from the caller stopping to the first audio of the answer, and what I measured:

| Stage | Target | Measured here |
|---|---|---|
| Endpointing (VAD plus turn detector wait) | 0.5 s min delay (configurable) | not measured (needs live speech) |
| STT, Groq `whisper-large-v3-turbo` | < 0.5 s | 0.33, 0.38, 0.60 s on three short synthetic utterances (macOS `say`) |
| Graph and LLM, first token | < 1 s | `gpt-oss-120b`: 1.1, 1.8, 2.6 s (3 turns, route plus extract or answer are sequential LLM calls). `gpt-oss-20b` through the full stack in text mode: mean 0.82 s over 4 turns |
| TTS first audio | < 0.4 s | **not measured**: Groq's TTS model requires the organisation admin to accept its terms in the Groq console, which has not been done |
| End to end | about 2 s | **not measured**: needs TTS and a live microphone |

The honest summary: STT plus first LLM token was about 1.4 to 3.2 s on `gpt-oss-120b`. That is above a comfortable voice budget, mostly because each turn makes two sequential model calls (route, then extract or answer) on a reasoning model. Obvious next steps are listed under limitations.

## Run it

Prerequisites: Docker, and a free [Groq](https://console.groq.com) key for the default providers.

```bash
cp .env.example .env            # put GROQ_API_KEY in it
docker compose up --build       # first build downloads the models into the image
```

Open http://localhost:8000, press **Start call**, allow the microphone and talk. You can also type in the box under the transcript.

For a text-only demo without TTS (the agent writes its replies instead of speaking them) set `TTS_PROVIDER=none`.

Services and host ports: Postgres `5434`, booking gRPC `50051`, FastAPI and web client `8000`, LiveKit `7880`, `7881`, `7882/udp`, agent metrics `9464`.

Local development:

```bash
/opt/homebrew/bin/python3.12 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
docker compose up -d db                      # Postgres + pgvector on :5434
python -m scripts.init_data                  # schema, sample providers, index the clinic documents
python -m booking_service &                  # gRPC service
python -m agent.main download-files          # Silero + turn detector
python -m agent.main dev                     # agent worker (needs livekit-server from compose)
python -m scripts.chat "what are your opening hours?"   # talk to the graph in the terminal
pytest -q && python -m scripts.eval
scripts/gen_proto.sh                         # regenerate stubs after editing the proto
```

`scripts/voice_roundtrip.py` runs synthetic caller audio through STT, the graph and TTS and prints each stage's latency without a microphone. `scripts/browser_demo.js` drives the web client in headless Chrome (used for the screenshots).

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | | One key for Whisper STT, the LLM and Groq TTS |
| `LLM_PROVIDER` | `groq` | `groq`, `openai`, `ollama` (any OpenAI-compatible endpoint) |
| `STT_PROVIDER` | `groq` | `groq`, `openai`, `deepgram` |
| `TTS_PROVIDER` | `groq` | `groq`, `openai`, `cartesia`, `none` (text-only) |
| `LLM_MODEL`, `STT_MODEL`, `TTS_MODEL`, `TTS_VOICE` | from YAML | Override the model names in `config/voicedesk.yaml` |
| `LLM_BASE_URL`, `LLM_API_KEY` | from YAML | Point at any OpenAI-compatible server |
| `OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, `CARTESIA_API_KEY` | | Keys for the swappable providers |
| `TURN_DETECTION` | `model` | `model` (LiveKit turn-detector model) or `vad` |
| `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | `ws://localhost:7880`, `devkey`, `secret` | LiveKit dev server credentials (dev defaults only) |
| `DATABASE_URL` | `postgresql://voicedesk:voicedesk@localhost:5434/voicedesk` | Postgres |
| `BOOKING_GRPC_ADDR` | `localhost:50051` | Booking service |

Prompts, default models per provider, RAG parameters, endpointing delays and eval thresholds are in `config/voicedesk.yaml`; bump its `version` when you change a prompt (it is attached to every metric).

## Project layout

```
agent/            LiveKit worker: session setup, providers, per-turn metrics
booking_service/  gRPC server, Postgres store, validation
voicedesk/        config, structured schemas, LLM client, RAG, LangGraph graph, gRPC client
server/           FastAPI token server, /healthz, /metrics, web client (static)
proto/            Protobuf contract
config/           versioned prompts, models and thresholds
data/             sample clinic documents and labelled eval sets
scripts/          init_data, eval, chat, voice_roundtrip, proto generation, screenshots
tests/            pytest suite
```

## Limitations

- **Voice latency with TTS and live speech is not measured.** The Groq TTS model needs terms accepted by the org admin; until then I only ran STT, graph and text-mode sessions end to end. The agent has been run in a real LiveKit room from a headless browser in text mode, not with spoken audio.
- Two sequential LLM calls per turn on a reasoning model are slow for voice. Likely fixes: run routing and extraction concurrently, use a small non-reasoning model for them, and cache the routing prompt.
- Dates and times are resolved by the model with a calendar hint; the gRPC service rejects impossible values, but a wrong-but-valid date would be booked after the caller confirms the read-back. The read-back is the safeguard.
- Appointments are identified by phone number only (no real authentication), and a booking is a sample record, not a real system of record.
- No barge-in tuning, background noise handling or telephony (SIP) integration. The adaptive interruption model is disabled by default by LiveKit in production mode; VAD-based interruption is used.
- The evaluation sets are small (32 retrieval questions, 28 intents, 6 extraction cases), so differences of a few points are noise. The corpus is 32 chunks; retrieval quality on a real clinic knowledge base is unknown.
- The Groq free tier limits tokens (8,000 per minute and 200,000 per day on the 120B model, hit during development), so heavy eval runs may need a paid tier or another provider.
- `livekit-server --dev` uses public development keys. Do not expose this stack to the internet as is.

## License

MIT, see `LICENSE`.
