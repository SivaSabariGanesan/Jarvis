# JARVIS — Local Personal AI Voice Agent

A real-time, local personal AI voice agent built with **LiveKit Agents**, **Ollama** (Docker with NVIDIA GPU acceleration), **faster-whisper** (Local STT), **Piper TTS** (Local high-fidelity speech synthesis), and **SQLite** memory persistence.

---

## Architecture Overview

```text
[ Microphone ]
       │
       ▼
 [ LiveKit Room ]
       │
       ▼ (Audio Stream)
[ Silero VAD + Local Whisper STT ] ──► (Transcribed Text)
                                               │
                                               ▼
                                   [ Local Ollama LLM ]
                                   (llama3.2:3b / qwen2.5:3b)
                                               │
                                               ▼ (Response Tokens)
 [ Speaker ] ◄── [ LiveKit Audio ] ◄── [ Local Piper TTS ]
```

---

## 1. System Requirements

* **Operating System**: Windows 10/11 (WSL2 / Docker Desktop), Linux (Ubuntu 20.04+), macOS (Apple Silicon).
* **Python**: Python `3.11` (recommended, managed automatically via `uv`).
* **Package Manager**: [`uv`](https://github.com/astral-sh/uv) (fast Python dependency manager).
* **Container Engine**: Docker Desktop with Docker Compose v2+.
* **GPU (Optional but Recommended)**: NVIDIA GPU with CUDA support (e.g. RTX 3050 / 4050 or higher).
  * Minimum 4GB-6GB VRAM for 3B models (`llama3.2:3b` ~2.0 GB VRAM, `base.en` Whisper ~0.3 GB VRAM).
  * System will gracefully fall back to CPU int8 inference if CUDA is unavailable.
* **Storage**:
  * Ollama Docker Image: ~3.5 GB
  * LLM Model (`llama3.2:3b`): ~2.0 GB
  * STT Model (`faster-whisper base.en`): ~150 MB
  * TTS Model (`Piper en_US-lessac-medium`): ~60 MB

---

## 2. Project Directory Structure

```text
jarvis/
├── agent/
│   ├── __init__.py
│   ├── main.py              # LiveKit CLI worker entrypoint
│   ├── agent.py             # Voice agent logic & session orchestrator
│   ├── config.py            # Typed settings & environment validation
│   ├── prompt.py            # JARVIS personality & operational guidelines
│   ├── ai/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract LLMClient interface
│   │   └── ollama_client.py # Ollama LLM provider & LiveKit adapter
│   ├── voice/
│   │   ├── __init__.py
│   │   ├── stt.py           # Local Whisper/faster-whisper STT StreamAdapter
│   │   └── tts.py           # Local Piper TTS ChunkedStream synthesizer
│   ├── memory/
│   │   ├── __init__.py
│   │   └── database.py      # SQLite conversation & session store
│   └── tools/
│       ├── __init__.py
│       └── registry.py      # Controlled tool registry interface
├── docker-compose.yml       # Ollama container with GPU passthrough & persistence
├── scripts/
│   └── health_check.py      # Diagnostic health check tool
├── data/                    # Local models, SQLite db, audio cache
├── .env.example             # Environment configuration template
├── .env                     # Local settings (git-ignored)
├── .gitignore
├── pyproject.toml
└── README.md
```

---

## 3. Installation & Setup

### Step 1: Install Dependencies with `uv`

Ensure `uv` is installed on your system. From the project root:

```bash
uv sync
```

This will automatically create a virtual environment (`.venv`) and install all required packages:
* `livekit-agents`
* `livekit-plugins-silero` (VAD)
* `livekit-plugins-openai` (Ollama provider)
* `faster-whisper` (Local STT)
* `piper-tts` (Local TTS)
* `pydantic-settings`, `httpx`, `numpy`, `soundfile`

---

## 4. Ollama Docker Setup & Model Download

### Step 1: Start Ollama in Docker

Run Docker Compose to start the Ollama container with persistent model volume and GPU acceleration:

```bash
docker compose up -d
```

Verify container status:

```bash
docker compose ps
```

### Step 2: Download the Local LLM Model

Download the default `llama3.2:3b` model (~2.0 GB):

```bash
docker exec -it jarvis-ollama ollama pull llama3.2:3b
```

*(Alternative recommended models: `qwen2.5:3b`, `mistral:7b-instruct-q4_k_m`, `phi3:mini`)*

---

## 5. Environment Configuration

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Configure the following variables in `.env`:

```env
# Assistant Identity
JARVIS_NAME=JARVIS
JARVIS_USER_TITLE=sir

# Ollama / Local LLM
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2:3b

# LiveKit Server Connection
LIVEKIT_URL=wss://<your-project>.livekit.cloud
LIVEKIT_API_KEY=<your-api-key>
LIVEKIT_API_SECRET=<your-api-secret>

# Local Voice Settings
WHISPER_MODEL_SIZE=base.en
WHISPER_DEVICE=cuda
PIPER_VOICE=en_US-lessac-medium
SQLITE_DB_PATH=data/jarvis_memory.db
```

---

## 6. LiveKit Server Configuration

You can connect JARVIS to either:

### Option A: Free LiveKit Cloud (Recommended for easiest testing)
1. Sign up at [cloud.livekit.io](https://cloud.livekit.io) (generous free tier).
2. Create a new project.
3. In Project Settings, copy your **Websocket URL**, **API Key**, and **API Secret** into `.env`.
4. Use the hosted web playground at [agents-playground.livekit.io](https://agents-playground.livekit.io).

### Option B: Local LiveKit Server
If you prefer 100% offline self-hosted infrastructure:
```bash
docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp livekit/livekit-server --dev
```
Then set in `.env`:
```env
LIVEKIT_URL=ws://127.0.0.1:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
```

---

## 7. Diagnostics & Health Check

Run the built-in diagnostic utility to verify all components (Docker, Ollama API, Model, GPU, Whisper STT, Piper TTS, LiveKit credentials):

```bash
uv run python scripts/health_check.py
```

Expected output:
```text
============================================================
 JARVIS System Health & Diagnostics
============================================================
[✓] Docker Engine            : Running
[✓] NVIDIA GPU               : NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB
[✓] Ollama API               : Connected at http://localhost:11434
[✓] Ollama Model             : Target model 'llama3.2:3b' is available in Ollama
[✓] Whisper STT              : Loaded faster-whisper 'base.en' on cuda
[✓] Piper TTS                : Loaded voice 'en_US-lessac-medium' (22050Hz)
[✓] LiveKit Credentials      : Configured for URL: wss://...
============================================================
[✓] All core local components are healthy and ready!
============================================================
```

---

## 8. Running JARVIS

Start the JARVIS agent worker in development mode (with auto-reload):

```bash
uv run python -m agent.main dev
```

### Initial Voice Interaction Test

1. Open [https://agents-playground.livekit.io](https://agents-playground.livekit.io) (or your local frontend).
2. Connect to the room using your LiveKit Cloud or local dev credentials.
3. JARVIS will speak upon connection:
   > *"Hello, sir. I'm ready. I can currently listen, understand your requests, reason using the local AI model, and respond through voice."*
4. Speak into your microphone:
   > *"Hey Jarvis, what can you do?"*
5. JARVIS will transcribe your voice locally, reason with `llama3.2:3b`, synthesize the response with Piper TTS, and speak back in real-time.

---

## 9. Troubleshooting & FAQ

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **`Cannot connect to Ollama at http://localhost:11434`** | Ollama container is not running | Run `docker compose up -d` and check `docker compose ps`. |
| **`Target model 'llama3.2:3b' not found`** | Model hasn't been pulled into Ollama volume | Run `docker exec -it jarvis-ollama ollama pull llama3.2:3b`. |
| **`CUDA out of memory`** | Large model loaded concurrently with STT/TTS | Use a 3B model (`llama3.2:3b` or `qwen2.5:3b`) and `base.en` Whisper model. |
| **`LiveKit connection timeout`** | Invalid `LIVEKIT_URL` or missing API keys | Verify `.env` credentials against your LiveKit project dashboard. |
| **`Whisper STT loading error on CUDA`** | Missing cuDNN or CUDA libraries in PATH | The agent will automatically fall back to CPU int8 inference. Set `WHISPER_DEVICE=cpu` in `.env` if desired. |

---

## 10. Future Milestones (Roadmap)

- **Milestone 2**: Memory expansion (Vector DB + SQLite embeddings).
- **Milestone 3**: Controlled computer tools (App launching, file search, web retrieval).
- **Milestone 4**: Vision & multimodal desktop understanding.
