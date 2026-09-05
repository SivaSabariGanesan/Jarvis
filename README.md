# JARVIS V3 — Local Personal Voice Agent & Secure Computer Control

A real-time, local personal AI voice agent and secure computer control assistant built with **LiveKit Agents**, **Ollama** (Docker with NVIDIA GPU acceleration), **openWakeWord** (Local "Jarvis" detection), **faster-whisper** (Local STT), **Piper TTS** (Local high-fidelity speech synthesis), **SQLite** memory persistence, and a **Deny-by-Default Security Tool Framework**.

---

## Architecture Overview (V3)

```text
                  [ User Voice / Text Input ]
                               │
                      ┌────────▼────────┐
                      │   Wake Word     │ ("Jarvis")
                      └────────┬────────┘
                               │
                      ┌────────▼────────┐
                      │ Faster-Whisper  │ (CPU STT)
                      └────────┬────────┘
                               │
                      ┌────────▼────────┐
                      │ Intent / LLM    │ (Ollama llama3.2:3b on GPU)
                      └────────┬────────┘
                               │
            ┌──────────────────▼──────────────────┐
            │       SECURITY VALIDATION LAYER     │
            │  • Deny-by-Default Execution        │
            │  • Path Sandboxing (JARVIS_WORKSPACE)│
            │  • Application & URL Allowlisting   │
            │  • Risk Tiering (LOW/MEDIUM/HIGH)   │
            │  • User Confirmation for HIGH Risk  │
            │  • Emergency Stop ("JARVIS STOP")   │
            │  • Structured Audit Logging         │
            └──────────────────┬──────────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
  [ Applications ]     [ System Info ]         [ Filesystem ]
  • open / close       • CPU / RAM / Disk      • create / read
  • status checks      • NVIDIA GPU status     • rename / move / copy
  • allowlist-only     • running apps          • delete (HIGH_RISK)
        │                      │                      │
        └──────────────────────┼──────────────────────┘
                               │
                      ┌────────▼────────┐
                      │    Piper TTS    │ (22050Hz)
                      └────────┬────────┘
                               │
                      [ Local Speakers ]
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

## 8. V2 — Local "Jarvis" Wake Word

JARVIS V2 operates as a **fully local, privacy-first voice assistant** with real-time wake-word detection for the keyword **"Jarvis"**.

### V2 Architecture Diagram

```text
                 🎤 MICROPHONE
                       │
                       ▼
              ┌─────────────────┐
              │ Local Wake Word │  (openWakeWord / ONNX on CPU)
              │   Detector      │
              └────────┬────────┘
                       │
                  "Jarvis"
                       │
                       ▼
                🟢 ACTIVATED (State: LISTENING)
                       │
                       ▼
              ┌─────────────────┐
              │      VAD        │  (Silero VAD on CPU)
              │  Silero VAD     │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Faster-Whisper  │  (Faster-Whisper on CPU int8)
              │      STT        │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  JARVIS AGENT   │  (State Machine: PROCESSING)
              │ Reasoning Engine│
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │     Ollama      │  (NVIDIA RTX 4050 GPU)
              │   Docker        │
              │ Llama 3.2 3B    │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │    Piper TTS    │  (Piper TTS on CPU)
              └────────┬────────┘
                       │
                       ▼
                   🔊 SPEAKER (State: SPEAKING)
                       │
                       ▼
                 Return to IDLE (Waiting for "Jarvis")
```

### Resource Allocation (GPU vs CPU)
To prevent GPU VRAM contention and eliminate CUDA Out-Of-Memory errors on 6GB laptop GPUs:
- **RTX 4050 GPU**: Dedicated 100% to **Ollama (`llama3.2:3b`)**.
- **CPU / RAM**: Runs **openWakeWord**, **Faster-Whisper (`int8`)**, **Silero VAD**, and **Piper TTS**.

### Finite State Machine
```text
  IDLE ──(detects "Jarvis")──► LISTENING ──(end of speech)──► PROCESSING ──(response ready)──► SPEAKING ──► IDLE
```
* **Self-Trigger Protection**: While in `SPEAKING` state, wake-word detection is automatically suppressed to prevent JARVIS from triggering on its own voice output.

### Configuration (`.env`)
```env
# Wake Word Engine (openWakeWord / ONNX)
WAKE_WORD=jarvis
WAKE_WORD_MODEL_PATH=
WAKE_WORD_THRESHOLD=0.5
WAKE_WORD_COOLDOWN=1.0
WAKE_WORD_ACTIVATION_RESPONSE="Yes, sir?"

# STT on CPU (reserves all VRAM for Ollama)
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_CPU_THREADS=4
```

### Running V2
1. **Standalone Direct Voice Runner (Mic + Speakers)**:
   ```bash
   .venv\Scripts\python.exe jarvis_local.py
   ```
2. **LiveKit Agent Worker**:
   ```bash
   .venv\Scripts\python.exe main.py dev
   ```

---

## 9. Troubleshooting & FAQ

| Issue | Cause | Solution |
| :--- | :--- | :--- |
| **`Cannot connect to Ollama at http://localhost:11434`** | Ollama container is not running | Run `docker compose up -d` and check `docker compose ps`. |
| **`Target model 'llama3.2:3b' not found`** | Model hasn't been pulled into Ollama volume | Run `docker exec -it jarvis-ollama ollama pull llama3.2:3b`. |
| **`CUDA out of memory in Ollama`** | Whisper STT or background tools competing for VRAM | Set `WHISPER_DEVICE=cpu` and `WHISPER_COMPUTE_TYPE=int8` in `.env` so GPU is 100% reserved for Ollama. |
| **`LiveKit connection timeout`** | Invalid `LIVEKIT_URL` or missing API keys | Verify `.env` credentials against your LiveKit project dashboard. |
| **`Wake word not triggering`** | Background noise or low mic input | Lower `WAKE_WORD_THRESHOLD=0.4` in `.env`. |

---

## 10. Future Milestones (Roadmap)

- **Milestone 3**: Controlled computer tools (App launching, file search, web retrieval).
- **Milestone 4**: Vision & multimodal desktop understanding.

