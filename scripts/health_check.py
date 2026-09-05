"""
Comprehensive Diagnostic and Health Check Script for JARVIS.
Verifies Docker, Ollama, Models, GPU, Audio Pipeline, and LiveKit settings.
"""

import os
import sys
import subprocess
import asyncio
import httpx
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import settings


def print_status(component: str, ok: bool, details: str = ""):
    symbol = "[OK]" if ok else "[FAIL]"
    color = "\033[92m" if ok else "\033[91m"
    reset = "\033[0m"
    try:
        print(f"{color}{symbol}{reset} {component.ljust(25)}: {details}")
    except UnicodeEncodeError:
        print(f"{symbol} {component.ljust(25)}: {details}")


async def check_docker() -> bool:
    try:
        res = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            print_status("Docker Engine", True, "Running")
            return True
        else:
            print_status("Docker Engine", False, "Docker is not running")
            return False
    except Exception as e:
        print_status("Docker Engine", False, f"Error checking Docker: {e}")
        return False


def check_gpu() -> bool:
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if res.returncode == 0:
            gpu_info = res.stdout.strip()
            print_status("NVIDIA GPU", True, gpu_info)
            return True
        else:
            print_status("NVIDIA GPU", False, "nvidia-smi returned non-zero (CPU fallback)")
            return False
    except Exception as e:
        print_status("NVIDIA GPU", False, "GPU not detected or nvidia-smi not in PATH")
        return False


async def check_ollama() -> bool:
    host = settings.ollama_host.rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            res = await client.get(f"{host}/api/version")
            if res.status_code == 200:
                ver = res.json().get("version", "unknown")
                print_status("Ollama API", True, f"Connected at {host} (version {ver})")
            else:
                print_status("Ollama API", False, f"Received HTTP {res.status_code} from {host}")
                return False

            # Check models
            models_res = await client.get(f"{host}/api/tags")
            if models_res.status_code == 200:
                models = [m["name"] for m in models_res.json().get("models", [])]
                target = settings.ollama_model
                has_target = any(target == m or target.split(":")[0] == m.split(":")[0] for m in models)
                if has_target:
                    print_status("Ollama Model", True, f"Target model '{target}' is available in Ollama")
                    return True
                else:
                    print_status(
                        "Ollama Model",
                        False,
                        f"Target model '{target}' not found. Available models: {models if models else 'None'}. Run: docker exec -it jarvis-ollama ollama pull {target}",
                    )
                    return False
            return True
    except Exception as e:
        print_status("Ollama API", False, f"Cannot connect to {host}. Is the container running? ({e})")
        return False


def check_tts() -> bool:
    try:
        from agent.voice.tts import LocalPiperTTS

        tts_engine = LocalPiperTTS()
        if tts_engine.voice:
            print_status("Piper TTS", True, f"Loaded voice '{settings.piper_voice}' ({tts_engine.sample_rate}Hz)")
            return True
        else:
            print_status("Piper TTS", False, "Piper voice failed to load")
            return False
    except Exception as e:
        print_status("Piper TTS", False, f"Error: {e}")
        return False


def check_stt() -> bool:
    try:
        from agent.voice.stt import LocalWhisperSTT

        stt_engine = LocalWhisperSTT()
        if stt_engine.whisper_model:
            print_status(
                "Whisper STT",
                True,
                f"Loaded faster-whisper '{settings.whisper_model_size}' on {settings.whisper_device}",
            )
            return True
        else:
            print_status("Whisper STT", False, "Whisper model failed to load")
            return False
    except Exception as e:
        print_status("Whisper STT", False, f"Error: {e}")
        return False


def check_livekit_config() -> bool:
    if settings.livekit_url and settings.livekit_api_key and settings.livekit_api_secret:
        print_status("LiveKit Credentials", True, f"Configured for URL: {settings.livekit_url}")
        return True
    else:
        print_status(
            "LiveKit Credentials",
            False,
            "LIVEKIT_URL, LIVEKIT_API_KEY, or LIVEKIT_API_SECRET missing in .env (Needed for voice connection)",
        )
        return False


def check_wakeword() -> bool:
    try:
        from agent.wakeword import OpenWakeWordDetector

        detector = OpenWakeWordDetector(
            wake_word=settings.wake_word,
            model_path=settings.wake_word_model_path if settings.wake_word_model_path else None,
            threshold=settings.wake_word_threshold,
        )
        if detector._model:
            print_status(
                "Wake Word Engine",
                True,
                f"openWakeWord ready for word '{settings.wake_word}' (threshold={settings.wake_word_threshold})",
            )
            return True
        else:
            print_status("Wake Word Engine", False, "Failed to load wake-word model")
            return False
    except Exception as e:
        print_status("Wake Word Engine", False, f"Error: {e}")
        return False


async def main():
    print("=" * 60)
    print(f" {settings.jarvis_name} System Health & Diagnostics (V2)")
    print("=" * 60)

    docker_ok = await check_docker()
    gpu_ok = check_gpu()
    ollama_ok = await check_ollama()
    wakeword_ok = check_wakeword()
    stt_ok = check_stt()
    tts_ok = check_tts()
    livekit_ok = check_livekit_config()

    print("=" * 60)
    if docker_ok and ollama_ok and wakeword_ok and stt_ok and tts_ok:
        print("\033[92m[OK] All core local V2 components are healthy and ready!\033[0m")
    else:
        print("\033[93m[!] Some components need attention before running full voice sessions.\033[0m")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
