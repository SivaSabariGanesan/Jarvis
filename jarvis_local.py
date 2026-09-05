"""
JARVIS Direct Local Voice Assistant.
Runs 100% locally on your machine with direct Microphone & Speaker support.
Pipeline: Local Mic -> Faster-Whisper STT -> Ollama (Llama 3.2) -> Piper TTS -> PC Speakers.
"""

import os
import sys
import time
import queue
import logging
import asyncio
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import httpx
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.config import settings
from agent.prompt import JARVIS_SYSTEM_PROMPT, INITIAL_GREETING
from agent.memory.database import JarvisDatabase
from agent.voice.tts import LocalPiperTTS
from faster_whisper import WhisperModel

# Logging setup
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("jarvis.local")


class JarvisLocalVoice:
    def __init__(self):
        print("\033[96m" + "=" * 60)
        print(f"  INITIALIZING {settings.jarvis_name} (100% DIRECT LOCAL VOICE)")
        print("=" * 60 + "\033[0m")

        self.db = JarvisDatabase()
        self.session_id = f"local-session-{int(time.time())}"
        self.db.create_session(session_id=self.session_id, room_name="local-pc")

        # 1. Faster-Whisper STT
        print(f"[1/3] Loading Faster-Whisper ({settings.whisper_model_size} on {settings.whisper_device})...")
        try:
            self.stt_model = WhisperModel(
                settings.whisper_model_size,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
            )
            print(f"\033[92m[OK] Whisper STT initialized on {settings.whisper_device}.\033[0m")
        except Exception as e:
            print(f"\033[93m[!] CUDA fallback to CPU int8: {e}\033[0m")
            self.stt_model = WhisperModel(settings.whisper_model_size, device="cpu", compute_type="int8")

        # 2. Piper TTS
        print(f"[2/3] Loading Piper TTS ({settings.piper_voice})...")
        self.tts = LocalPiperTTS()
        print(f"\033[92m[OK] Piper TTS voice ready ({self.tts.sample_rate}Hz).\033[0m")

        # 3. Ollama LLM check
        print(f"[3/3] Connecting to Ollama ({settings.ollama_model})...")
        self.ollama_host = settings.ollama_host.rstrip("/")
        self.messages = [{"role": "system", "content": JARVIS_SYSTEM_PROMPT}]

        # Audio recording params
        self.sample_rate = 16000
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.is_speaking = False

        print("\033[92m" + "=" * 60)
        print(f"  ALL SYSTEMS ONLINE. {settings.jarvis_name} IS READY.")
        print("=" * 60 + "\033[0m\n")

    def speak(self, text: str):
        """Synthesize text using Piper TTS and play through local speakers."""
        if not text or not text.strip():
            return
        
        self.is_speaking = True
        print(f"\n\033[94m{settings.jarvis_name}:\033[0m {text}")
        self.db.log_message(session_id=self.session_id, role="assistant", content=text)
        self.messages.append({"role": "assistant", "content": text})

        try:
            # Piper synthesis
            voice = self.tts.voice
            audio_bytes = b"".join(
                chunk.audio_int16_bytes for chunk in voice.synthesize(text) if hasattr(chunk, "audio_int16_bytes")
            )
            if audio_bytes:
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
                sd.play(audio_np, samplerate=self.tts.sample_rate)
                sd.wait()
        except Exception as e:
            logger.error(f"TTS Playback error: {e}")
        finally:
            self.is_speaking = False

    def ask_llm(self, user_text: str) -> str:
        """Send prompt to local Ollama and return assistant response."""
        self.messages.append({"role": "user", "content": user_text})
        self.db.log_message(session_id=self.session_id, role="user", content=user_text)

        payload = {
            "model": settings.ollama_model,
            "messages": self.messages,
            "stream": False,
            "options": {
                "temperature": 0.7,
            },
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                res = client.post(f"{self.ollama_host}/api/chat", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    reply = data.get("message", {}).get("content", "").strip()
                    return reply
                else:
                    return f"Error communicating with Ollama: HTTP {res.status_code}"
        except Exception as e:
            return f"I apologize, sir. I encountered an error with the local model: {e}"

    def record_until_silence(self, silence_threshold=0.015, max_seconds=15, silence_duration=1.5):
        """Record audio from default mic until silence is detected."""
        print("\033[93m🎙️  [Listening...] Speak now into your microphone...\033[0m", flush=True)

        recorded_chunks = []
        chunk_size = 1024
        silence_samples = int(self.sample_rate * silence_duration)
        consecutive_silent = 0
        has_speech = False

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32", blocksize=chunk_size) as stream:
            start_time = time.time()
            while time.time() - start_time < max_seconds:
                data, _ = stream.read(chunk_size)
                recorded_chunks.append(data.copy())
                amplitude = np.max(np.abs(data))

                if amplitude > silence_threshold:
                    has_speech = True
                    consecutive_silent = 0
                else:
                    if has_speech:
                        consecutive_silent += chunk_size
                        if consecutive_silent >= silence_samples:
                            break

        if not has_speech or len(recorded_chunks) == 0:
            return None

        audio_data = np.concatenate(recorded_chunks, axis=0).flatten()
        return audio_data

    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio numpy array with faster-whisper."""
        if audio_data is None or len(audio_data) < self.sample_rate * 0.5:
            return ""

        print("\033[90m⚙️  [Transcribing speech...]\033[0m", flush=True)
        segments, _ = self.stt_model.transcribe(audio_data, beam_size=5, vad_filter=True)
        text = " ".join(seg.text for seg in segments).strip()
        return text

    def run(self):
        """Main local voice interaction loop."""
        # Initial greeting
        self.speak(INITIAL_GREETING)

        print("\n\033[97m" + "-" * 60)
        print("  JARVIS DIRECT VOICE CONTROLS:")
        print("   - Press [ENTER] to speak via microphone")
        print("   - Type your message and press [ENTER] to type directly")
        print("   - Type 'exit' or 'quit' to stop")
        print("-" * 60 + "\033[0m\n")

        while True:
            try:
                user_input = input("\033[92m[Press ENTER to Speak / Type message]: \033[0m").strip()

                if user_input.lower() in ["exit", "quit", "q"]:
                    self.speak("Shutting down local systems. Goodbye, sir.")
                    break

                if user_input == "":
                    # Microphone Voice Mode
                    audio = self.record_until_silence()
                    if audio is None:
                        print("\033[90m(No speech detected, please try again)\033[0m\n")
                        continue

                    transcript = self.transcribe(audio)
                    if not transcript:
                        print("\033[90m(Could not understand audio, please try again)\033[0m\n")
                        continue

                    print(f"\033[93mYou (Voice):\033[0m {transcript}")
                    query = transcript
                else:
                    # Direct text mode
                    print(f"\033[93mYou (Text):\033[0m {user_input}")
                    query = user_input

                print("\033[90m⚙️  [Thinking...]\033[0m", flush=True)
                reply = self.ask_llm(query)
                self.speak(reply)
                print()

            except KeyboardInterrupt:
                print("\n\nSession terminated by user.")
                break
            except Exception as e:
                print(f"\033[91mError: {e}\033[0m")


def main():
    agent = JarvisLocalVoice()
    agent.run()


if __name__ == "__main__":
    main()
