"""
JARVIS V2 Standalone Local Voice Assistant.
100% Local Execution with continuous hands-free "Jarvis" Wake-Word Detection.
Pipeline: Microphone -> OpenWakeWord ("Jarvis") -> Faster-Whisper -> Ollama (Llama 3.2) -> Piper TTS -> Speakers.
"""

import os
import sys
import time
import queue
import logging
import threading
from typing import Optional, List, Dict
import numpy as np
import sounddevice as sd
import httpx
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agent.config import settings
from agent.prompt import JARVIS_SYSTEM_PROMPT, INITIAL_GREETING
from agent.memory.database import JarvisDatabase
from agent.voice.tts import LocalPiperTTS
from agent.wakeword import AgentState, StateMachine, OpenWakeWordDetector
from faster_whisper import WhisperModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jarvis.local")


class JarvisLocalVoiceV2:
    """
    JARVIS V2 Standalone Voice Assistant with continuous local wake-word detection.
    """

    def __init__(self):
        print("\033[96m" + "=" * 65)
        print(f"  INITIALIZING {settings.jarvis_name} V2 — LOCAL WAKE WORD ('{settings.wake_word.upper()}')")
        print("=" * 65 + "\033[0m")

        self.state_machine = StateMachine(AgentState.IDLE)
        self.db = JarvisDatabase()
        self.session_id = f"local-v2-{int(time.time())}"
        self.db.create_session(session_id=self.session_id, room_name="local-pc")

        # 1. Wake Word Detector
        print(f"[1/4] Initializing openWakeWord Detector for '{settings.wake_word}'...")
        self.wakeword_detector = OpenWakeWordDetector(
            wake_word=settings.wake_word,
            model_path=settings.wake_word_model_path if settings.wake_word_model_path else None,
            threshold=settings.wake_word_threshold,
            cooldown_seconds=settings.wake_word_cooldown,
        )
        print(f"\033[92m[OK] Wake-word engine initialized (Threshold: {settings.wake_word_threshold}).\033[0m")

        # 2. Faster-Whisper STT
        print(f"[2/4] Loading Faster-Whisper STT ({settings.whisper_model_size} on {settings.whisper_device})...")
        try:
            self.stt_model = WhisperModel(
                settings.whisper_model_size,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
            )
            print(f"\033[92m[OK] Faster-Whisper loaded on {settings.whisper_device}.\033[0m")
        except Exception as e:
            print(f"\033[93m[!] GPU fallback to CPU int8: {e}\033[0m")
            self.stt_model = WhisperModel(settings.whisper_model_size, device="cpu", compute_type="int8")

        # 3. Piper TTS
        print(f"[3/4] Loading Piper TTS ({settings.piper_voice})...")
        self.tts = LocalPiperTTS()
        print(f"\033[92m[OK] Piper TTS voice ready ({self.tts.sample_rate}Hz).\033[0m")

        # 4. Ollama LLM
        print(f"[4/4] Connecting to Ollama LLM ({settings.ollama_model})...")
        self.ollama_host = settings.ollama_host.rstrip("/")
        self.messages = [{"role": "system", "content": JARVIS_SYSTEM_PROMPT}]
        print(f"\033[92m[OK] Ollama ready at {self.ollama_host}.\033[0m")

        self.sample_rate = 16000
        self.chunk_size = 1280  # 80ms chunks for wake-word engine
        self._running = False

        print("\033[92m" + "=" * 65)
        print(f"  [INFO] JARVIS V2 ONLINE. ALL SYSTEMS OPERATIONAL.")
        print("=" * 65 + "\033[0m\n")

    def speak(self, text: str, suppress_wakeword: bool = True):
        """Synthesize text and play through local speakers with self-trigger protection."""
        if not text or not text.strip():
            return

        if suppress_wakeword:
            self.wakeword_detector.set_suppressed(True)
            self.state_machine.transition_to(AgentState.SPEAKING)

        print(f"\n\033[94m{settings.jarvis_name}:\033[0m {text}")
        self.db.log_message(session_id=self.session_id, role="assistant", content=text)
        self.messages.append({"role": "assistant", "content": text})

        try:
            voice = self.tts.voice
            audio_bytes = b"".join(
                chunk.audio_int16_bytes for chunk in voice.synthesize(text) if hasattr(chunk, "audio_int16_bytes")
            )
            if audio_bytes:
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
                sd.play(audio_np, samplerate=self.tts.sample_rate)
                sd.wait()
        except Exception as e:
            logger.error(f"TTS audio playback error: {e}")
        finally:
            if suppress_wakeword:
                self.state_machine.transition_to(AgentState.IDLE)
                self.wakeword_detector.reset()
                self.wakeword_detector.set_suppressed(False)

    def ask_llm(self, user_text: str) -> str:
        """Query local Ollama with user command."""
        self.messages.append({"role": "user", "content": user_text})
        self.db.log_message(session_id=self.session_id, role="user", content=user_text)

        payload = {
            "model": settings.ollama_model,
            "messages": self.messages,
            "stream": False,
            "options": {"temperature": 0.7},
        }

        try:
            with httpx.Client(timeout=60.0) as client:
                res = client.post(f"{self.ollama_host}/api/chat", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    return data.get("message", {}).get("content", "").strip()
                else:
                    return f"Error connecting to Ollama: HTTP {res.status_code}"
        except Exception as e:
            return f"I apologize, sir. An error occurred with the local reasoning model: {e}"

    def record_command(self, silence_threshold=0.015, max_seconds=12, silence_duration=1.3) -> Optional[np.ndarray]:
        """Record user command after wake word until end-of-speech silence is detected."""
        print("\033[93m🎙️  [LISTENING...] Say your command...\033[0m", flush=True)

        recorded_chunks = []
        block_size = 1024
        silence_samples = int(self.sample_rate * silence_duration)
        consecutive_silent = 0
        has_speech = False

        with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype="float32", blocksize=block_size) as stream:
            start_time = time.time()
            while time.time() - start_time < max_seconds:
                data, _ = stream.read(block_size)
                recorded_chunks.append(data.copy())
                amplitude = np.max(np.abs(data))

                if amplitude > silence_threshold:
                    has_speech = True
                    consecutive_silent = 0
                else:
                    if has_speech:
                        consecutive_silent += block_size
                        if consecutive_silent >= silence_samples:
                            break

        if not has_speech or len(recorded_chunks) == 0:
            return None

        return np.concatenate(recorded_chunks, axis=0).flatten()

    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio using local Faster-Whisper."""
        if audio_data is None or len(audio_data) < self.sample_rate * 0.4:
            return ""

        print("\033[90m⚙️  [PROCESSING: Transcribing speech...]\033[0m", flush=True)
        segments, _ = self.stt_model.transcribe(audio_data, beam_size=5, vad_filter=True)
        return " ".join(seg.text for seg in segments).strip()

    def run(self):
        """Continuous main loop with hands-free wake word detection."""
        self._running = True
        self.state_machine.transition_to(AgentState.IDLE)

        # Initial startup greeting
        self.speak(INITIAL_GREETING)

        print("\033[97m" + "-" * 65)
        print("  JARVIS V2 CONTROLS:")
        print("   • Say 'Jarvis' hands-free to activate")
        print("   • Press Ctrl+C in terminal to exit")
        print("-" * 65 + "\033[0m\n")

        print(f"\033[90m[INFO] State: {self.state_machine.current_state.value}\033[0m")
        print(f"\033[92m[INFO] Waiting for wake word '{settings.wake_word}'...\033[0m\n")

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.chunk_size,
        ) as mic_stream:

            while self._running:
                try:
                    # 1. State: IDLE — Read mic chunk for wake-word detection
                    if self.state_machine.is_state(AgentState.IDLE):
                        audio_chunk, _ = mic_stream.read(self.chunk_size)
                        detected = self.wakeword_detector.process_frame(audio_chunk)

                        if detected:
                            print("\n\033[96m" + "*" * 50)
                            print(f"  [INFO] Wake word '{settings.wake_word}' detected!")
                            print("*" * 50 + "\033[0m")

                            # 2. State: LISTENING
                            self.state_machine.transition_to(AgentState.LISTENING)
                            print(f"\033[90m[INFO] State: {self.state_machine.current_state.value}\033[0m")

                            # Spoken response: "Yes, sir?"
                            activation_phrase = settings.wake_word_activation_response.strip('"')
                            self.speak(activation_phrase, suppress_wakeword=False)

                            # 3. Capture user voice command
                            audio_command = self.record_command()
                            if audio_command is None:
                                print("\033[90m[INFO] No command detected. Returning to IDLE.\033[0m\n")
                                self.state_machine.transition_to(AgentState.IDLE)
                                print(f"\033[92m[INFO] Waiting for wake word '{settings.wake_word}'...\033[0m\n")
                                continue

                            # 4. State: PROCESSING
                            self.state_machine.transition_to(AgentState.PROCESSING)
                            print(f"\033[90m[INFO] State: {self.state_machine.current_state.value}\033[0m")

                            user_text = self.transcribe(audio_command)
                            if not user_text:
                                print("\033[90m[INFO] Could not understand speech. Returning to IDLE.\033[0m\n")
                                self.state_machine.transition_to(AgentState.IDLE)
                                print(f"\033[92m[INFO] Waiting for wake word '{settings.wake_word}'...\033[0m\n")
                                continue

                            print(f"\033[93mUser:\033[0m {user_text}")

                            # Query LLM
                            print("\033[90m[INFO] Sending request to Ollama...\033[0m", flush=True)
                            response_text = self.ask_llm(user_text)
                            print("\033[90m[INFO] Response generated.\033[0m")

                            # 5. State: SPEAKING (speak handles SPEAKING -> IDLE transition)
                            print(f"\033[90m[INFO] State: {AgentState.SPEAKING.value}\033[0m")
                            self.speak(response_text, suppress_wakeword=True)
                            print(f"\033[90m[INFO] State: {self.state_machine.current_state.value}\033[0m")
                            print(f"\033[92m[INFO] Waiting for wake word '{settings.wake_word}'...\033[0m\n")

                    else:
                        time.sleep(0.05)

                except KeyboardInterrupt:
                    print("\n[INFO] Stopping JARVIS voice assistant...")
                    self._running = False
                    break
                except Exception as e:
                    logger.error(f"Voice loop error: {e}")
                    self.state_machine.transition_to(AgentState.IDLE)
                    time.sleep(0.1)


def main():
    agent = JarvisLocalVoiceV2()
    agent.run()


if __name__ == "__main__":
    main()
