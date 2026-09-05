import os

# Limit BLAS/OpenMP thread allocation to prevent Windows memory pool exhaustion
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("jarvis.local")


class JarvisLocalVoiceV2:
    """
    JARVIS V2 Standalone Voice Assistant supporting both:
    1. Wake-Word voice detection ("Jarvis").
    2. Direct keyboard text input.
    """

    def __init__(self):
        print("\033[96m" + "=" * 65)
        print(f"  INITIALIZING {settings.jarvis_name} V2 — LOCAL VOICE & TEXT ASSISTANT")
        print("=" * 65 + "\033[0m", flush=True)

        self.state_machine = StateMachine(AgentState.IDLE)
        self.db = JarvisDatabase()
        self.session_id = f"local-v2-{int(time.time())}"
        self.db.create_session(session_id=self.session_id, room_name="local-pc")

        # 1. Wake Word Detector
        print(f"[1/4] Initializing openWakeWord Detector for '{settings.wake_word}'...", flush=True)
        self.wakeword_detector = OpenWakeWordDetector(
            wake_word=settings.wake_word,
            model_path=settings.wake_word_model_path if settings.wake_word_model_path else None,
            threshold=settings.wake_word_threshold,
            cooldown_seconds=settings.wake_word_cooldown,
        )
        print(f"\033[92m[OK] Wake-word engine initialized (Threshold: {settings.wake_word_threshold}).\033[0m", flush=True)

        # 2. Faster-Whisper STT (CPU int8)
        print(f"[2/4] Loading Faster-Whisper STT ({settings.whisper_model_size} on {settings.whisper_device})...", flush=True)
        try:
            self.stt_model = WhisperModel(
                settings.whisper_model_size,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
                cpu_threads=getattr(settings, "whisper_cpu_threads", 4),
            )
            print(f"\033[92m[OK] Faster-Whisper loaded on {settings.whisper_device} ({settings.whisper_compute_type}).\033[0m", flush=True)
        except Exception as e:
            print(f"\033[93m[!] CPU fallback: {e}\033[0m", flush=True)
            self.stt_model = WhisperModel(settings.whisper_model_size, device="cpu", compute_type="int8", cpu_threads=4)

        # 3. Piper TTS
        print(f"[3/4] Loading Piper TTS ({settings.piper_voice})...", flush=True)
        self.tts = LocalPiperTTS()
        print(f"\033[92m[OK] Piper TTS voice ready ({self.tts.sample_rate}Hz).\033[0m", flush=True)

        # 4. Ollama LLM
        print(f"[4/4] Connecting to Ollama LLM ({settings.ollama_model})...", flush=True)
        self.ollama_host = settings.ollama_host.rstrip("/")
        self.messages = [{"role": "system", "content": JARVIS_SYSTEM_PROMPT}]
        print(f"\033[92m[OK] Ollama ready at {self.ollama_host}.\033[0m", flush=True)

        self.sample_rate = 16000
        self.chunk_size = 1280  # 80ms chunks for openWakeWord
        self._running = False
        self._active_lock = threading.Lock()

        print("\033[92m" + "=" * 65)
        print(f"  [INFO] JARVIS V2 ONLINE. ALL SYSTEMS OPERATIONAL.")
        print("=" * 65 + "\033[0m\n", flush=True)

    def speak(self, text: str, suppress_wakeword: bool = True):
        """Synthesize text and play through local speakers with self-trigger protection."""
        if not text or not text.strip():
            return

        if suppress_wakeword:
            self.wakeword_detector.set_suppressed(True)
            self.state_machine.transition_to(AgentState.SPEAKING)

        print(f"\n\033[94m{settings.jarvis_name}:\033[0m {text}\n", flush=True)
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

        # Keep system prompt + last 12 messages for fast attention
        history = [self.messages[0]] + self.messages[-12:] if len(self.messages) > 13 else self.messages

        payload = {
            "model": settings.ollama_model,
            "messages": history,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 128,
            },
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                res = client.post(f"{self.ollama_host}/api/chat", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    reply = data.get("message", {}).get("content", "").strip()
                    print("\033[90m[INFO] Response received.\033[0m", flush=True)
                    return reply
                else:
                    return f"Error connecting to Ollama: HTTP {res.status_code}"
        except Exception as e:
            return f"I apologize, sir. An error occurred with the local reasoning model: {e}"

    def record_command(self, silence_threshold=0.012, max_seconds=12, silence_duration=1.2) -> Optional[np.ndarray]:
        """Record user command after wake word until end-of-speech silence is detected."""
        print("\033[93m🎙️  [LISTENING...] Speak your command into the mic now...\033[0m", flush=True)

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

    def voice_interaction_flow(self):
        """Perform voice command recording and reasoning."""
        self.state_machine.transition_to(AgentState.LISTENING)
        activation_phrase = settings.wake_word_activation_response.strip('"')
        self.speak(activation_phrase, suppress_wakeword=False)

        audio_cmd = self.record_command()
        if audio_cmd is None:
            print("\033[90m[INFO] No speech detected. Returning to IDLE.\033[0m\n", flush=True)
            self.state_machine.transition_to(AgentState.IDLE)
            return

        self.state_machine.transition_to(AgentState.PROCESSING)
        user_speech = self.transcribe(audio_cmd)
        if not user_speech:
            print("\033[90m[INFO] Could not understand speech. Returning to IDLE.\033[0m\n", flush=True)
            self.state_machine.transition_to(AgentState.IDLE)
            return

        print(f"\033[93mUser (Voice):\033[0m {user_speech}", flush=True)
        print("\033[90m[INFO] Sending request to Ollama...\033[0m", flush=True)
        reply = self.ask_llm(user_speech)
        self.speak(reply, suppress_wakeword=True)

    def text_interaction_flow(self, text: str):
        """Perform text reasoning and response."""
        print(f"\033[93mUser (Text):\033[0m {text}", flush=True)
        self.state_machine.transition_to(AgentState.PROCESSING)
        print("\033[90m[INFO] Sending request to Ollama...\033[0m", flush=True)
        reply = self.ask_llm(text)
        self.speak(reply, suppress_wakeword=True)

    def _background_wake_listener(self):
        """Continuously monitors microphone for 'Jarvis' wake-word in the background."""
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=self.chunk_size,
            ) as stream:
                while self._running:
                    if self.state_machine.is_state(AgentState.IDLE) and not self.wakeword_detector.is_suppressed:
                        audio_chunk, _ = stream.read(self.chunk_size)
                        detected = self.wakeword_detector.process_frame(audio_chunk)
                        if detected:
                            if self._active_lock.acquire(blocking=False):
                                try:
                                    print("\n\033[96m" + "*" * 50)
                                    print(f"  [INFO] Wake word '{settings.wake_word}' heard on mic!")
                                    print("*" * 50 + "\033[0m\n", flush=True)
                                    self.voice_interaction_flow()
                                finally:
                                    self._active_lock.release()
                    else:
                        time.sleep(0.05)
        except Exception as e:
            logger.debug(f"Wake listener background stream: {e}")

    def run(self):
        """Main loop: Always accepts keyboard typing and background wake-word activation."""
        self._running = True
        self.state_machine.transition_to(AgentState.IDLE)

        print("\033[97m" + "-" * 65)
        print("  JARVIS V2 CONTROLS:")
        print("   • [TYPE]:  Type your question below and press [ENTER]")
        print("   • [VOICE]: Or simply say 'Jarvis' hands-free into your mic")
        print("   • [MIC]:   Press [ENTER] on an empty line to start speaking directly")
        print("   • [EXIT]:  Type 'exit' to quit")
        print("-" * 65 + "\033[0m\n", flush=True)

        # Start background wake-word listener thread
        wake_thread = threading.Thread(target=self._background_wake_listener, daemon=True)
        wake_thread.start()

        # Initial startup greeting in background
        threading.Thread(target=lambda: self.speak(INITIAL_GREETING), daemon=True).start()

        while self._running:
            try:
                user_input = input("\033[92m[Type message OR Press ENTER to speak]: \033[0m").strip()

                if user_input.lower() in ("exit", "quit", "q"):
                    self.speak("Shutting down local systems. Goodbye, sir.")
                    self._running = False
                    break

                with self._active_lock:
                    if user_input == "":
                        # Direct voice trigger
                        self.voice_interaction_flow()
                    else:
                        # Direct text trigger
                        self.text_interaction_flow(user_input)

            except KeyboardInterrupt:
                print("\n[INFO] Stopping JARVIS...")
                self._running = False
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                self.state_machine.transition_to(AgentState.IDLE)


def main():
    agent = JarvisLocalVoiceV2()
    agent.run()


if __name__ == "__main__":
    main()
