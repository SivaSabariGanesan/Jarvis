"""
Local Wake Word Detector for JARVIS V2.
Powered by openWakeWord with local ONNX runtime execution.
Zero cloud dependencies, strictly local inference.
"""

import os
import time
import logging
import asyncio
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Callable, Dict, Any

import numpy as np

logger = logging.getLogger("jarvis.wakeword")

# Default paths
WAKEWORD_MODELS_DIR = Path(__file__).resolve().parent / "models"


class BaseWakeWordDetector(ABC):
    """Abstract base class for local wake-word detectors."""

    @abstractmethod
    def start(self) -> None:
        """Initialize and start the detector."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the detector and release resources."""
        pass

    @abstractmethod
    def process_frame(self, audio_frame: np.ndarray) -> bool:
        """
        Process a single audio frame (16kHz int16 or float32 PCM).
        Returns True if the wake word was detected in this frame.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal prediction buffers."""
        pass


class OpenWakeWordDetector(BaseWakeWordDetector):
    """
    Local Wake Word Detector implementation using openWakeWord with ONNX runtime.
    Detects "Jarvis" / "Hey Jarvis" entirely offline with low CPU/GPU footprint.
    """

    def __init__(
        self,
        wake_word: str = "jarvis",
        model_path: Optional[str] = None,
        threshold: float = 0.5,
        cooldown_seconds: float = 1.0,
        on_detected: Optional[Callable[[], None]] = None,
    ):
        self.wake_word = wake_word.strip().lower()
        self.configured_model_path = model_path
        self.threshold = float(threshold)
        self.cooldown_seconds = float(cooldown_seconds)
        self.on_detected = on_detected

        self._model = None
        self._model_key = None
        self._last_detection_time = 0.0
        self._is_running = False
        self._suppressed = False
        self._lock = threading.Lock()

        # Chunk size expected by openWakeWord (1280 samples = 80ms at 16kHz)
        self.chunk_size = 1280

        self._resolve_and_load_model()

    def _resolve_model_path(self) -> str:
        """
        Resolve the ONNX model file path.
        Priority:
        1. Explicit configured path (if exists)
        2. Custom model in agent/wakeword/models/ (e.g. jarvis.onnx, hey_jarvis.onnx)
        3. Built-in openWakeWord hey_jarvis_v0.1.onnx
        """
        # 1. Explicit configured path
        if self.configured_model_path:
            p = Path(self.configured_model_path)
            if p.exists():
                logger.info(f"Using explicitly configured wake-word model: {p}")
                return str(p)
            else:
                logger.warning(f"Configured wake-word model path not found: {p}")

        # 2. Check in agent/wakeword/models/
        WAKEWORD_MODELS_DIR.mkdir(parents=True, exist_ok=True)
        candidate_names = [
            f"{self.wake_word}.onnx",
            f"hey_{self.wake_word}.onnx",
            f"hey_{self.wake_word}_v0.1.onnx",
            "hey_jarvis_v0.1.onnx",
            "jarvis.onnx",
        ]
        for name in candidate_names:
            custom_path = WAKEWORD_MODELS_DIR / name
            if custom_path.exists():
                logger.info(f"Found custom wake-word model in models directory: {custom_path}")
                return str(custom_path)

        # 3. Fall back to standard openWakeWord ONNX model
        logger.info(f"Resolving openWakeWord built-in model for '{self.wake_word}'...")
        return "hey_jarvis_v0.1.onnx"

    def _resolve_and_load_model(self) -> None:
        """Load openWakeWord ONNX model."""
        try:
            import openwakeword
            import openwakeword.utils
            from openwakeword.model import Model

            model_target = self._resolve_model_path()
            logger.info(f"[INFO] Initializing openWakeWord detector with target: {model_target}")

            # Ensure openWakeWord base embedding and melspectrogram models exist
            try:
                openwakeword.utils.download_models()
            except Exception as e:
                logger.debug(f"openWakeWord base models check: {e}")

            self._model = Model(
                wakeword_models=[model_target],
                inference_framework="onnx",
            )

            # Extract the active model key
            model_keys = list(self._model.models.keys())
            if not model_keys:
                raise RuntimeError(f"Failed to load any wake word model from '{model_target}'")

            self._model_key = model_keys[0]
            self._is_running = True
            logger.info(
                f"[INFO] Wake-word engine initialized successfully. Active model key: '{self._model_key}', threshold: {self.threshold}"
            )

        except Exception as e:
            logger.error(f"[ERROR] Wake-word engine failed to initialize: {e}")
            logger.info(
                f"[INFO] Expected model location: {WAKEWORD_MODELS_DIR / 'jarvis.onnx'} or openwakeword built-in 'hey_jarvis_v0.1.onnx'"
            )
            raise RuntimeError(f"Wake-word initialization failed: {e}") from e

    def set_suppressed(self, suppressed: bool) -> None:
        """
        Suppress wake-word detection (e.g. while JARVIS is speaking)
        to prevent self-trigger feedback loops.
        """
        with self._lock:
            self._suppressed = suppressed
            if suppressed:
                self.reset()

    @property
    def is_suppressed(self) -> bool:
        with self._lock:
            return self._suppressed

    def start(self) -> None:
        with self._lock:
            self._is_running = True
            self.reset()
            logger.info(f"[INFO] Wake-word detector started for word: '{self.wake_word}'")

    def stop(self) -> None:
        with self._lock:
            self._is_running = False
            self.reset()
            logger.info("[INFO] Wake-word detector stopped.")

    def reset(self) -> None:
        """Reset openWakeWord buffer and internal scores."""
        with self._lock:
            if self._model:
                try:
                    self._model.reset()
                except Exception:
                    pass

    def process_frame(self, audio_frame: np.ndarray) -> bool:
        """
        Process a single audio frame of 16-bit PCM samples.
        Audio should be 16kHz mono.
        Returns True if wake word score exceeds threshold.
        """
        with self._lock:
            if not self._is_running or self._suppressed:
                return False

            if self._model is None or self._model_key is None:
                return False

            now = time.time()
            # Enforce cooldown period to prevent duplicate rapid triggers
            if now - self._last_detection_time < self.cooldown_seconds:
                return False

            # Convert to int16 1D numpy array if needed
            if audio_frame.dtype != np.int16:
                if audio_frame.dtype == np.float32:
                    audio_frame = (audio_frame * 32767).astype(np.int16)
                else:
                    audio_frame = audio_frame.astype(np.int16)

            audio_data = audio_frame.flatten()
            if len(audio_data) == 0:
                return False

            # Predict scores
            try:
                prediction = self._model.predict(audio_data)
                score = prediction.get(self._model_key, 0.0)

                if score >= self.threshold:
                    self._last_detection_time = now
                    logger.info(
                        f"[INFO] Wake word '{self.wake_word}' detected! (confidence: {score:.3f} >= {self.threshold})"
                    )
                    if self.on_detected:
                        try:
                            self.on_detected()
                        except Exception as cb_err:
                            logger.error(f"Error in on_detected callback: {cb_err}")
                    return True

            except Exception as e:
                logger.debug(f"Wake word prediction step error: {e}")
                return False

            return False
