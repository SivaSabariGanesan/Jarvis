"""
Local Speech-to-Text (STT) implementation using faster-whisper.
Integrates with LiveKit Agents and VAD via StreamAdapter.
"""

import io
import logging
import asyncio
import numpy as np
from typing import Optional

from faster_whisper import WhisperModel
import livekit.rtc as rtc
from livekit.agents import stt, vad
from livekit.agents.types import APIConnectOptions, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer

from agent.config import settings

logger = logging.getLogger("jarvis.voice.stt")


class LocalWhisperSTT(stt.STT):
    """
    Local Whisper STT engine powered by faster-whisper.
    """

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False,
                interim_results=False,
            )
        )
        self.model_size = model_size or settings.whisper_model_size
        self.requested_device = device or settings.whisper_device
        self.compute_type = compute_type or settings.whisper_compute_type
        self.whisper_model: Optional[WhisperModel] = None
        self._load_model()

    def _load_model(self):
        """Load the faster-whisper model with fallback to CPU if CUDA fails."""
        device = self.requested_device
        compute_type = self.compute_type

        try:
            logger.info(
                f"Loading faster-whisper model '{self.model_size}' on device={device} (compute_type={compute_type})..."
            )
            self.whisper_model = WhisperModel(
                self.model_size,
                device=device,
                device_index=0 if device == "cuda" else 0,
                compute_type=compute_type,
                cpu_threads=4,
            )
            logger.info(f"faster-whisper model '{self.model_size}' successfully loaded on {device}.")
        except Exception as e:
            if device != "cpu":
                logger.warning(
                    f"Failed to load faster-whisper on {device} ({e}). Falling back to CPU int8..."
                )
                self.whisper_model = WhisperModel(
                    self.model_size,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=4,
                )
                logger.info(f"faster-whisper model '{self.model_size}' loaded on CPU fallback.")
            else:
                logger.error(f"Failed to load faster-whisper model: {e}")
                raise

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions,
    ) -> stt.SpeechEvent:
        """Transcribe an audio buffer into text using faster-whisper."""
        if not self.whisper_model:
            raise RuntimeError("Whisper model is not initialized")

        # Combine audio buffer into a single AudioFrame
        frame = rtc.combine_audio_frames(buffer)
        if frame.samples_per_channel == 0:
            return stt.SpeechEvent(
                type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                alternatives=[],
            )

        # Convert 16-bit PCM to float32 numpy array
        np_audio = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0

        lang = None if language is NOT_GIVEN else language

        # Run inference in worker thread to prevent blocking the async loop
        def _transcribe():
            segments, info = self.whisper_model.transcribe(
                np_audio,
                beam_size=5,
                language=lang,
                vad_filter=True,
            )
            text = " ".join(seg.text for seg in segments).strip()
            return text, info.language, info.language_probability

        text, detected_lang, confidence = await asyncio.to_thread(_transcribe)

        logger.debug(f"Transcribed audio: '{text}' (lang={detected_lang}, conf={confidence:.2f})")

        speech_data = stt.SpeechData(
            language=detected_lang or "en",
            text=text,
            confidence=confidence,
        )

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[speech_data] if text else [],
        )


def create_streaming_stt(
    vad_instance: Optional[vad.VAD] = None,
    model_size: Optional[str] = None,
    device: Optional[str] = None,
) -> stt.STT:
    """
    Create a streaming STT pipeline by wrapping LocalWhisperSTT with a VAD StreamAdapter.
    """
    from livekit.plugins import silero

    active_vad = vad_instance or silero.VAD.load()
    base_stt = LocalWhisperSTT(model_size=model_size, device=device)
    return stt.StreamAdapter(stt=base_stt, vad=active_vad)
