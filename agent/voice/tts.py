"""
Local Text-to-Speech (TTS) implementation using Piper.
Synthesizes speech on the local machine and integrates with LiveKit Agents.
"""

import os
import io
import wave
import logging
import asyncio
import urllib.request
from pathlib import Path
from typing import Optional, Iterable

from piper.voice import PiperVoice
import livekit.rtc as rtc
from livekit.agents import tts
from livekit.agents.types import APIConnectOptions, DEFAULT_API_CONNECT_OPTIONS

from agent.config import settings

logger = logging.getLogger("jarvis.voice.tts")

# Mapping of standard Piper voice names to HuggingFace URLs
PIPER_VOICE_URLS = {
    "en_US-lessac-medium": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json",
    ),
    "en_GB-alan-medium": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_GB/alan/medium/en_GB-alan-medium.onnx.json",
    ),
    "en_US-amy-medium": (
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx",
        "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/en_US-amy-medium.onnx.json",
    ),
}


class _PiperChunkedStream(tts.ChunkedStream):
    """Chunked audio stream for Piper TTS synthesis."""

    def __init__(
        self,
        *,
        tts_instance: "LocalPiperTTS",
        input_text: str,
        conn_options: APIConnectOptions,
    ):
        super().__init__(
            tts=tts_instance,
            input_text=input_text,
            conn_options=conn_options,
        )
        self._piper_tts = tts_instance

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        voice = self._piper_tts.voice
        if not voice:
            raise RuntimeError("Piper voice model is not initialized")

        text = self.input_text.strip()
        if not text:
            return

        def _synthesize_chunks() -> list[bytes]:
            chunks = []
            # Synthesize raw PCM 16-bit audio
            for audio_chunk in voice.synthesize(text):
                data = getattr(audio_chunk, "audio_int16_bytes", None)
                if data:
                    chunks.append(data)
            return chunks

        audio_chunks = await asyncio.to_thread(_synthesize_chunks)

        for chunk_bytes in audio_chunks:
            output_emitter.push(chunk_bytes)


class LocalPiperTTS(tts.TTS):
    """
    Local Piper TTS engine supporting fast, high-quality offline speech synthesis.
    """

    def __init__(
        self,
        voice_name_or_path: Optional[str] = None,
        model_dir: str = "data/models/piper",
    ):
        self.voice_name = voice_name_or_path or settings.piper_voice
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.voice: Optional[PiperVoice] = None
        self._load_voice()

        sample_rate = self.voice.config.sample_rate if self.voice else 22050
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=sample_rate,
            num_channels=1,
        )

    def _ensure_voice_files(self) -> tuple[Path, Path]:
        """Ensure the onnx model and config json files exist on disk, downloading if necessary."""
        # Check if direct file path provided
        onnx_file = Path(self.voice_name)
        json_file = Path(f"{self.voice_name}.json")
        if onnx_file.exists() and json_file.exists():
            return onnx_file, json_file

        # Check in model_dir
        local_onnx = self.model_dir / f"{self.voice_name}.onnx"
        local_json = self.model_dir / f"{self.voice_name}.onnx.json"
        if local_onnx.exists() and local_json.exists():
            return local_onnx, local_json

        # Download from known URL if available
        if self.voice_name in PIPER_VOICE_URLS:
            onnx_url, json_url = PIPER_VOICE_URLS[self.voice_name]
            logger.info(f"Downloading Piper voice '{self.voice_name}' into {self.model_dir}...")
            if not local_json.exists():
                urllib.request.urlretrieve(json_url, local_json)
            if not local_onnx.exists():
                urllib.request.urlretrieve(onnx_url, local_onnx)
            logger.info(f"Piper voice '{self.voice_name}' download complete.")
            return local_onnx, local_json

        raise FileNotFoundError(
            f"Piper voice '{self.voice_name}' not found locally at {local_onnx} and no download URL known."
        )

    def _load_voice(self):
        """Load Piper ONNX model into memory."""
        try:
            onnx_path, json_path = self._ensure_voice_files()
            logger.info(f"Loading Piper voice from {onnx_path}...")
            self.voice = PiperVoice.load(str(onnx_path), config_path=str(json_path))
            logger.info(
                f"Piper voice '{self.voice_name}' ready (sample_rate={self.voice.config.sample_rate}Hz)."
            )
        except Exception as e:
            logger.error(f"Failed to load Piper voice '{self.voice_name}': {e}")
            raise

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> tts.ChunkedStream:
        return _PiperChunkedStream(
            tts_instance=self,
            input_text=text,
            conn_options=conn_options,
        )
