"""
JARVIS Voice Agent orchestrator.
Binds Silero VAD, Whisper STT, Ollama LLM, and Piper TTS into a real-time LiveKit AgentSession.
"""

import logging
from typing import Optional

import livekit.rtc as rtc
from livekit.agents import JobContext, vad
from livekit.agents.voice import Agent, AgentSession
from livekit.plugins import silero

from agent.config import settings
from agent.prompt import JARVIS_SYSTEM_PROMPT, INITIAL_GREETING
from agent.ai.ollama_client import OllamaClient
from agent.voice.stt import create_streaming_stt
from agent.voice.tts import LocalPiperTTS
from agent.memory.database import JarvisDatabase

logger = logging.getLogger("jarvis.agent")


class JarvisVoiceAgent:
    """
    JARVIS Voice Agent managing speech pipeline, conversational state,
    and memory persistence.
    """

    def __init__(self, db: Optional[JarvisDatabase] = None):
        self.db = db or JarvisDatabase()
        self.ollama_client = OllamaClient()

        logger.info(f"Initializing {settings.jarvis_name} Voice Agent components...")

        # 1. Voice Activity Detection (Silero VAD)
        logger.info("Loading Silero VAD...")
        self.vad_plugin = silero.VAD.load()

        # 2. Speech-to-Text (Faster-Whisper wrapped in StreamAdapter)
        logger.info("Initializing Local Whisper STT...")
        self.stt_plugin = create_streaming_stt(vad_instance=self.vad_plugin)

        # 3. Large Language Model (Ollama via LiveKit adapter)
        logger.info(f"Configuring Ollama LLM model='{settings.ollama_model}'...")
        self.llm_plugin = self.ollama_client.get_livekit_llm()

        # 4. Text-to-Speech (Local Piper TTS)
        logger.info(f"Loading Piper TTS voice='{settings.piper_voice}'...")
        self.tts_plugin = LocalPiperTTS()

        logger.info("All Voice Pipeline components initialized successfully.")

    def create_session(self, room: rtc.Room) -> AgentSession:
        """Create and configure a LiveKit AgentSession for the current room."""
        session_id = room.name or "local-session"
        self.db.create_session(session_id=session_id, room_name=room.name)

        session = AgentSession(
            vad=self.vad_plugin,
            stt=self.stt_plugin,
            llm=self.llm_plugin,
            tts=self.tts_plugin,
        )

        agent = Agent(
            instructions=JARVIS_SYSTEM_PROMPT,
        )

        @session.on("user_input_transcribed")
        def on_user_input(ev):
            if hasattr(ev, "text") and ev.text:
                logger.info(f"User: {ev.text}")
                self.db.log_message(session_id=session_id, role="user", content=ev.text)

        @session.on("conversation_item_added")
        def on_conversation_item(ev):
            item = getattr(ev, "item", None)
            if item and getattr(item, "role", "") == "assistant":
                content = getattr(item, "text_content", "") or getattr(item, "content", "")
                if content:
                    logger.info(f"{settings.jarvis_name}: {content}")
                    self.db.log_message(
                        session_id=session_id, role="assistant", content=str(content)
                    )

        return session, agent

    async def run(self, ctx: JobContext) -> None:
        """Entrypoint for a LiveKit room job."""
        logger.info(f"[INFO] Connecting {settings.jarvis_name} to LiveKit room: {ctx.room.name}")
        await ctx.connect()

        logger.info("[INFO] Connected to LiveKit. Waiting for participant...")
        participant = await ctx.wait_for_participant()
        logger.info(f"[INFO] Participant joined: {participant.identity}")

        session, agent = self.create_session(ctx.room)
        await session.start(agent, room=ctx.room)

        logger.info(f"[INFO] Voice agent ready. Sending initial greeting...")
        session.say(INITIAL_GREETING)


async def entrypoint(ctx: JobContext) -> None:
    """LiveKit worker entrypoint callback."""
    agent_instance = JarvisVoiceAgent()
    await agent_instance.run(ctx)
