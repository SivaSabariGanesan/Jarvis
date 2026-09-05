"""
JARVIS V2 Voice Agent Orchestrator.
Binds Silero VAD, Whisper STT, Ollama LLM, Piper TTS, and Local Wake-Word Detector ("Jarvis")
into an explicit 4-state finite state machine (IDLE -> LISTENING -> PROCESSING -> SPEAKING).
"""

import logging
import asyncio
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
from agent.wakeword import AgentState, StateMachine, OpenWakeWordDetector

logger = logging.getLogger("jarvis.agent")


class JarvisVoiceAgent:
    """
    JARVIS Voice Agent V2 managing speech pipeline, local wake-word gating,
    conversational state transitions, and memory persistence.
    """

    def __init__(self, db: Optional[JarvisDatabase] = None):
        self.db = db or JarvisDatabase()
        self.ollama_client = OllamaClient()
        self.state_machine = StateMachine(AgentState.IDLE)

        logger.info(f"[INFO] Initializing {settings.jarvis_name} Voice Agent V2 components...")
        logger.info(f"[INFO] State: {self.state_machine.current_state.value}")

        # 1. Local Wake Word Detector
        logger.info(f"[INFO] Initializing Wake-word engine for '{settings.wake_word}'...")
        try:
            self.wakeword_detector = OpenWakeWordDetector(
                wake_word=settings.wake_word,
                model_path=settings.wake_word_model_path if settings.wake_word_model_path else None,
                threshold=settings.wake_word_threshold,
                cooldown_seconds=settings.wake_word_cooldown,
            )
            logger.info(f"[INFO] Wake word: {settings.wake_word}")
        except Exception as e:
            logger.error(f"[ERROR] Wake-word engine failed to initialize: {e}")
            raise

        # 2. Voice Activity Detection (Silero VAD)
        logger.info("[INFO] Loading Silero VAD...")
        self.vad_plugin = silero.VAD.load()

        # 3. Speech-to-Text (Faster-Whisper wrapped in StreamAdapter)
        logger.info(f"[INFO] Initializing Local Whisper STT ({settings.whisper_model_size} on {settings.whisper_device})...")
        self.stt_plugin = create_streaming_stt(vad_instance=self.vad_plugin)

        # 4. Large Language Model (Ollama via LiveKit adapter)
        logger.info(f"[INFO] Configuring Ollama LLM model='{settings.ollama_model}'...")
        self.llm_plugin = self.ollama_client.get_livekit_llm()

        # 5. Text-to-Speech (Local Piper TTS)
        logger.info(f"[INFO] Loading Piper TTS voice='{settings.piper_voice}'...")
        self.tts_plugin = LocalPiperTTS()

        logger.info("[INFO] All Voice Pipeline components initialized successfully.")

    def create_session(self, room: rtc.Room) -> tuple[AgentSession, Agent]:
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

        # Self-trigger protection & State synchronization
        @session.on("user_input_transcribed")
        def on_user_input(ev):
            if hasattr(ev, "text") and ev.text:
                logger.info(f"[INFO] State: {AgentState.PROCESSING.value}")
                self.state_machine.transition_to(AgentState.PROCESSING)
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

        @session.on("agent_started_speaking")
        def on_agent_started_speaking():
            self.state_machine.transition_to(AgentState.SPEAKING)
            self.wakeword_detector.set_suppressed(True)
            logger.info(f"[INFO] State: {AgentState.SPEAKING.value}")

        @session.on("agent_stopped_speaking")
        def on_agent_stopped_speaking():
            self.wakeword_detector.set_suppressed(False)
            self.state_machine.transition_to(AgentState.IDLE)
            logger.info(f"[INFO] State: {AgentState.IDLE.value}")
            logger.info(f"[INFO] Waiting for wake word '{settings.wake_word}'...")

        return session, agent

    async def run(self, ctx: JobContext) -> None:
        """Entrypoint for a LiveKit room job."""
        logger.info(f"[INFO] JARVIS starting. Connecting to LiveKit room: {ctx.room.name}")
        await ctx.connect()

        logger.info("[INFO] Connected to LiveKit. Waiting for participant...")
        participant = await ctx.wait_for_participant()
        logger.info(f"[INFO] Participant joined: {participant.identity}")

        session, agent = self.create_session(ctx.room)
        await session.start(agent, room=ctx.room)

        logger.info(f"[INFO] State: {AgentState.IDLE.value}")
        logger.info(f"[INFO] Waiting for wake word '{settings.wake_word}'...")


async def entrypoint(ctx: JobContext) -> None:
    """LiveKit worker entrypoint callback."""
    agent_instance = JarvisVoiceAgent()
    await agent_instance.run(ctx)
