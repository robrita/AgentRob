"""Personal Voice TTS streaming service.

Uses Azure Speech SDK to synthesize audio with Personal Voice (speakerProfileId)
in real-time, streaming audio chunks as they're generated.
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
from collections.abc import AsyncIterator, Callable
from contextlib import suppress
from dataclasses import dataclass
from html import escape as xml_escape
from typing import Any

logger = logging.getLogger("agentrob.personal_voice")


@dataclass
class PersonalVoiceConfig:
    """Configuration for Personal Voice TTS."""
    speech_key: str
    speech_region: str
    speaker_profile_id: str
    base_voice: str = "DragonLatestNeural"
    language: str = "en-US"
    output_format: str = "Riff24Khz16BitMonoPcm"  # or Raw24Khz16BitMonoPcm for streaming

    @classmethod
    def from_env(cls) -> PersonalVoiceConfig | None:
        """Load config from environment variables."""
        speech_key = os.getenv("AZURE_SPEECH_KEY") or os.getenv("AZURE_VOICELIVE_SPEECH_KEY")
        speech_region = os.getenv("AZURE_SPEECH_REGION") or os.getenv("AZURE_VOICELIVE_SPEECH_REGION")
        speaker_profile_id = os.getenv("AZURE_VOICELIVE_SPEAKER_PROFILE_ID")
        
        if not all([speech_key, speech_region, speaker_profile_id]):
            return None
        
        base_voice = os.getenv("AZURE_VOICELIVE_TTS_BASE_VOICE", "DragonLatestNeural")
        language = os.getenv("AZURE_VOICELIVE_TTS_LANGUAGE", "en-US")
        
        return cls(
            speech_key=speech_key,
            speech_region=speech_region,
            speaker_profile_id=speaker_profile_id,
            base_voice=base_voice,
            language=language,
        )

    def is_valid(self) -> bool:
        return bool(self.speech_key and self.speech_region and self.speaker_profile_id)


def build_personal_voice_ssml(
    text: str,
    speaker_profile_id: str,
    voice_name: str = "DragonLatestNeural",
    language: str = "en-US",
) -> str:
    """Build SSML for Personal Voice synthesis.
    
    Uses mstts:ttsembedding with speakerProfileId for Personal Voice.
    """
    safe_text = xml_escape(text)
    safe_profile = xml_escape(speaker_profile_id)
    safe_voice = xml_escape(voice_name)
    safe_lang = xml_escape(language)
    
    return (
        "<speak version='1.0' "
        "xmlns='http://www.w3.org/2001/10/synthesis' "
        f"xml:lang='{safe_lang}' "
        "xmlns:mstts='http://www.w3.org/2001/mstts'>"
        f"<voice name='{safe_voice}'>"
        f"<mstts:ttsembedding speakerProfileId='{safe_profile}'>"
        f"<lang xml:lang='{safe_lang}'>{safe_text}</lang>"
        "</mstts:ttsembedding>"
        "</voice>"
        "</speak>"
    )


class SentenceBuffer:
    """Buffers text and yields complete sentences for synthesis."""
    
    # Sentence-ending patterns
    SENTENCE_END = re.compile(r'[.!?]\s*$')
    SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
    
    def __init__(self, min_chars: int = 20):
        self.buffer = ""
        self.min_chars = min_chars
    
    def add(self, text: str) -> list[str]:
        """Add text and return any complete sentences."""
        self.buffer += text
        sentences = []
        
        # Split on sentence boundaries
        parts = self.SENTENCE_SPLIT.split(self.buffer)
        
        # Keep last part in buffer if it doesn't end with sentence marker
        if len(parts) > 1:
            # All but last are complete sentences
            for part in parts[:-1]:
                if part.strip():
                    sentences.append(part.strip())
            self.buffer = parts[-1]
        elif self.SENTENCE_END.search(self.buffer) and len(self.buffer) >= self.min_chars:
            # Single complete sentence
            sentences.append(self.buffer.strip())
            self.buffer = ""
        
        return sentences
    
    def flush(self) -> str | None:
        """Flush remaining buffer content."""
        if self.buffer.strip():
            result = self.buffer.strip()
            self.buffer = ""
            return result
        return None


class PersonalVoiceSynthesizer:
    """Synthesizes audio using Personal Voice with TRUE streaming output."""
    
    def __init__(self, config: PersonalVoiceConfig):
        self.config = config
        self._sdk = None
        self._speech_config = None
    
    def _get_sdk(self):
        """Lazy load Speech SDK."""
        if self._sdk is None:
            try:
                import azure.cognitiveservices.speech as speechsdk
                self._sdk = speechsdk
            except ImportError as exc:
                raise RuntimeError(
                    "Azure Speech SDK not installed. "
                    "Add 'azure-cognitiveservices-speech' to dependencies."
                ) from exc
        return self._sdk
    
    def _get_speech_config(self):
        """Get or create speech config."""
        if self._speech_config is None:
            sdk = self._get_sdk()
            self._speech_config = sdk.SpeechConfig(
                subscription=self.config.speech_key,
                region=self.config.speech_region,
            )
            # Use raw PCM for streaming (no WAV header)
            self._speech_config.set_speech_synthesis_output_format(
                sdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm
            )
        return self._speech_config
    
    async def synthesize_to_chunks(
        self,
        text: str,
        chunk_callback: Callable[[bytes], Any] | None = None,
    ) -> AsyncIterator[bytes]:
        """Synthesize text to audio and yield chunks AS THEY STREAM.
        
        Uses event-based streaming for real-time audio output.
        
        Args:
            text: Text to synthesize
            chunk_callback: Optional callback for each chunk (for streaming)
            
        Yields:
            Audio chunks as bytes (raw PCM 24kHz 16-bit mono)
        """
        sdk = self._get_sdk()
        speech_config = self._get_speech_config()
        
        # Queue to receive streaming audio chunks
        audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        
        def synthesis_started(evt):
            logger.debug("Synthesis started")
        
        def synthesizing(evt):
            """Called repeatedly with audio chunks as they're generated."""
            if evt.result.audio_data:
                # Put chunk in queue (will be processed async)
                try:
                    audio_queue.put_nowait(evt.result.audio_data)
                except Exception:
                    pass
        
        def synthesis_completed(evt):
            """Called when synthesis is fully complete."""
            audio_queue.put_nowait(None)  # Signal completion
        
        def synthesis_canceled(evt):
            """Called when synthesis is canceled."""
            logger.warning("Synthesis canceled: %s", evt.cancellation_details.error_details if evt.cancellation_details else "unknown")
            audio_queue.put_nowait(None)  # Signal completion
        
        # Create synthesizer without audio config (we'll get audio from events)
        synthesizer = sdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=None,  # No output config - get audio from synthesizing events
        )
        
        # Connect event handlers
        synthesizer.synthesis_started.connect(synthesis_started)
        synthesizer.synthesizing.connect(synthesizing)
        synthesizer.synthesis_completed.connect(synthesis_completed)
        synthesizer.synthesis_canceled.connect(synthesis_canceled)
        
        ssml = build_personal_voice_ssml(
            text=text,
            speaker_profile_id=self.config.speaker_profile_id,
            voice_name=self.config.base_voice,
            language=self.config.language,
        )
        
        logger.debug("Starting streaming synthesis: text_len=%s", len(text))
        
        # Start synthesis asynchronously (non-blocking)
        loop = asyncio.get_event_loop()
        synthesis_future = loop.run_in_executor(
            None,
            lambda: synthesizer.speak_ssml_async(ssml).get()
        )
        
        # Yield chunks as they arrive from the synthesizing events
        try:
            while True:
                try:
                    # Wait for next chunk with timeout
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=10.0)
                    if chunk is None:
                        # Synthesis complete
                        break
                    if chunk_callback:
                        chunk_callback(chunk)
                    yield chunk
                except TimeoutError:
                    # Check if synthesis is still running
                    if synthesis_future.done():
                        break
        finally:
            # Ensure synthesis completes
            try:
                await synthesis_future
            except Exception:
                pass
    
    async def synthesize_streaming(
        self,
        text: str,
    ) -> tuple[bytes, int]:
        """Synthesize text and return full audio data.
        
        Returns:
            Tuple of (audio_bytes, sample_rate)
        """
        chunks = []
        async for chunk in self.synthesize_to_chunks(text):
            chunks.append(chunk)
        
        return b"".join(chunks), 24000


class StreamingPersonalVoiceTTS:
    """Manages streaming TTS for a conversation session.
    
    Buffers incoming text, synthesizes complete sentences,
    and streams audio chunks back.
    """
    
    def __init__(self, config: PersonalVoiceConfig):
        self.config = config
        self.synthesizer = PersonalVoiceSynthesizer(config)
        self.buffer = SentenceBuffer(min_chars=15)
        self._synthesis_queue: asyncio.Queue[str] = asyncio.Queue()
        self._audio_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._running = False
        self._synthesis_task: asyncio.Task | None = None
    
    async def start(self):
        """Start the synthesis background task."""
        if self._running:
            return
        self._running = True
        self._synthesis_task = asyncio.create_task(self._synthesis_loop())
        logger.info("Personal Voice TTS streaming started")
    
    async def stop(self):
        """Stop the synthesis background task."""
        self._running = False
        if self._synthesis_task:
            self._synthesis_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._synthesis_task
        logger.info("Personal Voice TTS streaming stopped")
    
    async def add_text(self, text: str) -> None:
        """Add text delta to the buffer."""
        sentences = self.buffer.add(text)
        for sentence in sentences:
            await self._synthesis_queue.put(sentence)
    
    async def flush(self) -> None:
        """Flush any remaining buffered text."""
        remaining = self.buffer.flush()
        if remaining:
            await self._synthesis_queue.put(remaining)
    
    async def get_audio_event(self) -> dict[str, Any] | None:
        """Get next audio event (non-blocking)."""
        try:
            return self._audio_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None
    
    async def iter_audio_events(self) -> AsyncIterator[dict[str, Any]]:
        """Iterate over audio events as they're generated."""
        while self._running or not self._audio_queue.empty():
            try:
                event = await asyncio.wait_for(
                    self._audio_queue.get(),
                    timeout=0.1
                )
                yield event
            except TimeoutError:
                continue
    
    async def _synthesis_loop(self):
        """Background loop that synthesizes queued text."""
        while self._running:
            try:
                # Wait for text to synthesize
                text = await asyncio.wait_for(
                    self._synthesis_queue.get(),
                    timeout=0.5
                )
                
                logger.info("Synthesizing sentence: %s...", text[:50] if len(text) > 50 else text)
                
                # Synthesize and queue audio events
                async for chunk in self.synthesizer.synthesize_to_chunks(text):
                    # Create audio delta event matching Voice Live format
                    audio_event = {
                        "type": "response.audio.delta",
                        "delta": base64.b64encode(chunk).decode("ascii"),
                        "audio_sampling_rate": 24000,
                    }
                    await self._audio_queue.put(audio_event)
                
                # Send audio done event for this segment
                await self._audio_queue.put({
                    "type": "response.audio.segment_done",
                })
                
            except TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Error in Personal Voice synthesis loop: %s", e)


def is_personal_voice_enabled() -> bool:
    """Check if Personal Voice TTS is enabled via environment."""
    voice_type = os.getenv("AZURE_VOICELIVE_TTS_VOICE_TYPE", "").strip().lower()
    return voice_type == "azure-personal"


def create_streaming_tts() -> StreamingPersonalVoiceTTS | None:
    """Create a streaming TTS instance if Personal Voice is configured."""
    if not is_personal_voice_enabled():
        return None
    
    config = PersonalVoiceConfig.from_env()
    if not config or not config.is_valid():
        logger.warning(
            "Personal Voice enabled but missing config. "
            "Set AZURE_SPEECH_KEY, AZURE_SPEECH_REGION, AZURE_VOICELIVE_SPEAKER_PROFILE_ID"
        )
        return None
    
    return StreamingPersonalVoiceTTS(config)
