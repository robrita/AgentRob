from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import suppress
from typing import Any
from urllib.parse import urlencode

import anyio
import httpx
import websockets
from azure.identity.aio import DefaultAzureCredential as AsyncDefaultAzureCredential

logger = logging.getLogger("agentrob.voice")

# Environment variables configuration for Azure Voice Live API with AI Foundry Agent
# Required:
#   AZURE_VOICE_LIVE_ENDPOINT: WebSocket endpoint URL for Voice Live API (e.g., https://<region>.voice.azure.com)
#   AI_FOUNDRY_PROJECT_NAME: Azure AI Foundry project name containing the agent
#   AI_FOUNDRY_AGENT_ID: Unique identifier for the agent configured in Azure AI Foundry
#
# Optional:
#   AZURE_VOICE_LIVE_API_VERSION: API version (default: 2025-10-01)
#   AI_FOUNDRY_AGENT_CONNECTION_STRING or AZURE_VOICELIVE_AGENT_CONNECTION_STRING: hub-based projects
#   AZURE_VOICELIVE_API_KEY or AZURE_VOICE_LIVE_API_KEY: Voice Live API key (service auth; agent token still required)
#   AZURE_VOICELIVE_TTS_VOICE: TTS voice name (e.g., en-IN-Meera:DragonHDV2.3Neural)
#   AZURE_VOICELIVE_TTS_VOICE_TYPE: Voice type - azure-standard, openai, etc. (default: azure-standard)
#   AZURE_VOICELIVE_TTS_TEMPERATURE: TTS temperature for voice variation (default: 0.8)
#   AZURE_VOICELIVE_VAD_THRESHOLD: Voice activity detection threshold (default: 0.3)
#   AZURE_VOICELIVE_VAD_PREFIX_PADDING_MS: VAD prefix padding in milliseconds (default: 200)
#   AZURE_VOICELIVE_VAD_SILENCE_DURATION_MS: Silence duration threshold in milliseconds (default: 200)
#   AZURE_VOICELIVE_AUDIO_SAMPLING_RATE: Input audio sampling rate in Hz (default: 24000)
#   AZURE_VOICE_LIVE_TRANSCRIPTION_LANGUAGES: Comma-separated language codes (default: en-US)
#
# Authentication:
#   Uses DefaultAzureCredential (supports: environment variables, managed identity, CLI login, etc.)
#   Requires scope: https://ai.azure.com/.default


class AzureVoiceLiveConfig:
    def __init__(self) -> None:
        self.endpoint = os.getenv("AZURE_VOICE_LIVE_ENDPOINT") or os.getenv("AZURE_VOICELIVE_ENDPOINT")
        self.api_version = os.getenv("AZURE_VOICE_LIVE_API_VERSION", "2025-10-01")
        self.agent_project_name = (
            os.getenv("AI_FOUNDRY_PROJECT_NAME") or os.getenv("AZURE_VOICELIVE_PROJECT_NAME")
        )
        self.agent_id = os.getenv("AI_FOUNDRY_AGENT_ID") or os.getenv("AZURE_VOICELIVE_AGENT_ID")
        self.agent_connection_string = os.getenv("AI_FOUNDRY_AGENT_CONNECTION_STRING") or os.getenv(
            "AZURE_VOICELIVE_AGENT_CONNECTION_STRING"
        )
        self.api_key = os.getenv("AZURE_VOICELIVE_API_KEY") or os.getenv("AZURE_VOICE_LIVE_API_KEY")

        self.tts_voice = os.getenv("AZURE_VOICELIVE_TTS_VOICE")
        self.tts_voice_type = os.getenv("AZURE_VOICELIVE_TTS_VOICE_TYPE", "azure-standard")
        self.tts_temperature = os.getenv("AZURE_VOICELIVE_TTS_TEMPERATURE")
        self.fallback_model = os.getenv("AZURE_VOICELIVE_MODEL", "gpt-4o-mini-realtime-preview")

        self.vad_threshold = os.getenv("AZURE_VOICELIVE_VAD_THRESHOLD")
        self.vad_prefix_padding_ms = os.getenv("AZURE_VOICELIVE_VAD_PREFIX_PADDING_MS")
        self.vad_silence_duration_ms = os.getenv("AZURE_VOICELIVE_VAD_SILENCE_DURATION_MS")

        self.input_audio_sampling_rate = os.getenv("AZURE_VOICELIVE_AUDIO_SAMPLING_RATE")
        self.output_audio_sampling_rate_s = os.getenv("AZURE_VOICELIVE_OUTPUT_AUDIO_SAMPLING_RATE")
        self.transcription_languages = os.getenv(
            "AZURE_VOICE_LIVE_TRANSCRIPTION_LANGUAGES", "en-US"
        )
        self.open_timeout_s = os.getenv("AZURE_VOICE_LIVE_OPEN_TIMEOUT", "30")
        self.connect_retries = os.getenv("AZURE_VOICE_LIVE_CONNECT_RETRIES", "3")
        self.retry_backoff_s = os.getenv("AZURE_VOICE_LIVE_RETRY_BACKOFF_SECONDS", "1.5")
        self.ping_interval_s = os.getenv("AZURE_VOICE_LIVE_PING_INTERVAL", "20")
        self.ping_timeout_s = os.getenv("AZURE_VOICE_LIVE_PING_TIMEOUT", "20")

    def _coerce_float(self, value: str, default: float) -> float:
        with suppress(ValueError):
            return float(value)
        return default

    def _coerce_int(self, value: str, default: int) -> int:
        with suppress(ValueError):
            return int(value)
        return default

    @property
    def open_timeout(self) -> float:
        return self._coerce_float(self.open_timeout_s or "", 30.0)

    @property
    def retries(self) -> int:
        retries = self._coerce_int(self.connect_retries or "", 3)
        return max(retries, 1)

    @property
    def retry_backoff(self) -> float:
        return self._coerce_float(self.retry_backoff_s or "", 1.5)

    @property
    def ping_interval(self) -> float:
        return self._coerce_float(self.ping_interval_s or "", 20.0)

    @property
    def ping_timeout(self) -> float:
        return self._coerce_float(self.ping_timeout_s or "", 20.0)

    @property
    def token_scope(self) -> str:
        endpoint = (self.endpoint or "").lower()
        if self.agent_connection_string:
            return "https://ml.azure.com/.default"
        if "cognitiveservices.azure.com" in endpoint or "openai.azure.com" in endpoint:
            return "https://cognitiveservices.azure.com/.default"
        return "https://ai.azure.com/.default"

    @property
    def agent_token_scope(self) -> str:
        if self.agent_connection_string:
            return "https://ml.azure.com/.default"
        return "https://ai.azure.com/.default"

    @property
    def output_audio_sampling_rate(self) -> int:
        if self.output_audio_sampling_rate_s:
            with suppress(ValueError):
                return int(self.output_audio_sampling_rate_s)
        if self.input_audio_sampling_rate:
            with suppress(ValueError):
                return int(self.input_audio_sampling_rate)
        return 24000

    @property
    def is_configured(self) -> bool:
        return bool(self.endpoint and self.agent_id and (self.agent_project_name or self.agent_connection_string))

    def resolve_endpoint(self) -> str:
        if not self.endpoint:
            raise RuntimeError("AZURE_VOICE_LIVE_ENDPOINT is not set.")
        normalized = self.endpoint.rstrip("/")
        if not normalized.startswith("https://"):
            raise RuntimeError("AZURE_VOICE_LIVE_ENDPOINT must start with https://")
        if "cognitiveservices.azure.com" in normalized and self.agent_project_name:
            logger.warning(
                "Endpoint is Cognitive Services but agent project is set. Foundry agents typically require services.ai.azure.com endpoints."
            )
        return normalized


def _build_turn_detection(config: AzureVoiceLiveConfig) -> dict[str, Any]:
    threshold = 0.3
    prefix_padding_ms = 200
    silence_duration_ms = 200

    if config.vad_threshold:
        with suppress(ValueError):
            threshold = float(config.vad_threshold)

    if config.vad_prefix_padding_ms:
        with suppress(ValueError):
            prefix_padding_ms = int(config.vad_prefix_padding_ms)

    if config.vad_silence_duration_ms:
        with suppress(ValueError):
            silence_duration_ms = int(config.vad_silence_duration_ms)

    return {
        "type": "azure_semantic_vad",
        "threshold": threshold,
        "prefix_padding_ms": prefix_padding_ms,
        "silence_duration_ms": silence_duration_ms,
        "interrupt_response": True,
    }


def _build_voice_config(config: AzureVoiceLiveConfig) -> dict[str, Any] | None:
    if not config.tts_voice:
        return None
    temperature = 0.8
    if config.tts_temperature:
        with suppress(ValueError):
            temperature = float(config.tts_temperature)
    return {
        "name": config.tts_voice,
        "type": (config.tts_voice_type or "azure-standard").strip().lower(),
        "temperature": temperature,
    }


def _build_session_config(config: AzureVoiceLiveConfig) -> dict[str, Any]:
    sampling_rate = 24000
    if config.input_audio_sampling_rate:
        with suppress(ValueError):
            sampling_rate = int(config.input_audio_sampling_rate)

    session: dict[str, Any] = {
        "input_audio_sampling_rate": sampling_rate,
        "turn_detection": _build_turn_detection(config),
        "input_audio_noise_reduction": {"type": "azure_deep_noise_suppression"},
        "input_audio_echo_cancellation": {"type": "server_echo_cancellation"},
        "input_audio_transcription": {
            "model": "azure-speech",
            "language": config.transcription_languages,
        },
    }

    voice = _build_voice_config(config)
    if voice:
        session["voice"] = voice

    return session


def _build_response_config(config: AzureVoiceLiveConfig, modalities: list[str] | None = None) -> dict[str, Any]:
    resolved_modalities = modalities or ["text", "audio"]
    response: dict[str, Any] = {
        "modalities": resolved_modalities,
    }
    if "audio" in resolved_modalities:
        response["output_audio_format"] = "pcm16"
        response["output_audio_sampling_rate"] = config.output_audio_sampling_rate
    return response


def get_output_audio_sampling_rate() -> int:
    """Return the configured output audio sampling rate for client use."""
    config = AzureVoiceLiveConfig()
    return config.output_audio_sampling_rate


class VoiceLiveAgentConnection:
    def __init__(self, config: AzureVoiceLiveConfig) -> None:
        self.config = config
        self.ws: Any | None = None

    def _with_overrides(self, *, endpoint: str | None = None, api_version: str | None = None) -> VoiceLiveAgentConnection:
        cfg = AzureVoiceLiveConfig()
        cfg.endpoint = endpoint or self.config.endpoint
        cfg.api_version = api_version or self.config.api_version
        cfg.agent_project_name = self.config.agent_project_name
        cfg.agent_id = self.config.agent_id
        cfg.agent_connection_string = self.config.agent_connection_string
        cfg.api_key = self.config.api_key
        cfg.tts_voice = self.config.tts_voice
        cfg.tts_voice_type = self.config.tts_voice_type
        cfg.tts_temperature = self.config.tts_temperature
        cfg.vad_threshold = self.config.vad_threshold
        cfg.vad_prefix_padding_ms = self.config.vad_prefix_padding_ms
        cfg.vad_silence_duration_ms = self.config.vad_silence_duration_ms
        cfg.input_audio_sampling_rate = self.config.input_audio_sampling_rate
        cfg.output_audio_sampling_rate_s = self.config.output_audio_sampling_rate_s
        cfg.transcription_languages = self.config.transcription_languages
        cfg.open_timeout_s = self.config.open_timeout_s
        cfg.connect_retries = self.config.connect_retries
        cfg.retry_backoff_s = self.config.retry_backoff_s
        cfg.ping_interval_s = self.config.ping_interval_s
        cfg.ping_timeout_s = self.config.ping_timeout_s
        return VoiceLiveAgentConnection(cfg)

    async def __aenter__(self) -> VoiceLiveAgentConnection:
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        await self.close()

    async def _get_service_token(self) -> str:
        credential = AsyncDefaultAzureCredential()
        try:
            token = await credential.get_token(self.config.token_scope)
            return token.token
        finally:
            await credential.close()

    async def _get_agent_access_token(self) -> str:
        credential = AsyncDefaultAzureCredential()
        try:
            token = await credential.get_token(self.config.agent_token_scope)
            return token.token
        finally:
            await credential.close()

    def _get_websocket_url(self, query: dict[str, Any], path: str = "voice-live/realtime") -> str:
        azure_ws_endpoint = (
            self.config.resolve_endpoint().rstrip("/").replace("https://", "wss://")
        )
        return f"{azure_ws_endpoint}/{path}?{urlencode(query)}"

    async def _log_http_error_details(self, ws_url: str, headers: dict[str, str]) -> tuple[int | None, str | None]:
        try:
            http_url = ws_url.replace("wss://", "https://").replace("ws://", "http://")
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(http_url, headers=headers)
            logger.warning(
                "HTTP diagnostic for Voice Live: status=%s headers=%s body=%s",
                resp.status_code,
                resp.headers,
                resp.text,
            )
            return resp.status_code, resp.text
        except Exception as exc:  # noqa: BLE001
            logger.debug("HTTP diagnostic failed: %s", exc)
            return None, None

    async def connect(self) -> None:
        if self.ws:
            return
        # Build attempt list (endpoint, api_version)
        attempts: list[tuple[str | None, str | None]] = [(self.config.endpoint, self.config.api_version)]
        # If using cognitiveservices, add services.ai fallback with newer API version
        if self.config.endpoint and "cognitiveservices.azure.com" in self.config.endpoint:
            attempts.append(
                (
                    self.config.endpoint.replace("cognitiveservices.azure.com", "services.ai.azure.com"),
                    "2025-10-01",
                )
            )
        last_exc: Exception | None = None
        for ep_idx, (endpoint_override, api_version_override) in enumerate(attempts, start=1):
            cfg_conn = self if (ep_idx == 1) else self._with_overrides(endpoint=endpoint_override, api_version=api_version_override)
            service_token = await cfg_conn._get_service_token()
            agent_token = await cfg_conn._get_agent_access_token()
            modes = [
                ("agent", agent_token),
                ("model", None),
            ]
            for mode, mode_token in modes:
                for path_variant in ("voice-live/realtime", "voicelive/realtime"):
                    if mode == "agent":
                        query: dict[str, Any] = {
                            "api-version": cfg_conn.config.api_version,
                            "agent-id": cfg_conn.config.agent_id,
                            "agent-access-token": mode_token,
                        }
                        if cfg_conn.config.agent_connection_string:
                            query["agent-connection-string"] = cfg_conn.config.agent_connection_string
                        else:
                            query["agent-project-name"] = cfg_conn.config.agent_project_name
                        label = "agent"
                    else:
                        query = {
                            "api-version": cfg_conn.config.api_version,
                            "model": cfg_conn.config.fallback_model,
                        }
                        label = f"model={cfg_conn.config.fallback_model}"
                    if cfg_conn.config.api_key:
                        query["api-key"] = cfg_conn.config.api_key
                    ws_url = cfg_conn._get_websocket_url(query, path_variant)
                    if cfg_conn.config.api_version == "2025-05-01-preview":
                        logger.warning("Using preview API version 2025-05-01-preview; try 2025-10-01 if you see 400 errors.")
                    logger.debug(
                        "Token scopes: service=%s agent=%s path=%s mode=%s",
                        cfg_conn.config.token_scope,
                        cfg_conn.config.agent_token_scope,
                        path_variant,
                        label,
                    )
                    safe_ws_url = ws_url
                    if mode_token:
                        safe_ws_url = safe_ws_url.replace(mode_token, "<redacted>")
                    if cfg_conn.config.api_key:
                        safe_ws_url = safe_ws_url.replace(cfg_conn.config.api_key, "<redacted>")
                    logger.info(
                        "Connecting to Voice Live websocket: %s (endpoint=%s api_version=%s path=%s mode=%s attempt-set=%s/%s)",
                        safe_ws_url,
                        cfg_conn.config.endpoint,
                        cfg_conn.config.api_version,
                        path_variant,
                        label,
                        ep_idx,
                        len(attempts),
                    )
                    headers = {
                        "Authorization": f"Bearer {service_token}",
                        "x-ms-client-request-id": str(uuid.uuid4()),
                    }
                    for attempt in range(1, cfg_conn.config.retries + 1):
                        try:
                            self.ws = await websockets.connect(
                                ws_url,
                                extra_headers=headers,
                                max_size=10 * 1024 * 1024,
                                ping_interval=cfg_conn.config.ping_interval,
                                ping_timeout=cfg_conn.config.ping_timeout,
                                open_timeout=cfg_conn.config.open_timeout,
                            )
                            return
                        except (TimeoutError, OSError, websockets.WebSocketException) as exc:
                            last_exc = exc
                            extra = ""
                            if isinstance(exc, websockets.exceptions.InvalidStatusCode):
                                extra = f" status={exc.status_code} headers={getattr(exc, 'headers', None)}"
                                status_code, body = await cfg_conn._log_http_error_details(ws_url, headers)
                                if status_code == 404:
                                    raise RuntimeError(
                                        "Voice Live resource not found. Verify the endpoint (services.ai.azure.com), region supports Voice Live, and the resource exists."
                                    ) from exc
                            logger.warning(
                                "Voice Live websocket connect failed (attempt %s/%s; endpoint=%s api_version=%s path=%s mode=%s): %s%s",
                                attempt,
                                cfg_conn.config.retries,
                                cfg_conn.config.endpoint,
                                cfg_conn.config.api_version,
                                path_variant,
                                label,
                                exc,
                                extra,
                            )
                            if attempt < cfg_conn.config.retries:
                                await anyio.sleep(cfg_conn.config.retry_backoff * attempt)

        raise RuntimeError(
            "Voice Live websocket connection timed out or failed; verify endpoint,"
            " network access, API version, and Azure credentials."
        ) from last_exc

    async def close(self) -> None:
        if self.ws:
            await self.ws.close()
            self.ws = None

    def _generate_id(self, prefix: str) -> str:
        return f"{prefix}{int(time.time() * 1000)}"

    async def send_event(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        if not self.ws:
            raise RuntimeError("Voice Live API is not connected")
        payload: dict[str, Any] = {"event_id": self._generate_id("evt_"), "type": event_type}
        if data:
            payload.update(data)
        await self.ws.send(json.dumps(payload))

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        if not self.ws:
            raise StopAsyncIteration
        message = await self.ws.recv()
        if message is None:
            raise StopAsyncIteration
        return message


def connect_realtime() -> VoiceLiveAgentConnection:
    config = AzureVoiceLiveConfig()
    if not config.is_configured:
        raise RuntimeError("Azure Voice Live Agent is not configured.")
    return VoiceLiveAgentConnection(config)


async def configure_voice_live_session(target: VoiceLiveAgentConnection) -> None:
    config = AzureVoiceLiveConfig()
    session = _build_session_config(config)
    await target.send_event("session.update", {"session": session})


async def handle_client_message(target: VoiceLiveAgentConnection, message: str) -> None:
    try:
        payload = json.loads(message)
    except json.JSONDecodeError:
        return

    if not isinstance(payload, dict):
        return

    message_type = payload.get("type")
    data = payload.get("payload", payload)
    logger.debug("Client message type=%s", message_type)

    if message_type == "input_audio_buffer.append":
        audio = data.get("audio")
        if audio:
            try:
                raw_len = len(base64.b64decode(audio))
            except Exception:
                raw_len = int(len(audio) * 0.75)
            if not hasattr(handle_client_message, "audio_chunks"):
                handle_client_message.audio_chunks = 0
            handle_client_message.audio_chunks += 1
            if handle_client_message.audio_chunks % 20 == 1:
                logger.info(
                    "Received mic audio chunk #%s base64_len=%s bytes≈%s",
                    handle_client_message.audio_chunks,
                    len(audio),
                    raw_len,
                )
            await target.send_event("input_audio_buffer.append", {"audio": audio})
        return

    if message_type == "input_audio_buffer.commit":
        await target.send_event("input_audio_buffer.commit")
        return

    if message_type == "input_audio_buffer.clear":
        await target.send_event("input_audio_buffer.clear")
        return

    if message_type == "conversation.item.create":
        if isinstance(data, dict):
            await target.send_event("conversation.item.create", data)
        return

    if message_type == "response.create":
        config = AzureVoiceLiveConfig()
        response_data = _build_response_config(config)
        if isinstance(data, dict):
            incoming = data.get("response")
            if isinstance(incoming, dict):
                response_data = {**response_data, **incoming}
        await target.send_event("response.create", {"response": response_data})
        return

    if message_type == "response.cancel":
        await target.send_event("response.cancel")
        return


def serialize_server_event(event: Any, inject_sample_rate: int | None = None) -> str:
    if isinstance(event, str):
        try:
            parsed = json.loads(event)
            if isinstance(parsed, dict):
                event_type = parsed.get("type", "")
                # Inject sample rate into audio events for frontend compatibility
                if inject_sample_rate and (
                    event_type.startswith("response.audio")
                    or event_type.startswith("response.output_audio")
                ):
                    if "audio_sampling_rate" not in parsed:
                        parsed["audio_sampling_rate"] = inject_sample_rate
                    return json.dumps(parsed)
        except json.JSONDecodeError:
            pass
        return event
    if isinstance(event, dict):
        event_type = event.get("type", "")
        if (
            inject_sample_rate
            and (
                event_type.startswith("response.audio")
                or event_type.startswith("response.output_audio")
            )
            and "audio_sampling_rate" not in event
        ):
            event = {**event, "audio_sampling_rate": inject_sample_rate}
        return json.dumps(event)
    return json.dumps({"type": "unknown", "payload": event})


def _extract_text_delta(event: dict[str, Any]) -> str | None:
    event_type = event.get("type")
    if event_type in {
        "response.output_text.delta",
        "response.text.delta",
        "response.audio_transcript.delta",
    }:
        delta = event.get("delta")
        if isinstance(delta, str):
            return delta
        if isinstance(delta, dict):
            text = delta.get("text")
            if isinstance(text, str):
                return text

    if event_type in {
        "response.output_text",
        "response.text",
        "response.output_text.done",
        "response.text.done",
        "response.audio_transcript.done",
    }:
        text = event.get("text") or event.get("output_text")
        if isinstance(text, str):
            return text

    if event_type in {"response.content_part.added", "response.content_part.done"}:
        part = event.get("content_part") or event.get("part")
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text")
            if isinstance(text, str):
                return text

    if event_type in {"response.output_item.added", "response.output_item.done"}:
        item = event.get("item")
        if isinstance(item, dict):
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in {"text", "output_text"}:
                        text = part.get("text") or part.get("output_text")
                        if isinstance(text, str):
                            return text

    payload = event.get("payload")
    if isinstance(payload, dict):
        text = payload.get("text") or payload.get("output_text")
        if isinstance(text, str):
            return text
    return None


def _is_response_done(event: dict[str, Any]) -> bool:
    return event.get("type") in {
        "response.done",
        "response.completed",
        "response.output_text.done",
        "response.text.done",
        "response.audio_transcript.done",
        "response.output_item.done",
        "response.content_part.done",
    }


def _extract_text_delta_only(event: dict[str, Any]) -> str | None:
    event_type = event.get("type")
    if event_type in {
        "response.output_text.delta",
        "response.text.delta",
        "response.audio_transcript.delta",
    }:
        delta = event.get("delta")
        if isinstance(delta, str):
            return delta
        if isinstance(delta, dict):
            text = delta.get("text")
            if isinstance(text, str):
                return text

    if event_type == "response.content_part.added":
        part = event.get("content_part") or event.get("part")
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text")
            if isinstance(text, str):
                return text

    if event_type == "response.output_item.added":
        item = event.get("item")
        if isinstance(item, dict):
            content = item.get("content")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") in {"text", "output_text"}:
                        text = part.get("text") or part.get("output_text")
                        if isinstance(text, str):
                            return text
    return None


async def fetch_agent_text_reply(
    prompt: str,
    transcript_enabled: bool = True,
    timeout_s: float = 20.0,
) -> str:
    # transcript_enabled is retained for compatibility; transcripts are always requested/returned
    async with connect_realtime() as azure_conn:
        await configure_voice_live_session(azure_conn)
        config = AzureVoiceLiveConfig()

        logger.info("Sending prompt to Voice Live agent")
        await azure_conn.send_event(
            "conversation.item.create",
            {
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            },
        )
        modalities = ["text"]
        await azure_conn.send_event(
            "response.create",
            {"response": _build_response_config(config, modalities)},
        )

        chunks: list[str] = []

        try:
            with anyio.fail_after(timeout_s):
                async for raw in azure_conn:
                    if isinstance(raw, str):
                        logger.debug("Voice Live event: %s", raw)
                    try:
                        event = json.loads(raw) if isinstance(raw, str) else raw
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(event, dict):
                        continue

                    delta = _extract_text_delta(event)
                    if delta:
                        chunks.append(delta)

                    if _is_response_done(event):
                        logger.info("Voice Live response completed")
                        break
        except TimeoutError:
            logger.warning("Voice Live response timed out")
            pass

        reply = "".join(chunks).strip()
        if not reply:
            logger.warning("Voice Live response completed with no text output")
        return reply


async def stream_agent_text_reply(
    prompt: str,
    timeout_s: float = 20.0,
    modalities: list[str] | None = None,
) -> AsyncIterator[str]:
    async with connect_realtime() as azure_conn:
        await configure_voice_live_session(azure_conn)
        config = AzureVoiceLiveConfig()

        logger.info("Sending prompt to Voice Live agent (streaming)")
        await azure_conn.send_event(
            "conversation.item.create",
            {
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}],
                }
            },
        )
        await azure_conn.send_event(
            "response.create",
            {"response": _build_response_config(config, modalities or ["text"])},
        )

        try:
            with anyio.fail_after(timeout_s):
                async for raw in azure_conn:
                    if isinstance(raw, str):
                        logger.debug("Voice Live event: %s", raw)
                    try:
                        event = json.loads(raw) if isinstance(raw, str) else raw
                    except json.JSONDecodeError:
                        continue

                    if not isinstance(event, dict):
                        continue

                    delta = _extract_text_delta_only(event)
                    if delta:
                        yield delta

                    if _is_response_done(event):
                        logger.info("Voice Live response completed (streaming)")
                        break
        except TimeoutError:
            logger.warning("Voice Live response timed out (streaming)")
            return


async def pump_messages(
    source: AsyncIterator[str],
    target: Any,
) -> None:
    async for message in source:
        await handle_client_message(target, message)
