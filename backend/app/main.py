from __future__ import annotations

import json
import logging
import os
from contextlib import suppress

import anyio
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .models import (
    MessageRequest,
    MessageResponse,
    MicRequest,
    PauseRequest,
    SessionCreateResponse,
    SessionState,
    ToggleRequest,
    TranscriptItem,
)
from .services.azure_voice_live import (
    configure_voice_live_session,
    connect_realtime,
    fetch_agent_text_reply,
    get_output_audio_sampling_rate,
    handle_client_message,
    serialize_server_event,
    stream_agent_text_reply,
)
from .store import SessionStore

load_dotenv()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("agentrob")

app = FastAPI(title="AgentRob Backend", version="1.0.0")
store = SessionStore()

cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"] ,
)


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/api/session", response_model=SessionCreateResponse)
async def create_session() -> SessionCreateResponse:
    session = store.create()
    return SessionCreateResponse(
        session_id=session.id,
        ws_url=f"/api/session/{session.id}/realtime",
    )


@app.get("/api/session/{session_id}", response_model=SessionState)
async def get_session(session_id: str) -> SessionState:
    session = store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.post("/api/session/{session_id}/mic", response_model=SessionState)
async def update_mic(session_id: str, payload: MicRequest) -> SessionState:
    session = store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session.mic_listening = payload.listening
    store.update(session)
    return session


@app.post("/api/session/{session_id}/pause", response_model=SessionState)
async def update_pause(session_id: str, payload: PauseRequest) -> SessionState:
    session = store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session.paused = payload.paused
    store.update(session)
    return session


@app.post("/api/session/{session_id}/sounds", response_model=SessionState)
async def update_sounds(session_id: str, payload: ToggleRequest) -> SessionState:
    session = store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    session.sounds_on = payload.enabled
    store.update(session)
    return session


@app.post("/api/session/{session_id}/leave", response_model=SessionState)
async def leave_session(session_id: str) -> SessionState:
    session = store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    store.end(session_id)
    session = store.get(session_id)
    return session


@app.get("/api/session/{session_id}/transcript", response_model=list[TranscriptItem])
async def get_transcript(session_id: str) -> list[TranscriptItem]:
    session = store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.transcript


@app.post("/api/session/{session_id}/message", response_model=MessageResponse)
async def post_message(session_id: str, payload: MessageRequest) -> MessageResponse:
    session = store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    user_item = TranscriptItem(author=payload.author, text=payload.text)
    store.append_transcript(session_id, user_item)

    if payload.respond and not session.paused:
        try:
            reply_text = await fetch_agent_text_reply(payload.text)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent response failed for session %s", session_id)
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        if reply_text:
            store.append_transcript(
                session_id, TranscriptItem(author="AgentRob", text=reply_text)
            )

    session = store.get(session_id)
    return MessageResponse(transcript=session.transcript if session else [])


@app.post("/api/session/{session_id}/message/stream")
async def post_message_stream(session_id: str, payload: MessageRequest) -> StreamingResponse:
    session = store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    user_item = TranscriptItem(author=payload.author, text=payload.text)
    store.append_transcript(session_id, user_item)

    async def event_stream() -> anyio.AsyncIterator[str]:
        if not payload.respond or session.paused:
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        buffer: list[str] = []
        try:
            async for delta in stream_agent_text_reply(payload.text, modalities=["text"]):
                buffer.append(delta)
                yield f"data: {json.dumps({'type': 'delta', 'text': delta})}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent streaming response failed for session %s", session_id)
            yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"
            return

        reply_text = "".join(buffer).strip()
        if not reply_text:
            try:
                reply_text = await fetch_agent_text_reply(payload.text, timeout_s=40.0)
                if reply_text:
                    yield f"data: {json.dumps({'type': 'delta', 'text': reply_text})}\n\n"
            except Exception as exc:  # noqa: BLE001
                logger.exception("Agent fallback response failed for session %s", session_id)
                yield f"data: {json.dumps({'type': 'error', 'detail': str(exc)})}\n\n"
                return
        if reply_text:
            store.append_transcript(
                session_id, TranscriptItem(author="AgentRob", text=reply_text)
            )

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


@app.websocket("/api/session/{session_id}/realtime")
async def realtime(session_id: str, websocket: WebSocket) -> None:
    session = store.get(session_id)
    if not session:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    logger.info("Realtime websocket connected session %s", session_id)

    # Get the configured output audio sample rate to inject into events
    output_sample_rate = get_output_audio_sampling_rate()

    # Keep-alive interval in seconds (send ping every 2 minutes to prevent Azure idle timeout)
    KEEPALIVE_INTERVAL = 120

    try:
        async with connect_realtime() as azure_conn:
            await configure_voice_live_session(azure_conn)

            async def client_to_azure() -> None:
                try:
                    while True:
                        message = await websocket.receive_text()
                        try:
                            payload = json.loads(message)
                            message_type = payload.get("type")
                            if message_type == "input_audio_buffer.append":
                                audio = payload.get("payload", {}).get("audio") or payload.get("audio")
                                if isinstance(audio, str):
                                    logger.debug("Client audio chunk base64_len=%s", len(audio))
                            else:
                                logger.debug("Client message type=%s", message_type)
                        except Exception:
                            logger.debug("Client message non-json")
                        await handle_client_message(azure_conn, message)
                except WebSocketDisconnect:
                    await azure_conn.close()
                except Exception:  # noqa: BLE001
                    await azure_conn.close()

            async def azure_to_client() -> None:
                try:
                    async for event in azure_conn:
                        try:
                            parsed = json.loads(event) if isinstance(event, str) else event
                            if isinstance(parsed, dict):
                                etype = parsed.get("type")
                                if etype:
                                    logger.debug("Azure event type=%s", etype)
                                    if etype.startswith("response.audio") or etype.startswith("response.output_audio"):
                                        delta = parsed.get("delta") or parsed.get("audio")
                                        if isinstance(delta, str):
                                            logger.debug("Azure audio delta len=%s", len(delta))
                        except Exception:
                            pass
                        # Inject sample rate into audio events for frontend compatibility
                        await websocket.send_text(serialize_server_event(event, inject_sample_rate=output_sample_rate))
                except Exception:  # noqa: BLE001
                    await websocket.close(code=1011)

            async def keepalive() -> None:
                """Send periodic empty audio buffer to prevent Azure idle timeout."""
                try:
                    while True:
                        await anyio.sleep(KEEPALIVE_INTERVAL)
                        # Send a minimal silent audio chunk to keep connection alive
                        # This is a tiny PCM16 silent sample (4 bytes = 2 samples at 0)
                        import base64
                        silent_audio = base64.b64encode(b"\x00\x00\x00\x00").decode()
                        await azure_conn.send_event("input_audio_buffer.append", {"audio": silent_audio})
                        logger.debug("Sent keepalive ping to Azure Voice Live")
                except Exception:  # noqa: BLE001
                    pass  # Keepalive failure is not fatal

            async with anyio.create_task_group() as tg:
                tg.start_soon(client_to_azure)
                tg.start_soon(azure_to_client)
                tg.start_soon(keepalive)
    except Exception as exc:  # noqa: BLE001
        try:
            if websocket.application_state.name != "DISCONNECTED":
                await websocket.send_text(
                    json.dumps({"type": "error", "detail": str(exc)})
                )
        except RuntimeError:
            pass
        finally:
            with suppress(Exception):
                await websocket.close(code=1011)
        return
