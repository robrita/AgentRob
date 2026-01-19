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
    AzureVoiceLiveConfig,
    configure_voice_live_session,
    connect_realtime,
    fetch_agent_text_reply,
    get_output_audio_sampling_rate,
    handle_client_message,
    serialize_server_event,
    stream_agent_text_reply,
)
from .services.personal_voice_tts import (
    PersonalVoiceConfig,
    PersonalVoiceSynthesizer,
    SentenceBuffer,
    is_personal_voice_enabled,
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

    # Check if Personal Voice hybrid mode is enabled
    use_personal_voice = is_personal_voice_enabled()
    personal_voice_config = PersonalVoiceConfig.from_env() if use_personal_voice else None
    pv_synthesizer = PersonalVoiceSynthesizer(personal_voice_config) if personal_voice_config else None

    if use_personal_voice and pv_synthesizer:
        logger.info(
            "Personal Voice hybrid mode ENABLED - Voice Live text-only, Speech SDK for audio"
        )
    else:
        logger.info("Standard Voice Live mode - Azure handles both text and audio")

    try:
        async with connect_realtime() as azure_conn:
            await configure_voice_live_session(azure_conn)

            # Shared state for Personal Voice synthesis
            sentence_buffer = SentenceBuffer(min_chars=5) if pv_synthesizer else None
            synthesis_queue: anyio.abc.ObjectSendStream | None = None
            synthesis_receive: anyio.abc.ObjectReceiveStream | None = None
            # Track synthesis state to prevent echo loops
            is_synthesizing = False
            # Don't send mic audio until session is ready
            session_ready = False
            # Track pending sentences to synthesize
            pending_sentences = 0
            # Lock for pending_sentences counter
            pending_lock = anyio.Lock()
            # Cooldown timestamp - block audio until this time passes
            synthesis_cooldown_until = 0.0
            # Track if current response has queued any sentences
            response_has_content = False
            # Flag to clear buffer on first audio after cooldown
            need_post_cooldown_clear = False
            # Grace period - cancel responses even after cooldown for extra safety
            response_grace_until = 0.0
            # Flag to clear buffer when grace period ends
            need_post_grace_clear = False
            # Track if we've received actual user speech (transcription)
            user_has_spoken = False
            # Track if we've already cancelled responses in the current "no speech" period
            cancelled_for_no_speech = False
            # Track if current response was accepted (not cancelled/ignored)
            current_response_valid = False
            
            if pv_synthesizer:
                # create_memory_object_stream returns (send_stream, receive_stream)
                # First positional arg is max_buffer_size
                synthesis_queue, synthesis_receive = anyio.create_memory_object_stream(20)

            async def client_to_azure() -> None:
                nonlocal is_synthesizing, session_ready, synthesis_cooldown_until
                nonlocal need_post_cooldown_clear, response_grace_until, need_post_grace_clear
                try:
                    while True:
                        message = await websocket.receive_text()
                        should_forward = True
                        
                        try:
                            import time
                            payload = json.loads(message)
                            message_type = payload.get("type")
                            
                            # In hybrid mode, block messages based on state
                            if pv_synthesizer:
                                current_time = time.time()
                                in_cooldown = current_time < synthesis_cooldown_until
                                in_grace = current_time < response_grace_until
                                
                                # Block all audio messages until session is ready
                                if not session_ready:
                                    if message_type in ("input_audio_buffer.append", "input_audio_buffer.commit", "response.create"):
                                        should_forward = False
                                
                                # During synthesis: ALLOW audio (for user interrupts) but response cancellation handles echo
                                # During cooldown: BLOCK audio (echo from speaker after synthesis)
                                elif in_cooldown:
                                    if message_type == "input_audio_buffer.append":
                                        should_forward = False
                                
                                # First audio after cooldown - clear buffer once
                                elif need_post_cooldown_clear and message_type == "input_audio_buffer.append":
                                    need_post_cooldown_clear = False
                                    logger.info("Post-cooldown: clearing buffer before resuming")
                                    await azure_conn.send_event("input_audio_buffer.clear")
                                    should_forward = False  # Don't forward this chunk
                                
                                # When grace period ends, clear buffer again and set flag
                                elif need_post_grace_clear and not in_grace and message_type == "input_audio_buffer.append":
                                    need_post_grace_clear = False
                                    logger.info("Post-grace: clearing buffer before full resume")
                                    await azure_conn.send_event("input_audio_buffer.clear")
                                    should_forward = False  # Don't forward this chunk
                                
                                if message_type not in ("input_audio_buffer.append",):
                                    logger.debug("Client msg type=%s forward=%s ready=%s synth=%s cooldown=%s grace=%s", 
                                                message_type, should_forward, session_ready, is_synthesizing, in_cooldown, in_grace)
                            else:
                                if message_type != "input_audio_buffer.append":
                                    logger.debug("Client message type=%s", message_type)
                                    
                        except Exception:
                            logger.debug("Client message non-json")
                        
                        if should_forward:
                            await handle_client_message(azure_conn, message)
                except WebSocketDisconnect:
                    await azure_conn.close()
                except Exception:  # noqa: BLE001
                    await azure_conn.close()

            async def azure_to_client() -> None:
                """Forward events from Azure to client, with Personal Voice TTS if enabled."""
                import base64
                import time
                nonlocal sentence_buffer, synthesis_queue, is_synthesizing, session_ready
                nonlocal pending_sentences, synthesis_cooldown_until, response_has_content
                nonlocal response_grace_until, need_post_grace_clear, user_has_spoken
                nonlocal cancelled_for_no_speech, current_response_valid

                try:
                    async for event in azure_conn:
                        try:
                            parsed = json.loads(event) if isinstance(event, str) else event
                        except json.JSONDecodeError:
                            await websocket.send_text(serialize_server_event(event, inject_sample_rate=output_sample_rate))
                            continue

                        if not isinstance(parsed, dict):
                            await websocket.send_text(serialize_server_event(event, inject_sample_rate=output_sample_rate))
                            continue

                        etype = parsed.get("type", "")

                        # Log key events
                        if etype in ("session.updated", "session.created"):
                            logger.info("Azure event: %s - %s", etype, json.dumps(parsed)[:500])
                        elif etype == "error":
                            # Log errors but don't crash on harmless ones
                            error_code = parsed.get("error", {}).get("code", "")
                            if error_code in ("response_cancel_not_active", "conversation_already_has_active_response"):
                                logger.debug("Ignoring harmless error: %s", error_code)
                            else:
                                logger.warning("Azure error: %s", json.dumps(parsed)[:500])
                            continue  # Don't forward errors to client
                        elif etype.startswith("response."):
                            logger.debug("Azure response event: %s", etype)
                        
                        # Detect user transcription - mark that user has actually spoken
                        # Only set this when we get actual transcription text, not just speech_started
                        if pv_synthesizer and etype == "conversation.item.input_audio_transcription.completed":
                            transcript = parsed.get("transcript", "")
                            # Only count as "spoken" if there's actual text content
                            if isinstance(transcript, str) and transcript.strip():
                                logger.info("User speech transcribed: '%s'", transcript[:50])
                                
                                # If user interrupts during synthesis, cancel current response
                                if is_synthesizing:
                                    logger.info("User interrupt detected - cancelling current response")
                                    await azure_conn.send_event("response.cancel")
                                    # Don't reset is_synthesizing here - let synthesis task finish gracefully
                                
                                user_has_spoken = True
                                cancelled_for_no_speech = False  # Reset cancel flag for next turn
                                
                                # If no response is currently active/valid, request one now
                                # This handles the case where VAD triggered before transcription completed
                                if not current_response_valid and not is_synthesizing:
                                    logger.info("Requesting response after transcription")
                                    await azure_conn.send_event("response.create")

                        # Mark session ready after session.updated (enables mic audio)
                        if pv_synthesizer and etype == "session.updated":
                            # Clear any audio that was buffered during connection setup
                            await azure_conn.send_event("input_audio_buffer.clear")
                            logger.info("Cleared Azure input buffer after session update")
                            session_ready = True
                            logger.info("Session ready - mic audio and commits now enabled")
                            await websocket.send_text(serialize_server_event(event, inject_sample_rate=output_sample_rate))
                            continue

                        # Personal Voice hybrid mode: intercept transcript and synthesize audio
                        if pv_synthesizer and sentence_buffer and synthesis_queue:
                            # When response starts, suppress mic to prevent echo
                            if etype == "response.created":
                                import time
                                current_time = time.time()
                                
                                # If we're synthesizing, in cooldown, OR in grace period - cancel
                                in_cooldown = current_time < synthesis_cooldown_until
                                in_grace = current_time < response_grace_until
                                
                                # Also cancel if user hasn't spoken yet (proactive greeting prevention)
                                # But only send cancel once to avoid flooding Azure
                                if not user_has_spoken:
                                    if not cancelled_for_no_speech:
                                        logger.info("Cancelling response - waiting for user to speak first")
                                        await azure_conn.send_event("response.cancel")
                                        cancelled_for_no_speech = True
                                    current_response_valid = False
                                    # Don't forward and don't clear buffer repeatedly
                                    continue
                                
                                if is_synthesizing or in_cooldown or in_grace:
                                    logger.warning("Cancelling echo-triggered response (synth=%s cooldown=%s grace=%s)", 
                                                  is_synthesizing, in_cooldown, in_grace)
                                    await azure_conn.send_event("response.cancel")
                                    await azure_conn.send_event("input_audio_buffer.clear")
                                    current_response_valid = False
                                    continue  # Don't forward this response.created
                                
                                # This is a valid response - accept it
                                current_response_valid = True
                                is_synthesizing = True
                                response_has_content = False  # Reset for new response
                                logger.info("Response started - suppressing mic input")
                                # Clear any buffered audio to prevent echo processing
                                await azure_conn.send_event("input_audio_buffer.clear")
                                await websocket.send_text(serialize_server_event(event, inject_sample_rate=output_sample_rate))
                                continue

                            # When response is done, check if we need to reset state
                            if etype == "response.done":
                                logger.info("Response done")
                                was_valid = current_response_valid
                                current_response_valid = False  # Reset for next response
                                # If response had no content (empty response), reset synthesis state immediately
                                if not response_has_content and pending_sentences == 0:
                                    logger.info("Empty response - resetting synthesis state")
                                    is_synthesizing = False
                                # Only forward if it was a valid response
                                if was_valid:
                                    await websocket.send_text(serialize_server_event(event, inject_sample_rate=output_sample_rate))
                                continue

                            # Intercept text events and synthesize audio
                            # In text-only mode, we get response.text.delta instead of audio_transcript.delta
                            if etype in ("response.audio_transcript.delta", "response.text.delta", "response.output_text.delta"):
                                delta = parsed.get("delta", "")
                                # Only queue for synthesis if this is a valid response we accepted
                                if isinstance(delta, str) and delta and current_response_valid:
                                    response_has_content = True  # Mark that we have content
                                    # Buffer text and queue complete sentences for synthesis
                                    sentences = sentence_buffer.add(delta)
                                    for sentence in sentences:
                                        logger.info("Queueing sentence for Personal Voice: %s...", sentence[:40])
                                        async with pending_lock:
                                            pending_sentences += 1
                                        await synthesis_queue.send(sentence)
                                # Only forward text delta to client if valid response
                                if current_response_valid:
                                    await websocket.send_text(serialize_server_event(event, inject_sample_rate=output_sample_rate))
                                continue

                            elif etype in ("response.audio_transcript.done", "response.text.done", "response.output_text.done"):
                                # Only process if this is a valid response
                                if current_response_valid:
                                    # Flush any remaining buffered text
                                    remaining = sentence_buffer.flush()
                                    if remaining:
                                        response_has_content = True  # Mark that we have content
                                        logger.info("Queueing final sentence for Personal Voice: %s...", remaining[:40])
                                        async with pending_lock:
                                            pending_sentences += 1
                                        await synthesis_queue.send(remaining)
                                    # Forward event to client
                                    await websocket.send_text(serialize_server_event(event, inject_sample_rate=output_sample_rate))
                                continue

                            elif etype == "response.audio.delta":
                                # Suppress Voice Live audio.delta in hybrid mode (we generate our own)
                                logger.debug("Suppressing Voice Live audio.delta in hybrid mode")
                                continue

                            elif etype == "response.audio.done":
                                # Suppress Voice Live audio.done in hybrid mode
                                logger.debug("Suppressing Voice Live audio.done in hybrid mode")
                                continue

                        # Standard mode or non-audio events: forward to client
                        await websocket.send_text(serialize_server_event(event, inject_sample_rate=output_sample_rate))

                except Exception as e:  # noqa: BLE001
                    logger.warning("Error in azure_to_client: %s", e)
                    with suppress(Exception):
                        await websocket.close(code=1011)

            async def personal_voice_synthesizer_task() -> None:
                """Background task that synthesizes sentences to audio using Personal Voice."""
                import base64
                import time
                nonlocal synthesis_receive, is_synthesizing, pending_sentences
                nonlocal synthesis_cooldown_until, need_post_cooldown_clear
                nonlocal response_grace_until, need_post_grace_clear, user_has_spoken
                nonlocal cancelled_for_no_speech

                if not pv_synthesizer or not synthesis_receive:
                    return

                try:
                    async for sentence in synthesis_receive:
                        if not sentence:
                            async with pending_lock:
                                pending_sentences = max(0, pending_sentences - 1)
                            continue

                        logger.info("Synthesizing Personal Voice: %s...", sentence[:50] if len(sentence) > 50 else sentence)

                        try:
                            # Synthesize and stream audio chunks to client
                            chunk_count = 0
                            async for audio_chunk in pv_synthesizer.synthesize_to_chunks(sentence):
                                chunk_count += 1
                                # Create audio delta event matching Voice Live format
                                audio_event = {
                                    "type": "response.audio.delta",
                                    "delta": base64.b64encode(audio_chunk).decode("ascii"),
                                    "audio_sampling_rate": 24000,
                                }
                                await websocket.send_text(json.dumps(audio_event))

                            logger.info("Personal Voice synthesis complete: %d chunks sent", chunk_count)

                        except Exception as synth_err:  # noqa: BLE001
                            logger.exception("Personal Voice synthesis error: %s", synth_err)
                        finally:
                            # Decrement pending counter
                            async with pending_lock:
                                pending_sentences = max(0, pending_sentences - 1)
                                
                                # If no more pending sentences, synthesis is truly done
                                if pending_sentences == 0:
                                    logger.info("All synthesis complete - setting cooldown and clearing buffer")
                                    # Set cooldown period - block audio for 8 seconds
                                    synthesis_cooldown_until = time.time() + 8.0
                                    # Set grace period - cancel responses for 15 seconds total
                                    response_grace_until = time.time() + 15.0
                                    need_post_cooldown_clear = True
                                    need_post_grace_clear = True
                                    # Reset user_has_spoken so next response requires new speech
                                    user_has_spoken = False
                                    cancelled_for_no_speech = False  # Allow next cancel cycle
                                    # Clear any audio that was buffered during synthesis
                                    await azure_conn.send_event("input_audio_buffer.clear")
                                    is_synthesizing = False
                                    logger.info("Mic re-enabled (cooldown 8s, grace period 15s, waiting for new speech)")

                except anyio.EndOfStream:
                    logger.debug("Personal Voice synthesis queue closed")
                except Exception as e:  # noqa: BLE001
                    logger.exception("Error in personal_voice_synthesizer_task: %s", e)
                finally:
                    is_synthesizing = False

            async with anyio.create_task_group() as tg:
                tg.start_soon(client_to_azure)
                tg.start_soon(azure_to_client)
                if pv_synthesizer:
                    tg.start_soon(personal_voice_synthesizer_task)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Realtime websocket error for session %s: %s", session_id, exc)
        with suppress(Exception):
            if websocket.application_state.name != "DISCONNECTED":
                await websocket.send_text(
                    json.dumps({"type": "error", "detail": str(exc)})
                )
        with suppress(Exception):
            await websocket.close(code=1011)
        return
