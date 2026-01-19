from __future__ import annotations

import threading
import uuid

from .models import SessionState, TranscriptItem


class SessionStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionState] = {}

    def create(self) -> SessionState:
        with self._lock:
            session_id = str(uuid.uuid4())
            state = SessionState(id=session_id)
            self._sessions[session_id] = state
            return state

    def get(self, session_id: str) -> SessionState | None:
        with self._lock:
            return self._sessions.get(session_id)

    def update(self, session: SessionState) -> None:
        with self._lock:
            self._sessions[session.id] = session

    def append_transcript(self, session_id: str, item: TranscriptItem) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.transcript.append(item)

    def end(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.active = False
                session.mic_listening = False
                session.paused = False
