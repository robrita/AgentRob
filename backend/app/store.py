from __future__ import annotations

import threading
import time
import uuid

from .models import SessionState, TranscriptItem

# Default session TTL: 30 minutes
DEFAULT_SESSION_TTL_SECONDS = 30 * 60


class SessionStore:
    def __init__(self, ttl_seconds: int = DEFAULT_SESSION_TTL_SECONDS) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionState] = {}
        self._last_access: dict[str, float] = {}
        self._ttl_seconds = ttl_seconds

    def _cleanup_expired(self) -> None:
        """Remove sessions that haven't been accessed within TTL. Called with lock held."""
        if self._ttl_seconds <= 0:
            return
        now = time.time()
        expired = [
            sid for sid, last in self._last_access.items()
            if now - last > self._ttl_seconds
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._last_access.pop(sid, None)

    def _touch(self, session_id: str) -> None:
        """Update last access time for a session. Called with lock held."""
        self._last_access[session_id] = time.time()

    def create(self) -> SessionState:
        with self._lock:
            self._cleanup_expired()
            session_id = str(uuid.uuid4())
            state = SessionState(id=session_id)
            self._sessions[session_id] = state
            self._touch(session_id)
            return state

    def get(self, session_id: str) -> SessionState | None:
        with self._lock:
            self._cleanup_expired()
            session = self._sessions.get(session_id)
            if session:
                self._touch(session_id)
            return session

    def update(self, session: SessionState) -> None:
        with self._lock:
            self._sessions[session.id] = session
            self._touch(session.id)

    def append_transcript(self, session_id: str, item: TranscriptItem) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            session.transcript.append(item)
            self._touch(session_id)

    def end(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.active = False
                session.mic_listening = False
                session.paused = False
                self._touch(session_id)
