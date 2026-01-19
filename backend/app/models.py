from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TranscriptItem(BaseModel):
    author: str
    text: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SessionState(BaseModel):
    id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    mic_listening: bool = True
    paused: bool = False
    sounds_on: bool = True
    active: bool = True
    transcript: list[TranscriptItem] = Field(default_factory=list)


class SessionCreateResponse(BaseModel):
    session_id: str
    ws_url: str


class ToggleRequest(BaseModel):
    enabled: bool


class MicRequest(BaseModel):
    listening: bool


class PauseRequest(BaseModel):
    paused: bool


class MessageRequest(BaseModel):
    text: str
    author: str = "You"
    respond: bool = True
    transcript_enabled: bool = True  # UI toggle; backend always stores transcripts


class MessageResponse(BaseModel):
    transcript: list[TranscriptItem]


class ErrorResponse(BaseModel):
    detail: str


class RealtimeEvent(BaseModel):
    type: Literal["text", "audio", "control"]
    payload: dict
