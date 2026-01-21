import "./styles.css";
import { App } from "./components/App.js";
import { Filters, setupFilters } from "./components/Filter.js";
import { renderData } from "./components/Grid.js";
import { DEFAULT_SESSION_STATE, STORAGE_KEYS } from "./utils/constants.js";
import { icons } from "./utils/icons.js";

const API_BASE = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(
  /\/$/,
  ""
);
const API_ROOT = `${API_BASE}/api`;

const appRoot = document.querySelector("#app");
appRoot.innerHTML = App();
const appContainer = document.querySelector(".app");

const audioVizCanvas = document.getElementById("audioViz");
let audioVizCtx = null;
let audioVizRaf = null;
let audioVizGradient = null;

const setupAudioVisualizer = () => {
  if (!audioVizCanvas) return;
  audioVizCtx = audioVizCanvas.getContext("2d");
  if (!audioVizCtx) return;

  const resize = () => {
    const { width, height } = audioVizCanvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    audioVizCanvas.width = Math.max(1, Math.round(width * dpr));
    audioVizCanvas.height = Math.max(1, Math.round(height * dpr));
    audioVizCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    audioVizGradient = audioVizCtx.createLinearGradient(0, 0, width, 0);
    audioVizGradient.addColorStop(0, "rgba(255, 255, 255, 0.15)");
    audioVizGradient.addColorStop(0.45, "rgba(255, 255, 255, 0.9)");
    audioVizGradient.addColorStop(0.55, "rgba(255, 255, 255, 0.9)");
    audioVizGradient.addColorStop(1, "rgba(255, 255, 255, 0.15)");
  };

  window.addEventListener("resize", resize, { passive: true });
  resize();

  const draw = () => {
    if (!audioVizCtx || !audioVizCanvas) {
      audioVizRaf = requestAnimationFrame(draw);
      return;
    }

    const { width, height } = audioVizCanvas.getBoundingClientRect();
    audioVizCtx.clearRect(0, 0, width, height);

    if (playbackAnalyser && playbackAnalyserData) {
      playbackAnalyser.getByteFrequencyData(playbackAnalyserData);
      const barCount = 40;
      const step = Math.max(1, Math.floor(playbackAnalyserData.length / barCount));
      const barWidth = width / barCount;
      const maxHeight = height * 0.8;
      const centerY = height * 0.5;

      audioVizCtx.fillStyle = audioVizGradient || "rgba(255, 255, 255, 0.9)";
      for (let i = 0; i < barCount; i += 1) {
        const value = playbackAnalyserData[i * step] / 255;
        const barHeight = Math.max(6, value * maxHeight);
        const x = i * barWidth + barWidth * 0.2;
        const w = Math.max(2, barWidth * 0.6);
        const y = centerY - barHeight / 2;
        const radius = Math.min(10, w / 2, barHeight / 2);
        audioVizCtx.beginPath();
        if (audioVizCtx.roundRect) {
          audioVizCtx.roundRect(x, y, w, barHeight, radius);
        } else {
          audioVizCtx.rect(x, y, w, barHeight);
        }
        audioVizCtx.fill();
      }
    }

    audioVizRaf = requestAnimationFrame(draw);
  };

  if (audioVizRaf) cancelAnimationFrame(audioVizRaf);
  audioVizRaf = requestAnimationFrame(draw);
};

const transcriptBody = document.getElementById("transcriptBody");
const transcriptFilter = document.getElementById("transcriptFilter");

if (transcriptFilter) {
  transcriptFilter.innerHTML = Filters();
}

const transcriptData = [];

const { applyFilters } = setupFilters({
  data: transcriptData,
  onFiltered: (filtered) => renderData(filtered, transcriptBody),
});

applyFilters();

const sessionState = { ...DEFAULT_SESSION_STATE };
sessionStorage.removeItem(STORAGE_KEYS.sessionId);
let sessionId = null;

const micButton = document.getElementById("micButton");
const micStatus = document.getElementById("micStatus");
const pauseButton = document.getElementById("pauseButton");
const leaveButton = document.getElementById("leaveButton");
const transcriptPanel = document.getElementById("transcriptPanel");
const transcriptToggle = document.getElementById("transcriptToggle");
const transcriptClose = document.getElementById("transcriptClose");
const soundToggle = document.getElementById("soundToggle");
const messageForm = document.getElementById("messageForm");
const messageInput = document.getElementById("messageInput");
const loadingScreen = document.getElementById("loadingScreen");
const loadingTitle = loadingScreen?.querySelector(".loading-title");
const loadingSubtitle = loadingScreen?.querySelector(".loading-subtitle");
const sendButton = messageForm?.querySelector("button[type=submit]");

const AUDIO_TARGET_SAMPLE_RATE = 24000;
let micStream = null;
let micAudioContext = null;
let micProcessor = null;
let micSocket = null;
let micSocketReady = null;
let lastVoiceTranscript = "";
let lastVoiceTranscriptAt = 0;
let micStopping = false;
let micCommitInterval = null;
let micLastCommitTs = 0;
let micResponding = false;
let micRespondingAt = 0;
let micChunkCount = 0;
let micLastLogTs = 0;

// Speech detection state for robust interrupt triggering
let speechFrameCount = 0; // Count of consecutive frames above threshold
const SPEECH_FRAMES_REQUIRED = 4; // Require multiple frames of speech before interrupt
const SPEECH_RMS_THRESHOLD = 0.04; // Higher threshold to avoid ambient noise triggers
let lastSpeechDetectedAt = 0; // Track when we last detected speech
let wasMicListeningBeforePause = false;
let isBackendReady = false;
let hasSentMicAudio = false;
let firstAudioChunkReady = null;
let resolveFirstAudioChunk = null;
const micLog = (...args) => console.debug("[mic]", ...args);

// Module-level agent message state for realtime WebSocket (shared by voice + text)
let realtimeAgentIndex = null;
let realtimeAgentText = "";

const ensureRealtimeAgentMessage = () => {
  if (realtimeAgentIndex !== null) return;
  transcriptData.push({ author: "AgentRob", text: "" });
  realtimeAgentIndex = transcriptData.length - 1;
  if (typeof applyFilters === "function") {
    applyFilters();
  } else {
    renderData(transcriptData, transcriptBody);
  }
};

const updateRealtimeAgentMessage = (text) => {
  if (realtimeAgentIndex === null) return;
  transcriptData[realtimeAgentIndex].text = text;
  if (typeof applyFilters === "function") {
    applyFilters();
  } else {
    renderData(transcriptData, transcriptBody);
  }
};

const resetRealtimeAgentMessage = () => {
  realtimeAgentIndex = null;
  realtimeAgentText = "";
};

setupAudioVisualizer();

const apiFetch = async (path, options = {}) => {
  try {
    const response = await fetch(`${API_ROOT}${path}`, {
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {}),
      },
      ...options,
    });

    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || response.statusText);
    }
    return await response.json();
  } catch (error) {
    console.warn("Backend request failed:", error);
    return null;
  }
};

const setLoadingState = (loading, options = {}) => {
  isBackendReady = !loading;
  if (loadingScreen) {
    loadingScreen.classList.toggle("hidden", !loading);
  }
  if (loadingTitle && options.title) {
    loadingTitle.textContent = options.title;
  }
  if (loadingSubtitle && options.subtitle) {
    loadingSubtitle.textContent = options.subtitle;
  }
  const disabled = !!loading;
  [
    micButton,
    pauseButton,
    leaveButton,
    transcriptToggle,
    transcriptClose,
    soundToggle,
    sendButton,
    messageInput,
  ].forEach((el) => {
    if (el) el.disabled = disabled;
  });
  if (appContainer) {
    appContainer.setAttribute("aria-busy", String(loading));
  }
};

const resetFirstAudioChunkWait = () => {
  if (firstAudioChunkReady) return;
  firstAudioChunkReady = new Promise((resolve) => {
    resolveFirstAudioChunk = resolve;
  });
};

const markFirstAudioChunkSent = () => {
  if (hasSentMicAudio) return;
  hasSentMicAudio = true;
  if (resolveFirstAudioChunk) {
    resolveFirstAudioChunk();
  }
};

const releaseFirstAudioChunkWait = () => {
  if (resolveFirstAudioChunk) {
    resolveFirstAudioChunk();
  }
};

const waitForRealtimeReady = async () => {
  const retryDelayMs = 1000;
  let attempts = 0;
  while (true) {
    attempts += 1;
    const session = await ensureSession();
    if (session) {
      try {
        await ensureRealtimeSocket();
        return true;
      } catch (err) {
        console.warn("Realtime socket not ready yet", err);
      }
    }
    setLoadingState(true, {
      title: "Connecting to AgentRob…",
      subtitle: `Waiting for realtime audio connection (${attempts}s)`
    });
    await new Promise((resolve) => setTimeout(resolve, retryDelayMs));
  }
};

const postMessage = async (payload) => {
  try {
    const response = await fetch(`${API_ROOT}/session/${sessionId}/message`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const detail = await response.text();
      return { error: detail || response.statusText };
    }

    return { data: await response.json() };
  } catch (error) {
    return { error: error?.message || "Failed to reach backend" };
  }
};

const streamMessage = async (payload) => {
  try {
    const response = await fetch(`${API_ROOT}/session/${sessionId}/message/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const detail = await response.text();
      return { error: detail || response.statusText };
    }

    if (!response.body || !response.body.getReader) {
      return { error: "Streaming not supported by this browser" };
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let agentIndex = null;
    let agentText = "";
    let sawDelta = false;

    const findBoundary = (text) => {
      const lf = text.indexOf("\n\n");
      const crlf = text.indexOf("\r\n\r\n");
      if (lf === -1 && crlf === -1) return null;
      if (lf === -1) return { index: crlf, length: 4 };
      if (crlf === -1) return { index: lf, length: 2 };
      return lf < crlf ? { index: lf, length: 2 } : { index: crlf, length: 4 };
    };

    const ensureAgentMessage = () => {
      if (agentIndex !== null) return;
      transcriptData.push({ author: "AgentRob", text: "" });
      agentIndex = transcriptData.length - 1;
      if (applyFilters) {
        applyFilters();
      } else {
        renderData(transcriptData, transcriptBody);
      }
    };

    const updateAgentMessage = (text) => {
      if (agentIndex === null) return;
      transcriptData[agentIndex].text = text;
      if (applyFilters) {
        applyFilters();
      } else {
        renderData(transcriptData, transcriptBody);
      }
    };

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary = findBoundary(buffer);
      while (boundary) {
        const chunk = buffer.slice(0, boundary.index).trim();
        buffer = buffer.slice(boundary.index + boundary.length);
        if (chunk.startsWith("data:")) {
          const payloadText = chunk.replace(/^data:\s*/i, "");
          try {
            const event = JSON.parse(payloadText);
            if (event.type === "delta") {
              ensureAgentMessage();
              agentText += event.text || "";
              updateAgentMessage(agentText);
              sawDelta = true;
            } else if (event.type === "error") {
              appendMessage("System", event.detail || "Streaming error");
            } else if (event.type === "done") {
              return { data: { transcript: transcriptData } };
            }
          } catch (err) {
            console.warn("Failed to parse stream event", err);
          }
        }
        boundary = findBoundary(buffer);
      }
    }

    return { data: { transcript: transcriptData, sawDelta } };
  } catch (error) {
    return { error: error?.message || "Failed to reach backend" };
  }
};

const loadSessionState = () => {
  const mic = sessionStorage.getItem(STORAGE_KEYS.mic);
  const transcript = sessionStorage.getItem(STORAGE_KEYS.transcript);
  const paused = sessionStorage.getItem(STORAGE_KEYS.paused);
  const sounds = sessionStorage.getItem(STORAGE_KEYS.sounds);

  if (mic !== null) sessionState.micListening = mic === "true";
  if (transcript !== null) sessionState.transcriptOpen = transcript === "true";
  if (paused !== null) sessionState.paused = paused === "true";
  if (sounds !== null) sessionState.soundsOn = sounds === "true";
};

const persistSessionState = () => {
  sessionStorage.setItem(STORAGE_KEYS.mic, String(sessionState.micListening));
  sessionStorage.setItem(
    STORAGE_KEYS.transcript,
    String(sessionState.transcriptOpen)
  );
  sessionStorage.setItem(STORAGE_KEYS.paused, String(sessionState.paused));
  sessionStorage.setItem(STORAGE_KEYS.sounds, String(sessionState.soundsOn));
};

const updateMicUI = () => {
  const micIcon = micButton?.querySelector(".icon");
  micStatus.textContent = sessionState.micListening ? "Listening" : "Muted";
  micButton.setAttribute(
    "aria-label",
    sessionState.micListening ? "Mute mic" : "Unmute mic"
  );
  micButton.classList.toggle("active", sessionState.micListening);
  if (micIcon) {
    micIcon.innerHTML = sessionState.micListening ? icons.mic : icons.micMuted;
  }
};

const updateTranscriptUI = () => {
  transcriptPanel.classList.toggle("hidden", !sessionState.transcriptOpen);
  if (appContainer) {
    appContainer.classList.toggle(
      "transcript-open",
      sessionState.transcriptOpen
    );
  }
  const ariaLabel = sessionState.transcriptOpen
    ? "Close transcript"
    : "Open transcript";
  [transcriptToggle, transcriptClose].forEach((btn) => {
    if (btn) btn.setAttribute("aria-label", ariaLabel);
  });
  [transcriptToggle].forEach((btn) => {
    if (btn) btn.classList.toggle("active", sessionState.transcriptOpen);
  });
};

const updatePauseUI = () => {
  pauseButton.classList.toggle("active", sessionState.paused);
  pauseButton.setAttribute(
    "aria-label",
    sessionState.paused ? "Resume conversation" : "Pause conversation"
  );
  if (micButton) {
    micButton.disabled = sessionState.paused;
  }
};

const updateSoundUI = () => {
  const ariaLabel = sessionState.soundsOn
    ? "Turn off notification sounds"
    : "Turn on notification sounds";
  [soundToggle].forEach((btn) => {
    if (btn) btn.setAttribute("aria-label", ariaLabel);
  });
  [soundToggle].forEach((btn) => {
    if (btn) btn.classList.toggle("active", !sessionState.soundsOn);
  });
};

const renderState = () => {
  updateMicUI();
  updateTranscriptUI();
  updatePauseUI();
  updateSoundUI();
  persistSessionState();
};

const appendMessage = (author, text) => {
  transcriptData.push({ author, text });
  if (applyFilters) {
    applyFilters();
  } else {
    renderData(transcriptData, transcriptBody);
  }
};

const replaceTranscript = (items = []) => {
  transcriptData.length = 0;
  items.forEach((item) => {
    transcriptData.push({ author: item.author, text: item.text });
  });
  if (applyFilters) {
    applyFilters();
  } else {
    renderData(transcriptData, transcriptBody);
  }
};

const syncSessionState = (session) => {
  if (!session) return;
  sessionState.micListening = !!session.mic_listening;
  sessionState.paused = !!session.paused;
  sessionState.soundsOn = !!session.sounds_on;
};

const ensureSession = async () => {
  if (sessionId) {
    const existing = await apiFetch(`/session/${sessionId}`);
    if (existing) {
      syncSessionState(existing);
      renderState();
      return existing;
    }
  }

  const created = await apiFetch("/session", { method: "POST" });
  if (!created) return null;
  sessionId = created.session_id;
  sessionStorage.setItem(STORAGE_KEYS.sessionId, sessionId);
  const session = await apiFetch(`/session/${sessionId}`);
  if (session) {
    syncSessionState(session);
    renderState();
  }
  return session;
};

const refreshTranscript = async () => {
  if (!sessionId) return;
  const transcript = await apiFetch(`/session/${sessionId}/transcript`);
  if (transcript) {
    replaceTranscript(transcript);
  }
};

const getWsBase = () => API_BASE.replace(/^http/i, (match) => (match === "https" ? "wss" : "ws"));

const ensureRealtimeSocket = async () => {
  if (!sessionId) return null;

  if (micSocket && micSocket.readyState === WebSocket.OPEN) {
    return micSocket;
  }

  if (micSocket && micSocket.readyState === WebSocket.CONNECTING && micSocketReady) {
    return micSocketReady;
  }

  const wsUrl = `${getWsBase()}/api/session/${sessionId}/realtime`;
  micSocketReady = new Promise((resolve, reject) => {
    micSocket = new WebSocket(wsUrl);

    micSocket.addEventListener("open", () => resolve(micSocket));
    micSocket.addEventListener("error", (event) => {
      console.warn("Mic websocket error", event);
      reject(new Error("Failed to connect to realtime endpoint"));
    });
    micSocket.addEventListener("close", () => {
      micSocket = null;
      micSocketReady = null;
    });
    // Agent message state is now at module level (realtimeAgentIndex, realtimeAgentText)
    // Use ensureRealtimeAgentMessage() and updateRealtimeAgentMessage() instead

    micSocket.addEventListener("message", (event) => {
      try {
        const payload = JSON.parse(event.data);
        const type = payload?.type;
        const responseId = payload?.response_id || payload?.event_id || null;
        
        // Capture user voice input transcript (dedupe within a short window)
        if (
          type === "conversation.item.input_audio_transcription.completed" ||
          type === "input_audio_transcription.completed" ||
          type === "conversation.item.created"
        ) {
          const transcript = payload?.transcript || payload?.item?.content?.[0]?.transcript;
          if (typeof transcript === "string" && transcript.trim()) {
            const normalized = transcript.trim();
            const now = Date.now();
            const isDuplicate =
              normalized === lastVoiceTranscript && now - lastVoiceTranscriptAt < 1500;
            if (!isDuplicate) {
              lastVoiceTranscript = normalized;
              lastVoiceTranscriptAt = now;
              appendMessage("You", normalized);
            }
          }
        }
        
        if (
          type === "response.output_text.delta" ||
          type === "response.text.delta" ||
          type === "response.audio_transcript.delta"
        ) {
          // Reject text from cancelled responses
          if (isResponseCancelled) {
            return;
          }
          const delta = payload?.delta?.text || payload?.delta;
          if (typeof delta === "string" && delta) {
            ensureRealtimeAgentMessage();
            realtimeAgentText += delta;
            updateRealtimeAgentMessage(realtimeAgentText);
          }
        }
        
        // When a new response starts, stop any old audio FIRST and clear rejection
        if (type === "response.created" || type === "response.started") {
          stopAllPlayback(false); // Not an interrupt, just cleanup
          currentResponseId = responseId || Date.now().toString();
          cancelledResponseId = null;
          isResponseCancelled = false; // Clear rejection flag - accept audio from new response
          audioAcceptedAfter = Date.now(); // Only accept audio from now
          // Reset agent message for new response
          resetRealtimeAgentMessage();
        }
        
        // Handle response cancellation/interruption from server
        if (
          type === "response.cancelled" ||
          type === "response.interrupted" ||
          type === "response.failed"
        ) {
          console.debug("[voice] server confirmed cancelled/interrupted:", type);
          stopAllPlayback(false);
          micResponding = false;
          currentResponseId = null;
          isResponseCancelled = true; // Reject any straggling audio
        }
        
        // Handle input audio buffer speech detection (server-side VAD detected speech)
        // Only interrupt if the agent is actively responding - this prevents
        // false triggers from ambient noise that the server's VAD may pick up
        if (type === "input_audio_buffer.speech_started") {
          if (micResponding && activeAudioSources.length > 0) {
            console.debug("[voice] server-side VAD speech detected, interrupting playback");
            cancelCurrentResponse();
          } else {
            console.debug("[voice] server-side VAD speech detected but not interrupting (no active playback)");
          }
        }
        
        const isAudioType =
          type && (type.startsWith("response.audio") || type.startsWith("response.output_audio"));
        if (isAudioType) {
          // AGGRESSIVE REJECTION: If response is cancelled, reject ALL audio
          if (isResponseCancelled) {
            console.debug("[voice] REJECTING audio - response was cancelled");
            return;
          }
          
          // Also reject if we have no current response
          if (!currentResponseId) {
            console.debug("[voice] REJECTING audio - no active response");
            return;
          }
          
          const audioInfo = extractAudioDelta(payload);
          if (audioInfo?.audio) {
            const resolvedRate = audioInfo.sampleRate || AUDIO_TARGET_SAMPLE_RATE;
            console.debug(
              "[voice] ACCEPTING audio delta len",
              audioInfo.audio.length,
              "type",
              type
            );
            if (isProbablyBase64(audioInfo.audio)) {
              enqueuePlayback(audioInfo.audio, resolvedRate);
            } else {
              console.debug("[voice] skip audio delta (not base64)");
            }
          }
        }
        if (
          type === "response.done" ||
          type === "response.completed" ||
          type === "response.output_text.done" ||
          type === "response.text.done" ||
          type === "response.audio_transcript.done"
        ) {
          micResponding = false;
          // Use module-level reset instead of local variables
          resetRealtimeAgentMessage();
        }
      } catch {
        // ignore non-json events for now
      }
    });
  });

  return micSocketReady;
};

const downsampleBuffer = (buffer, inputRate, targetRate) => {
  if (targetRate === inputRate) return buffer;
  if (targetRate > inputRate) return buffer;
  const ratio = inputRate / targetRate;
  const newLength = Math.round(buffer.length / ratio);
  const result = new Float32Array(newLength);
  let offsetResult = 0;
  let offsetBuffer = 0;

  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio);
    let accum = 0;
    let count = 0;
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < buffer.length; i += 1) {
      accum += buffer[i];
      count += 1;
    }
    result[offsetResult] = accum / count;
    offsetResult += 1;
    offsetBuffer = nextOffsetBuffer;
  }

  return result;
};

const floatTo16BitPCM = (floatBuffer) => {
  const output = new Int16Array(floatBuffer.length);
  for (let i = 0; i < floatBuffer.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, floatBuffer[i]));
    output[i] = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
  }
  return output;
};

const arrayBufferToBase64 = (buffer) => {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = "";
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode(...chunk);
  }
  return btoa(binary);
};

const stripDataUrlPrefix = (s) => {
  if (typeof s !== "string") return s;
  const match = s.match(/^data:.*?;base64,(.*)$/i);
  return match ? match[1] : s;
};

const isProbablyBase64 = (s) => {
  if (typeof s !== "string") return false;
  const cleaned = stripDataUrlPrefix(s).replace(/\s+/g, "");
  if (cleaned.length < 4) return false;
  return /^[A-Za-z0-9+/=_-]+$/.test(cleaned);
};

const normalizeBase64 = (s) => {
  const clean = stripDataUrlPrefix(s)
    .replace(/\s+/g, "")
    .replace(/-/g, "+")
    .replace(/_/g, "/");
  const pad = clean.length % 4;
  return pad === 0 ? clean : clean + "=".repeat(4 - pad);
};

const coerceSampleRate = (value) => {
  const rate = Number(value);
  // Validate that it's a reasonable audio sample rate (8kHz to 48kHz)
  return Number.isFinite(rate) && rate >= 8000 && rate <= 48000 ? Math.round(rate) : null;
};

const extractAudioDelta = (payload) => {
  if (!payload) return null;
  // Try multiple possible sample rate field names from server events
  const sampleRate =
    coerceSampleRate(payload.audio_sampling_rate) ||
    coerceSampleRate(payload.output_audio_sampling_rate) ||
    coerceSampleRate(payload.sample_rate) ||
    coerceSampleRate(payload.sampling_rate) ||
    coerceSampleRate(payload.audio_sample_rate) ||
    null;

  const direct = payload.delta ?? payload.audio ?? payload?.payload?.audio;
  if (typeof direct === "string") {
    return { audio: direct, sampleRate };
  }

  if (direct && typeof direct === "object") {
    const nestedRate =
      coerceSampleRate(direct.sample_rate) ||
      coerceSampleRate(direct.sampling_rate) ||
      coerceSampleRate(direct.audio_sample_rate) ||
      coerceSampleRate(direct.audio_sampling_rate) ||
      sampleRate;
    const audio =
      direct.audio ||
      direct.data ||
      direct.chunk ||
      direct.base64 ||
      direct.audio_base64 ||
      null;
    if (typeof audio === "string") {
      return { audio, sampleRate: nestedRate };
    }
  }

  return null;
};

const base64ToArrayBuffer = (base64) => {
  try {
    if (!isProbablyBase64(base64)) return null;
    const normalized = normalizeBase64(base64);
    const binary = atob(normalized);
    const len = binary.length;
    if (len === 0) return null;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i += 1) {
      bytes[i] = binary.charCodeAt(i);
    }
    return bytes.buffer;
  } catch (err) {
    console.debug("[voice] base64 decode failed", err);
    return null;
  }
};

let playbackContext = null;
let playbackAnalyser = null;
let playbackAnalyserData = null;
let playbackQueue = Promise.resolve();
let playbackNextStartTime = 0;
let activeAudioSources = []; // Track active audio sources for interruption
let currentResponseId = null; // Track which response we're playing audio for
let cancelledResponseId = null; // Track the response that was cancelled to reject its audio
let lastInterruptTime = 0; // Track when last interrupt happened
let audioAcceptedAfter = 0; // Only accept audio received after this timestamp
let isResponseCancelled = false; // Flag to reject all audio until new response
let masterGainNode = null; // Master gain for instant muting during interrupts
let playbackVersion = 0; // Incremented on each cancel to invalidate queued audio
const INTERRUPT_DELAY_MS = 1500; // Delay before playing new audio after interrupt

/**
 * Stop all currently playing and queued audio immediately.
 * Used when user interrupts the agent.
 */
const stopAllPlayback = (isInterrupt = false) => {
  const sourceCount = activeAudioSources.length;
  if (sourceCount > 0) {
    console.debug("[voice] stopping all playback, active sources:", sourceCount);
  }
  
  // Increment version to invalidate all queued audio
  playbackVersion++;
  
  // FIRST: Immediately mute via gain node (instant, no latency)
  if (masterGainNode) {
    masterGainNode.gain.setValueAtTime(0, playbackContext?.currentTime || 0);
  }
  
  // Stop all active audio sources immediately
  for (const source of activeAudioSources) {
    try {
      source.stop(0); // Stop immediately (0 = now)
      source.disconnect();
    } catch (e) {
      // Source may already be stopped
    }
  }
  activeAudioSources = [];
  
  // Reset playback queue - create a fresh resolved promise
  playbackQueue = Promise.resolve();
  playbackNextStartTime = 0;
  
  // Record interrupt time for delay before new audio
  if (isInterrupt) {
    lastInterruptTime = Date.now();
    
    // Close and recreate the AudioContext to fully clear the audio pipeline
    if (playbackContext) {
      try {
        playbackContext.close();
      } catch (e) {
        // Ignore close errors
      }
      playbackContext = null;
      playbackAnalyser = null;
      masterGainNode = null;
    }
  }
};

/**
 * Cancel the current agent response.
 * Sends cancel event to backend and stops audio playback.
 */
const cancelCurrentResponse = async () => {
  // Only cancel if there's actually an active response
  if (!currentResponseId && !micResponding) {
    console.debug("[voice] No active response to cancel, skipping");
    return;
  }
  
  // Set flag to reject ALL incoming audio until new response starts
  isResponseCancelled = true;
  
  // Mark the current response as cancelled
  if (currentResponseId) {
    cancelledResponseId = currentResponseId;
  }
  
  // Stop audio playback FIRST - synchronously and immediately, mark as interrupt
  stopAllPlayback(true);
  currentResponseId = null;
  
  console.debug("[voice] CANCELLED - rejecting all audio until new response starts");
  
  // Reset responding state immediately
  micResponding = false;
  micRespondingAt = 0;
  
  // Reset speech detection state
  speechFrameCount = 0;
  
  // Send cancel event to the server (don't await - fire and forget for speed)
  // Note: Only send response.cancel, not input_audio_buffer.clear (causes errors with server-side VAD)
  sendMicEvent("response.cancel");
};

const ensurePlaybackContext = () => {
  if (!playbackContext) {
    // Don't specify sampleRate - let browser use its default (usually 44100 or 48000)
    // This allows proper resampling of incoming audio at any sample rate
    playbackContext = new (window.AudioContext || window.webkitAudioContext)();
  }
  if (playbackContext.state === "suspended") {
    playbackContext.resume();
  }
  return playbackContext;
};

const ensureMasterGain = (audioCtx) => {
  if (!masterGainNode && audioCtx) {
    masterGainNode = audioCtx.createGain();
    masterGainNode.gain.value = 1.0;
    masterGainNode.connect(audioCtx.destination);
  }
  return masterGainNode;
};

const ensurePlaybackAnalyser = (audioCtx) => {
  if (!audioCtx) return null;
  // Ensure master gain exists first
  const gain = ensureMasterGain(audioCtx);
  if (!playbackAnalyser) {
    playbackAnalyser = audioCtx.createAnalyser();
    playbackAnalyser.fftSize = 256;
    playbackAnalyser.smoothingTimeConstant = 0.85;
    playbackAnalyserData = new Uint8Array(playbackAnalyser.frequencyBinCount);
    // Connect analyser to master gain instead of directly to destination
    playbackAnalyser.connect(gain);
  }
  return playbackAnalyser;
};

const audioBufferFromBase64 = async (audioCtx, base64Audio, sampleRate = AUDIO_TARGET_SAMPLE_RATE) => {
  const arrayBuffer = base64ToArrayBuffer(base64Audio);
  if (!arrayBuffer) return null;
  const bytes = new Uint8Array(arrayBuffer);
  if (bytes.byteLength === 0) return null;
  
  // Check for WAV header (RIFF)
  const header = String.fromCharCode(...bytes.slice(0, 4));
  if (header === "RIFF") {
    return audioCtx.decodeAudioData(arrayBuffer.slice(0));
  }
  
  // Try native decoding first for other formats
  try {
    return await audioCtx.decodeAudioData(arrayBuffer.slice(0));
  } catch (err) {
    console.debug("[voice] decodeAudioData failed, falling back to PCM16", err);
  }
  
  // Fallback: Interpret as raw PCM16 little-endian audio
  const byteLength = bytes.byteLength;
  if (byteLength % 2 !== 0) {
    console.debug("[voice] pcm16 byteLength not even, truncating", byteLength);
  }
  const sampleCount = Math.floor(byteLength / 2);
  if (sampleCount <= 0) return null;
  
  // Convert PCM16 (little-endian signed 16-bit) to Float32
  const float32 = new Float32Array(sampleCount);
  const view = new DataView(arrayBuffer);
  for (let i = 0; i < sampleCount; i += 1) {
    // Read as little-endian signed 16-bit integer and normalize to [-1, 1]
    const sample = view.getInt16(i * 2, true);
    float32[i] = sample / 32768.0; // Use 32768.0 for proper normalization
  }
  
  // Use the provided sample rate (from server) for correct playback speed
  const sourceSampleRate = sampleRate || AUDIO_TARGET_SAMPLE_RATE;
  
  // Create buffer at the source sample rate - AudioContext will resample if needed
  const audioBuffer = audioCtx.createBuffer(1, sampleCount, sourceSampleRate);
  audioBuffer.getChannelData(0).set(float32);
  return audioBuffer;
};

const playAudioBase64 = async (base64Audio, sampleRate, capturedVersion) => {
  // Check version - if it changed, audio is stale
  if (capturedVersion !== playbackVersion) {
    console.debug("[voice] play BLOCKED - version mismatch (stale audio)");
    return;
  }
  
  const audioCtx = ensurePlaybackContext();
  const audioBuffer = await audioBufferFromBase64(audioCtx, base64Audio, sampleRate);
  if (!audioBuffer) return;
  
  // Check version again after async decode
  if (capturedVersion !== playbackVersion) {
    console.debug("[voice] play BLOCKED after decode - version mismatch");
    return;
  }
  
  // Ensure master gain is at full volume before playing
  const gain = ensureMasterGain(audioCtx);
  if (gain) {
    gain.gain.setValueAtTime(1.0, audioCtx.currentTime);
  }
  
  const source = audioCtx.createBufferSource();
  source.buffer = audioBuffer;
  const analyser = ensurePlaybackAnalyser(audioCtx);
  if (analyser) {
    source.connect(analyser);
  } else if (gain) {
    source.connect(gain);
  } else {
    source.connect(audioCtx.destination);
  }
  
  // FINAL check right before starting - this is the critical gate
  if (capturedVersion !== playbackVersion || isResponseCancelled) {
    console.debug("[voice] play BLOCKED at start - cancelled or version mismatch");
    source.disconnect();
    return;
  }
  
  // Track this source for potential interruption
  activeAudioSources.push(source);
  
  // Schedule playback to avoid gaps and overlaps between chunks
  const now = audioCtx.currentTime;
  const startTime = Math.max(now, playbackNextStartTime);
  source.start(startTime);
  
  // Update next start time to after this buffer finishes
  playbackNextStartTime = startTime + audioBuffer.duration;
  
  // Return a promise that resolves when the audio finishes playing
  return new Promise((resolve) => {
    source.onended = () => {
      // Remove from active sources
      const idx = activeAudioSources.indexOf(source);
      if (idx !== -1) activeAudioSources.splice(idx, 1);
      resolve();
    };
    // Fallback timeout in case onended doesn't fire
    setTimeout(() => {
      const idx = activeAudioSources.indexOf(source);
      if (idx !== -1) activeAudioSources.splice(idx, 1);
      resolve();
    }, (audioBuffer.duration * 1000) + 100);
  });
};

const enqueuePlayback = (base64Audio, sampleRate) => {
  // Double-check: reject if response was cancelled
  if (isResponseCancelled) {
    console.debug("[voice] enqueue BLOCKED - response cancelled");
    return playbackQueue;
  }
  
  if (!base64Audio || base64Audio.length < 4) {
    console.debug("[voice] skip audio chunk (too short)");
    return playbackQueue;
  }
  
  // Capture current version - if it changes, this audio is stale
  const capturedVersion = playbackVersion;
  
  // Check if we need to delay playback after an interrupt
  const timeSinceInterrupt = Date.now() - lastInterruptTime;
  const needsDelay = timeSinceInterrupt < INTERRUPT_DELAY_MS;
  
  const approxBytes = Math.round((base64Audio.length * 3) / 4);
  if (needsDelay) {
    console.debug("[voice] delaying audio chunk, time since interrupt:", timeSinceInterrupt, "ms");
  } else {
    console.debug("[voice] enqueue chunk bytes≈", approxBytes);
  }
  
  playbackQueue = playbackQueue
    .then(async () => {
      // Check version - if it changed, audio is stale
      if (capturedVersion !== playbackVersion) {
        console.debug("[voice] play BLOCKED - version changed while queued");
        return;
      }
      
      // Check again before playing - might have been cancelled while waiting
      if (isResponseCancelled) {
        console.debug("[voice] play BLOCKED - response cancelled while queued");
        return;
      }
      
      // Wait for remaining delay if we were recently interrupted
      const remainingDelay = INTERRUPT_DELAY_MS - (Date.now() - lastInterruptTime);
      if (remainingDelay > 0) {
        console.debug("[voice] waiting", remainingDelay, "ms before playback");
        await new Promise(resolve => setTimeout(resolve, remainingDelay));
      }
      
      // Check one more time after delay
      if (isResponseCancelled || capturedVersion !== playbackVersion) {
        console.debug("[voice] play BLOCKED after delay - cancelled or version mismatch");
        return;
      }
      
      return playAudioBase64(base64Audio, sampleRate, capturedVersion);
    })
    .catch((err) => console.warn("[voice] playback error", err));
  return playbackQueue;
};

const sendMicEvent = async (type, payload = {}) => {
  const ws = await ensureRealtimeSocket();
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  ws.send(JSON.stringify({ type, payload }));
};

const scheduleRealtimeResponses = () => {
  if (micCommitInterval) return;
  // With server-side VAD enabled, Azure automatically:
  // 1. Detects speech start/end
  // 2. Commits the audio buffer when speech ends
  // 3. Creates the response
  // We just need a timeout to reset stale responding state
  micCommitInterval = window.setInterval(async () => {
    if (!sessionState.micListening || sessionState.paused || micStopping) return;
    const now = Date.now();
    
    // Timeout stale responses after 15 seconds
    if (micResponding && now - micRespondingAt > 15000) {
      console.debug("[mic] response timeout, resetting state");
      micResponding = false;
      micRespondingAt = 0;
    }
  }, 1000);
};

const stopRealtimeResponses = () => {
  if (micCommitInterval) {
    window.clearInterval(micCommitInterval);
    micCommitInterval = null;
  }
  micResponding = false;
  micRespondingAt = 0;
};

const stopMicCapture = async () => {
  micStopping = true;
  try {
    stopRealtimeResponses();
    
    // Stop any ongoing playback
    stopAllPlayback();
    
    // Reset speech detection state
    speechFrameCount = 0;
    lastSpeechDetectedAt = 0;
    
    if (micProcessor) {
      micProcessor.disconnect();
      micProcessor.onaudioprocess = null;
      micProcessor = null;
    }
    if (micAudioContext) {
      await micAudioContext.close();
      micAudioContext = null;
    }
    if (micStream) {
      micStream.getTracks().forEach((track) => track.stop());
      micStream = null;
    }

    // Note: With server-side VAD, commit/clear are not supported
    // The server handles audio buffer management automatically
  } finally {
    micStopping = false;
  }
};

const closeRealtimeSocket = () => {
  if (micSocket) {
    try {
      micSocket.close();
    } catch (err) {
      console.debug("[mic] failed to close websocket", err);
    }
  }
  micSocket = null;
  micSocketReady = null;
};

const startMicCapture = async () => {
  if (micStream || micStopping) return;
  if (!sessionId) {
    await ensureSession();
  }
  if (!sessionId) return;

  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
  } catch (error) {
    console.warn("Microphone permission denied or unavailable", error);
    sessionState.micListening = false;
    renderState();
    releaseFirstAudioChunkWait();
    if (sessionId) {
      apiFetch(`/session/${sessionId}/mic`, {
        method: "POST",
        body: JSON.stringify({ listening: false }),
      }).then(syncSessionState);
    }
    return;
  }

  await ensureRealtimeSocket();
  micLog("Mic capture started");
  await sendMicEvent("input_audio_buffer.clear");
  scheduleRealtimeResponses();

  // Don't force sample rate - let browser use native rate for mic compatibility
  // The downsampleBuffer() function handles conversion to AUDIO_TARGET_SAMPLE_RATE
  micAudioContext = new (window.AudioContext || window.webkitAudioContext)();
  const input = micAudioContext.createMediaStreamSource(micStream);
  micProcessor = micAudioContext.createScriptProcessor(4096, 1, 1);

  micProcessor.onaudioprocess = async (event) => {
    if (!sessionState.micListening || sessionState.paused || micStopping) return;
    const inputBuffer = event.inputBuffer.getChannelData(0);
    
    // Client-side speech detection: check if there's significant audio energy
    // Use sustained detection - require multiple consecutive frames above threshold
    // to avoid false triggers from transient noise
    if (micResponding) {
      let sumSquares = 0;
      for (let i = 0; i < inputBuffer.length; i++) {
        sumSquares += inputBuffer[i] * inputBuffer[i];
      }
      const rms = Math.sqrt(sumSquares / inputBuffer.length);
      
      if (rms > SPEECH_RMS_THRESHOLD) {
        speechFrameCount += 1;
        lastSpeechDetectedAt = Date.now();
        
        // Only trigger interrupt if we've had sustained speech for multiple frames
        if (speechFrameCount >= SPEECH_FRAMES_REQUIRED) {
          console.debug("[mic] sustained speech detected (RMS:", rms.toFixed(4), ", frames:", speechFrameCount, ") - interrupting");
          speechFrameCount = 0; // Reset after interrupt
          cancelCurrentResponse();
        }
      } else {
        // Reset frame count if we drop below threshold
        // Allow brief dips by only resetting if we've been quiet for a while
        const quietDuration = Date.now() - lastSpeechDetectedAt;
        if (quietDuration > 150) { // 150ms grace period for natural speech pauses
          speechFrameCount = 0;
        }
      }
    } else {
      // Not responding, reset speech detection state
      speechFrameCount = 0;
    }
    
    const downsampled = downsampleBuffer(
      inputBuffer,
      micAudioContext.sampleRate,
      AUDIO_TARGET_SAMPLE_RATE
    );
    const pcm16 = floatTo16BitPCM(downsampled);
    const base64Audio = arrayBufferToBase64(pcm16.buffer);
    micChunkCount += 1;
    const now = Date.now();
    if (now - micLastLogTs > 1000) {
      micLog(
        `sending chunk #${micChunkCount} bytes=${pcm16.byteLength} base64_len=${base64Audio.length}`
      );
      micLastLogTs = now;
    }
    await sendMicEvent("input_audio_buffer.append", { audio: base64Audio });
    markFirstAudioChunkSent();
  };

  input.connect(micProcessor);
  micProcessor.connect(micAudioContext.destination);
};

micButton.addEventListener("click", () => {
  if (!isBackendReady) return;
  sessionState.micListening = !sessionState.micListening;
  renderState();
  if (sessionId) {
    apiFetch(`/session/${sessionId}/mic`, {
      method: "POST",
      body: JSON.stringify({ listening: sessionState.micListening }),
    }).then(syncSessionState);
  }
  if (sessionState.micListening) {
    startMicCapture();
  } else {
    stopMicCapture();
  }
});

pauseButton.addEventListener("click", async () => {
  if (!isBackendReady) return;
  sessionState.paused = !sessionState.paused;
  if (sessionState.paused) {
    wasMicListeningBeforePause = sessionState.micListening;
    sessionState.micListening = false;
  } else {
    sessionState.micListening = wasMicListeningBeforePause;
  }
  renderState();

  // If pausing, cancel any ongoing response, stop playback, and stop mic capture
  if (sessionState.paused) {
    if (micResponding) {
      await cancelCurrentResponse();
    }
    await stopMicCapture();
  } else if (sessionState.micListening) {
    startMicCapture();
  }

  if (sessionId) {
    apiFetch(`/session/${sessionId}/pause`, {
      method: "POST",
      body: JSON.stringify({ paused: sessionState.paused }),
    }).then(syncSessionState);
    apiFetch(`/session/${sessionId}/mic`, {
      method: "POST",
      body: JSON.stringify({ listening: sessionState.micListening }),
    }).then(syncSessionState);
  }
});

leaveButton.addEventListener("click", () => {
  if (!isBackendReady) return;
  if (!confirm("Leave the session?")) return;
  (async () => {
    sessionState.micListening = false;
    sessionState.paused = false;
    renderState();

    if (micResponding) {
      await cancelCurrentResponse();
    }
    await stopMicCapture();
    closeRealtimeSocket();

    if (sessionId) {
      apiFetch(`/session/${sessionId}/leave`, {
        method: "POST",
      }).then(syncSessionState);
    }

    sessionStorage.removeItem(STORAGE_KEYS.sessionId);
    sessionStorage.removeItem(STORAGE_KEYS.mic);
    sessionStorage.removeItem(STORAGE_KEYS.transcript);
    sessionStorage.removeItem(STORAGE_KEYS.paused);
    sessionStorage.removeItem(STORAGE_KEYS.sounds);

    sessionId = null;
    replaceTranscript([]);
    alert("Session ended.");
  })();
});

[transcriptToggle, transcriptClose].forEach((btn) => {
  if (!btn) return;
  btn.addEventListener("click", () => {
    if (!isBackendReady) return;
    sessionState.transcriptOpen = !sessionState.transcriptOpen;
    renderState();
  });
});

[soundToggle].forEach((btn) => {
  if (!btn) return;
  btn.addEventListener("click", () => {
    if (!isBackendReady) return;
    sessionState.soundsOn = !sessionState.soundsOn;
    renderState();
    if (sessionId) {
      apiFetch(`/session/${sessionId}/sounds`, {
        method: "POST",
        body: JSON.stringify({ enabled: sessionState.soundsOn }),
      }).then(syncSessionState);
    }
  });
});

/**
 * Send a text message through the realtime WebSocket.
 * This keeps text and voice on the same Azure connection for unified context.
 */
const sendTextViaWebSocket = async (text) => {
  const ws = await ensureRealtimeSocket();
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    throw new Error("WebSocket not connected");
  }
  
  // DO NOT clear isResponseCancelled here - keep rejecting old audio
  // until response.created arrives for the new response
  // This prevents old audio chunks still in flight from playing
  cancelledResponseId = null;
  
  // Reset agent message state for fresh response
  resetRealtimeAgentMessage();
  
  ws.send(JSON.stringify({ type: "text.send", payload: { text } }));
  
  // Mark that we're now responding (waiting for agent reply)
  micResponding = true;
  micRespondingAt = Date.now();
};

messageForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!isBackendReady) return;
  const value = messageInput.value.trim();
  if (!value) return;
  
  // If agent is currently responding, cancel it first
  if (micResponding) {
    await cancelCurrentResponse();
  }
  
  // Add user message to transcript immediately
  appendMessage("You", value);
  messageInput.value = "";
  
  if (!sessionId) {
    await ensureSession();
  }
  
  if (sessionId) {
    try {
      // Send text through the same WebSocket as voice for unified context
      await sendTextViaWebSocket(value);
    } catch (err) {
      console.error("[form] WebSocket send failed:", err);
      appendMessage("System", "Failed to send message. Please wait for connection to be ready.");
    }
  }
});

const initializeApp = async () => {
  loadSessionState();
  renderState();
  setLoadingState(true, {
    title: "Connecting to AgentRob…",
    subtitle: "Warming up the backend, please wait.",
  });
  await waitForRealtimeReady();
  await refreshTranscript();
  if (sessionState.micListening && !sessionState.paused) {
    resetFirstAudioChunkWait();
    await startMicCapture();
    await firstAudioChunkReady;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  setLoadingState(false);
};

initializeApp();

// TODO: Replace with LiveKit + Azure Voice Live streaming integration.
// TODO: Replace with GPT-4.1 response generation and transcript sync.
