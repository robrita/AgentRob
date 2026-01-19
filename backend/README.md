# AgentRob Backend

FastAPI service that manages sessions, transcript state, text replies, and a realtime WebSocket proxy for Azure Voice Live.

## Quick start (uv)

1. Install uv: `pip install uv`.
2. Sync dependencies: `uv sync`.
3. Copy `.env.example` to `.env` and fill Azure settings.
4. Run: `uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`.

## Quick start (pip)

1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and fill Azure settings.
4. Run: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`.

## Using Azure Personal Voice

The backend supports using Azure Custom Personal Voice for TTS with real-time streaming using a **hybrid approach**:

1. **Voice Live API** handles the conversation (speech-to-text, LLM reasoning, response generation)
2. **Azure Speech SDK** synthesizes audio using your Personal Voice clone in real-time

This gives you the best of both worlds: fast conversational AI with your custom voice.

### Prerequisites

1. **Create a Personal Voice**: Use the [Custom Voice API](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/personal-voice-create-voice) to create a Personal Voice and obtain your `speakerProfileId`.
2. **S0 Speech Resource**: Personal Voice requires a paid (S0) Azure Speech resource.
3. **Speech SDK Credentials**: You need the Speech resource key and region for TTS synthesis.

### Configuration

Set the following environment variables in your `.env` file:

```bash
# Enable Personal Voice hybrid mode
AZURE_VOICELIVE_TTS_VOICE_TYPE=azure-personal

# Your speaker profile ID from the Custom Voice API
AZURE_VOICELIVE_SPEAKER_PROFILE_ID=your-speaker-profile-id

# Azure Speech SDK credentials (for Personal Voice TTS synthesis)
AZURE_SPEECH_KEY=your-speech-key
AZURE_SPEECH_REGION=southeastasia  # or your Speech resource region

# Optional: Base voice model (default: DragonLatestNeural)
AZURE_VOICELIVE_TTS_BASE_VOICE=DragonLatestNeural

# Optional: Language (default: en-US)
AZURE_VOICELIVE_TTS_LANGUAGE=en-US
```

### How It Works (Hybrid Mode)

When Personal Voice is enabled, the system operates in hybrid mode:

1. **Voice Live API** receives your speech and converts it to text (STT)
2. **Voice Live API** processes the text through the AI agent and streams back text response
3. As sentences complete, they are queued for Personal Voice synthesis
4. **Azure Speech SDK** synthesizes each sentence using your Personal Voice (SSML with `mstts:ttsembedding`)
5. Audio chunks are streamed back to the client in real-time

This approach ensures:
- ⚡ **Fast response** - Audio starts playing as soon as the first sentence is synthesized
- 🎤 **Your voice** - Uses your custom Personal Voice clone
- 🔄 **Real-time streaming** - No waiting for full response before audio plays

## Endpoints

- POST /api/session
- GET /api/session/{id}
- POST /api/session/{id}/mic
- POST /api/session/{id}/pause
- POST /api/session/{id}/sounds
- POST /api/session/{id}/leave
- GET /api/session/{id}/transcript
- POST /api/session/{id}/message
- WS /api/session/{id}/realtime

## Make Commands

The Makefile provides convenient shortcuts for common development tasks:

### `make help`
Displays all available make targets.

### `make setup`
Initializes the project by syncing dependencies using `uv sync`.

### `make sync`
Syncs dependencies with the lock file using `uv sync`.

### `make run`
Starts the development server with hot reload:
```
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
The API will be available at `http://localhost:8000`.

### `make lint`
Runs code quality checks using Ruff:
```
uv run ruff check .
```

## Lint

Run: `uv run ruff check .`
