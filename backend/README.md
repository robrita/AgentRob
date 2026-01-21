# AgentRob Backend

FastAPI service that manages sessions, transcript state, text replies, and a realtime WebSocket proxy for Azure Voice Live.

## Quick start (uv)

1. Install uv: `pip install uv`.
2. Sync dependencies: `uv sync`.
3. Activate virtual environment:
   - **Windows**: `.venv\Scripts\activate`
   - **Linux/macOS**: `source .venv/bin/activate`
4. Copy `.env.example` to `.env` and fill Azure settings.
5. Run: `uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`.

## Quick start (pip)

1. Create and activate a virtual environment:
   - **Windows**: `.venv\Scripts\activate`
   - **Linux/macOS**: `source .venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`.
3. Copy `.env.example` to `.env` and fill Azure settings.
4. Run: `uvicorn app.main:app --reload --host 0.0.0.0 --port 8000`.

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
