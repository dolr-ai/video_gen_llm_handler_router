### Video Gen LLM Handler
Split a creative prompt into audio-only and video-only via Google Gemini.

Requirements: `GEMINI_API_KEY`; optional `GEMINI_MODEL` (default `gemini-2.0-flash`), `LLM_TIMEOUT_S`, `MAX_CONCURRENCY`, `RETRY_ATTEMPTS`. Auth: Bearer JWT (EdDSA/Ed25519) verified via `consts.PUBLIC_KEY_TO_VERIFY_INCOMING_CALLS`; payload must equal `{"sub":"yral-video-gen-llm-handler","company":"Yral"}`.

Run: `pip install -r requirements.txt && uvicorn app.main:app --port 8000`

Endpoints:
- POST `/v1/split` (auth) body `{"prompt":"..."}` → `{"audio_prompt","video_prompt","model","request_id","latency_ms"}`
- GET `/v1/status` (auth) → service status
- GET `/healthz`, `/v1/healthz` → `{"ok": true}`; GET `/version` → `{"version","model"}`

Curl:
`curl -sS -X POST http://localhost:8000/v1/split -H "Authorization: Bearer $JWT_TOKEN" -H "Content-Type: application/json" -d '{"prompt":"..."}'`