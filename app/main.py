import os
import json
import uuid
import time
import asyncio
import logging
import re
from typing import Dict, Tuple, Optional

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from fastapi.responses import JSONResponse
import jwt

import consts
# Google Gemini (google.genai)
from google import genai
from dotenv import load_dotenv

load_dotenv()


# Logging: concise JSON-ish to stdout
logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger("prompt-splitter")


# Config (env-first, sane defaults)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is required")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
LLM_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "20"))
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", "200"))
RETRY_ATTEMPTS = int(os.environ.get("RETRY_ATTEMPTS", "2"))

# JWT Configuration
if not consts.PUBLIC_KEY_TO_VERIFY_INCOMING_CALLS:
    raise RuntimeError("PUBLIC_KEY_TO_VERIFY_INCOMING_CALLS is required")

JWT_PAYLOAD = {
    "sub": "yral-video-gen-llm-handler",
    "company": "Yral",
}


# SDK client and concurrency guard
client = genai.Client(api_key=GEMINI_API_KEY)
sem = asyncio.Semaphore(MAX_CONCURRENCY)

# JWT Security
security = HTTPBearer()

def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    """
    Verify JWT token and return payload if valid.
    Raises HTTPException if token is invalid.
    """
    try:
        # Decode and verify the JWT token
        payload = jwt.decode(
            credentials.credentials,
            consts.PUBLIC_KEY_TO_VERIFY_INCOMING_CALLS,
            algorithms=["EdDSA"],
        )
        
        # Verify payload matches expected values
        if payload != JWT_PAYLOAD:
            log.warning(f"JWT payload mismatch. Expected: {JWT_PAYLOAD}, Received: {payload}")
            raise HTTPException(
                status_code=401,
                detail="Invalid token payload"
            )
        
        return payload
        
    except jwt.ExpiredSignatureError:
        log.warning("JWT token has expired")
        raise HTTPException(
            status_code=401,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError as e:
        log.warning(f"Invalid JWT token: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )
    except Exception as e:
        log.error(f"JWT verification error: {str(e)}")
        raise HTTPException(
            status_code=401,
            detail="Authentication failed"
        )


app = FastAPI(title="Prompt Splitter", version="1.0.0")


class SplitIn(BaseModel):
    prompt: str


class SplitOut(BaseModel):
    audio_prompt: str
    video_prompt: str
    model: str
    request_id: str
    latency_ms: int


SYSTEM_INSTRUCTIONS = (
    "You split a single creative prompt into two. This is part of an automation.\n"
    "It's important to encorporate every detail of the prompt into the output. Your job is to segment the prompt into two parts, one for the audio prompt and one for the video prompt.\n"
    "If an audio description is not present, you should return a likely audio description that suits the given requirement.\n"
    "This is a part of an automation. You are not allowed to add any extra text of comments. You always respond with a fenced JSON code block that is to be consumed by rest of the process"
    "Reply ONLY with a fenced JSON code block and nothing else.\n"
    "Format EXACTLY as:\n"
    "```json\n"
    "{\n"
    "  \"audio_prompt\": \"<sound/music/voiceover-only description, no visuals>\",\n"
    "  \"video_prompt\": \"<visuals-only description, no audio words>\"\n"
    "}\n"
    "```\n"
    "No extra keys, no comments, no prose."
)


def _extract_text_from_response(resp) -> str:
    raw = getattr(resp, "text", None) or getattr(resp, "output_text", None)
    if isinstance(raw, str) and raw.strip():
        return raw
    try:
        if hasattr(resp, "candidates") and resp.candidates:
            cand = resp.candidates[0]
            content = getattr(cand, "content", None)
            if content is not None and hasattr(content, "parts") and content.parts:
                part = content.parts[0]
                txt = getattr(part, "text", None)
                if isinstance(txt, str) and txt.strip():
                    return txt
    except Exception:
        pass
    return ""


def _parse_llm_output(text: str) -> Tuple[str, str]:
    """Parse (audio_prompt, video_prompt) from JSON or simple key:value lines."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Empty LLM output")
    s = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, flags=re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    obj = None
    try:
        obj = json.loads(s)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", s)
        if m:
            try:
                obj = json.loads(m.group(0))
            except Exception:
                obj = None
    if isinstance(obj, dict):
        audio = obj.get("audio_prompt")
        video = obj.get("video_prompt")
        if isinstance(audio, str) and isinstance(video, str) and audio.strip() and video.strip():
            return audio.strip(), video.strip()
    audio = video = None
    cleaned = re.sub(r"^```[a-zA-Z]*|```$", "", s, flags=re.MULTILINE)
    for line in cleaned.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        k = k.strip().lower()
        v = v.strip()
        if k == "audio_prompt" and not audio:
            audio = v
        elif k == "video_prompt" and not video:
            video = v
    if not audio or not video:
        raise ValueError("Missing audio_prompt or video_prompt in model output")
    return audio, video


def _call_gemini_sync(prompt: str) -> Dict[str, str]:
    contents = [
        {"role": "user", "parts": [{"text": f"{SYSTEM_INSTRUCTIONS}\n\nUser prompt:\n{prompt}"}]}
    ]

    last_err = None
    for attempt in range(RETRY_ATTEMPTS + 1):
        try:
            resp = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
            )
            text = _extract_text_from_response(resp)
            if not text:
                raise RuntimeError("Empty LLM response text")
            audio, video = _parse_llm_output(text)
            return {"audio_prompt": audio, "video_prompt": video}
        except Exception as e:
            last_err = e
            delay = 0.2 * (2 ** attempt)  # 0.2s, 0.4s, 0.8s
            time.sleep(delay)

    raise last_err or RuntimeError("Gemini call failed")


async def split_with_llm(prompt: str) -> Dict[str, str]:
    # google.genai is currently sync-only: offload to thread
    async with sem:
        try:
            async with asyncio.timeout(LLM_TIMEOUT_S):
                return await asyncio.to_thread(_call_gemini_sync, prompt)
        except (TimeoutError, asyncio.TimeoutError):
            raise HTTPException(status_code=504, detail="LLM timeout")


@app.post("/v1/split", response_model=SplitOut)
async def split(req: SplitIn, token_payload: dict = Depends(verify_jwt_token)):
    t0 = time.perf_counter()
    try:
        result = await split_with_llm(req.prompt)
        audio = (result or {}).get("audio_prompt")
        video = (result or {}).get("video_prompt")
        if not isinstance(audio, str) or not isinstance(video, str):
            raise HTTPException(status_code=502, detail="Invalid LLM response")

        latency_ms = int((time.perf_counter() - t0) * 1000)
        resp = SplitOut(
            audio_prompt=audio,
            video_prompt=video,
            model=GEMINI_MODEL,
            request_id=str(uuid.uuid4()),
            latency_ms=latency_ms,
        )
        log.info(json.dumps({
            "event": "split.ok", "latency_ms": latency_ms, "model": GEMINI_MODEL
        }))
        return resp
    except HTTPException:
        raise
    except Exception as e:
        log.info(json.dumps({"event": "split.error", "error": str(e)}))
        return JSONResponse(status_code=502, content={"error": "Upstream error", "detail": str(e)})


@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/v1/healthz")
def healthz():
    return {"ok": True}

@app.get("/version")
def version():
    return {"version": app.version, "model": GEMINI_MODEL}

@app.get("/v1/status")
def authenticated_status(token_payload: dict = Depends(verify_jwt_token)):
    """Authenticated endpoint that provides detailed service status"""
    return {
        "version": app.version, 
        "model": GEMINI_MODEL,
        "max_concurrency": MAX_CONCURRENCY,
        "llm_timeout_s": LLM_TIMEOUT_S,
        "retry_attempts": RETRY_ATTEMPTS,
        "authenticated": True,
        "payload": token_payload
    }




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)