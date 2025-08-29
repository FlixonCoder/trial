import os
import asyncio
import base64
import json
from datetime import datetime
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.logging import setup_logger
from Routes.transcriber import AssemblyAIStreamingTranscriber
from Services.Gemini_service import stream_llm_response
from Services.Tts_service import speak  # uses Murf SDK wrapper

app = FastAPI()

# Allow all origins for now (you can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] if React/JS frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

setup_logger()

# Simple in-memory store for per-session API keys
# Keys example: { "gemini_api_key": "...", "stt_api_key": "...", "tts_api_key": "...", "weather_api_key": "...", "websearch_api_key": "..." }
app.state.session_configs: Dict[str, Dict[str, str]] = {}

OUTPUT_DIR = os.path.join("Agent", "Output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHAT_DIR = "chat_histories"
os.makedirs(CHAT_DIR, exist_ok=True)


def save_chat_message(session_id: str, role: str, content: str):
    """Append a chat message to the session's JSON file."""
    file_path = os.path.join(CHAT_DIR, f"{session_id}.json")

    # Load existing history
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []

    # Append new message
    history.append({
        "role": role,
        "content": content,
        "timestamp": datetime.utcnow().isoformat()
    })

    # Save back
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


# ----- Config endpoints (UI posts keys here) -----
class ConfigPayload(BaseModel):
    session_id: str
    keys: Dict[str, str] = {}


def _sanitize_keys(keys: Dict[str, str]) -> Dict[str, str]:
    return {k: v.strip() for k, v in (keys or {}).items() if isinstance(v, str) and v.strip()}


@app.post("/config")
async def set_config(payload: ConfigPayload):
    sanitized = _sanitize_keys(payload.keys)
    app.state.session_configs[payload.session_id] = sanitized
    return {"status": "ok", "session_id": payload.session_id, "keys_set": list(sanitized.keys())}


@app.get("/config/{session_id}")
async def get_config(session_id: str):
    data = app.state.session_configs.get(session_id, {})
    # Redact values
    redacted = {k: ("•••" + v[-4:] if len(v) > 6 else "•••") for k, v in data.items()}
    return {"session_id": session_id, "keys": redacted, "has_keys": bool(data)}


@app.delete("/config/{session_id}")
async def clear_config(session_id: str):
    app.state.session_configs.pop(session_id, None)
    return {"status": "cleared", "session_id": session_id}


# --- update stream_llm_and_tts to use session keys and save AGENT replies ---
async def stream_llm_and_tts(final_text: str, websocket: WebSocket, session_id: str, session_keys: Optional[Dict[str, str]] = None):
    full_reply = []

    # Stream Gemini with per-session keys (fallback to .env inside the service if missing)
    async for chunk_text in stream_llm_response(
        prompt=final_text,
        sessionId=session_id,
        gemini_api_key=(session_keys or {}).get("gemini_api_key"),
        weather_api_key=(session_keys or {}).get("weather_api_key"),
        websearch_api_key=(session_keys or {}).get("websearch_api_key"),
    ):
        if not chunk_text:
            continue

        # Send chunk to frontend (for live streaming effect)
        await websocket.send_json({"type": "llm", "text": chunk_text})
        full_reply.append(chunk_text)

        # TTS for the chunk, using per-session key if provided (fallback to .env inside service)
        try:
            audio_bytes = speak(chunk_text, api_key=(session_keys or {}).get("tts_api_key"))
            if audio_bytes:
                audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
                await websocket.send_json({"type": "audio", "b64": audio_b64})
        except Exception as tts_err:
            await websocket.send_json({
                "type": "error",
                "message": f"Murf TTS error on chunk: {tts_err}"
            })

    # ✅ After streaming is finished, save one clean agent reply
    stitched_reply = "".join(full_reply).strip()
    if stitched_reply:
        save_chat_message(session_id, "agent", stitched_reply)

    print("\n✅ Full Gemini response:", stitched_reply)


@app.get("/history/{session_id}")
async def get_history(session_id: str):
    file_path = os.path.join(CHAT_DIR, f"{session_id}.json")
    if not os.path.exists(file_path):
        return JSONResponse(content={"history": []})
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(content={"history": data})


@app.delete("/history/{session_id}")
async def reset_history(session_id: str):
    file_path = os.path.join(CHAT_DIR, f"{session_id}.json")
    if os.path.exists(file_path):
        os.remove(file_path)
    return JSONResponse(content={"status": "reset", "session_id": session_id})


@app.websocket("/ws/audio")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🎤 Client connected")

    # --- Get sessionId from frontend query params ---
    session_id = websocket.query_params.get("session") or "default_session"

    file_path = os.path.join(OUTPUT_DIR, "recorded_audio.webm")
    if os.path.exists(file_path):
        os.remove(file_path)

    loop = asyncio.get_event_loop()

    async def on_final_async(text: str):
        # Save USER message
        save_chat_message(session_id, "user", text)

        await websocket.send_json({"type": "final", "text": text})

        # Re-read latest keys on each user turn (user may update settings mid-session)
        keys = app.state.session_configs.get(session_id, {})
        await stream_llm_and_tts(final_text=text, websocket=websocket, session_id=session_id, session_keys=keys)

    def on_final(text: str):
        asyncio.run_coroutine_threadsafe(on_final_async(text), loop)

    # Use per-session STT key if available (fallback happens in the transcriber class)
    current_keys = app.state.session_configs.get(session_id, {})
    stt_key = current_keys.get("stt_api_key")

    from Routes.transcriber import AssemblyAIStreamingTranscriber  # local import to ensure latest settings
    transcriber = AssemblyAIStreamingTranscriber(
        sample_rate=16000,
        on_final_callback=on_final,
        api_key=stt_key,
    )

    try:
        with open(file_path, "ab") as f:
            while True:
                data = await websocket.receive_bytes()
                f.write(data)
                transcriber.stream_audio(data)

    except Exception as e:
        print(f"⚠️ WebSocket connection closed: {e}")
        try:
            await websocket.send_json({"type": "info", "message": "WebSocket closed"})
        except Exception:
            pass

    finally:
        transcriber.close()
        print(f"✅ Audio saved at {file_path}")