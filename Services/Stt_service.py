import assemblyai as aai
import os
import logging

logger = logging.getLogger(__name__)
aai.settings.api_key = os.getenv("ASSEMBLY_AI_API_KEY") or ""


def transcribe_audio(audio_bytes: bytes, api_key: str | None = None) -> str:
    """
    Non-streaming transcription (not used by the WebSocket path right now).
    Accepts an optional api_key to override .env.
    """
    prev_key = aai.settings.api_key
    if api_key:
        aai.settings.api_key = api_key
    try:
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_bytes)
        if transcript.status == "error":
            logger.error(f"AssemblyAI error: {transcript.error}")
            raise ValueError(f"AssemblyAI error: {transcript.error}")
        return transcript.text
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise
    finally:
        # Restore original key
        aai.settings.api_key = prev_key