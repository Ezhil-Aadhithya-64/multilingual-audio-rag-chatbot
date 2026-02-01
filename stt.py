"""
Speech-to-Text Module
Converts audio to text using OpenAI Whisper (API or local model).
Handles multiple languages automatically.
"""

import logging
import whisper
from typing import Optional, Dict, Union
from pathlib import Path
from pydub import AudioSegment
import openai
import os
import config

# Ensure ffmpeg is visible for Whisper
os.environ["PATH"] = str(config.FFMPEG_PATH) + os.pathsep + os.environ.get("PATH", "")
# Configure logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class SpeechToText:
    """
    Speech-to-Text converter using Whisper model.
    Supports both OpenAI API and local Whisper models.
    """

    def __init__(
            self,
            use_local: bool = config.USE_LOCAL_WHISPER,
            model_name: str = config.STT_LOCAL_MODEL,
            api_key: Optional[str] = None
    ):
        """
        Initialize STT model.

        Args:
            use_local: Use local Whisper model if True, else use OpenAI API
            model_name: Model size for local (tiny/base/small/medium/large)
            api_key: OpenAI API key (required if use_local=False)
        """
        self.use_local = use_local
        self.model_name = model_name

        if use_local:
            logger.info(f"Loading local Whisper model: {model_name}")
            self.model = whisper.load_model(model_name)
            logger.info("Local Whisper model loaded successfully")
        else:
            logger.info("Using OpenAI Whisper API")
            openai.api_key = api_key or config.OPENAI_API_KEY
            if not openai.api_key:
                raise ValueError("OpenAI API key required for API-based STT")

    def transcribe_audio(
            self,
            audio_path: Union[str, Path],
            language: Optional[str] = None,
            return_metadata: bool = True
    ) -> Dict[str, any]:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Target language code (auto-detected if None)
            return_metadata: Include metadata like language, confidence

        Returns:
            Dictionary with transcription and metadata
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info(f"Transcribing audio: {audio_path}")

        try:
            if self.use_local:
                result = self._transcribe_local(audio_path, language)
            else:
                result = self._transcribe_api(audio_path, language)

            logger.info("Transcription completed successfully")

            if return_metadata:
                return {
                    "text": result["text"],
                    "language": result.get("language", "unknown"),
                    "segments": result.get("segments", []),
                    "duration": self._get_audio_duration(audio_path)
                }
            else:
                return {"text": result["text"]}

        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            raise

    def _transcribe_local(
            self,
            audio_path: Path,
            language: Optional[str] = None
    ) -> Dict:
        """Transcribe using local Whisper model."""
        options = {"fp16": False}  # Disable FP16 for CPU compatibility

        if language:
            options["language"] = language

        result = self.model.transcribe(str(audio_path), **options)
        return result

    def _transcribe_api(
            self,
            audio_path: Path,
            language: Optional[str] = None
    ) -> Dict:
        """Transcribe using OpenAI Whisper API."""
        with open(audio_path, "rb") as audio_file:
            params = {
                "model": config.STT_MODEL,
                "file": audio_file
            }

            if language:
                params["language"] = language

            response = openai.Audio.transcribe(**params)

            return {
                "text": response["text"],
                "language": language or "unknown"
            }

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get audio duration in seconds."""
        try:
            audio = AudioSegment.from_file(str(audio_path))
            return len(audio) / 1000.0  # Convert ms to seconds
        except Exception as e:
            logger.warning(f"Could not determine audio duration: {e}")
            return 0.0

    def chunk_audio(
            self,
            audio_path: Union[str, Path],
            chunk_length_ms: int = config.AUDIO_CHUNK_LENGTH_MS
    ) -> list[Path]:
        """
        Split large audio file into smaller chunks for processing.

        Args:
            audio_path: Path to audio file
            chunk_length_ms: Chunk length in milliseconds

        Returns:
            List of paths to audio chunks
        """
        audio_path = Path(audio_path)
        audio = AudioSegment.from_file(str(audio_path))

        chunks = []
        chunk_dir = config.AUDIO_DIR / "chunks"
        chunk_dir.mkdir(exist_ok=True)

        for i, start_ms in enumerate(range(0, len(audio), chunk_length_ms)):
            end_ms = min(start_ms + chunk_length_ms, len(audio))
            chunk = audio[start_ms:end_ms]

            chunk_path = chunk_dir / f"{audio_path.stem}_chunk_{i}.wav"
            chunk.export(str(chunk_path), format="wav")
            chunks.append(chunk_path)

            logger.info(f"Created chunk {i + 1}: {chunk_path}")

        return chunks


# Example usage
if __name__ == "__main__":
    # Initialize STT
    stt = SpeechToText(use_local=True, model_name="base")

    # Example: Transcribe a sample audio file
    # audio_file = config.AUDIO_DIR / "sample.wav"
    # result = stt.transcribe_audio(audio_file)
    # print(f"Transcription: {result['text']}")
    # print(f"Language: {result['language']}")

    print("STT module loaded successfully")
