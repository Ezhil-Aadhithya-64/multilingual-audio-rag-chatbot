"""
Text-to-Speech Module
Converts text responses to audio using Coqui TTS.
Supports multilingual speech synthesis with automatic fallback.
"""

import logging
from typing import Optional
from pathlib import Path
from datetime import datetime
import config
import re


# Configure logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

def clean_text_for_tts(text: str) -> str:
    # Remove markdown symbols
    text = re.sub(r"\*\*|\*", "", text)

    # Replace smart punctuation
    text = text.replace("‘", "'").replace("’", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("—", "-")

    # Remove emojis / non-ascii
    text = text.encode("ascii", errors="ignore").decode()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

class TextToSpeech:
    """
    Text-to-speech converter using Coqui TTS with automatic fallback.
    Supports multilingual speech generation.
    """

    def __init__(
            self,
            model_name: str = config.TTS_MODEL,
            use_gpu: bool = False
    ):
        """
        Initialize TTS model.

        Args:
            model_name: Coqui TTS model name
            use_gpu: Use GPU for inference if available
        """
        logger.info(f"Initializing TTS model: {model_name}")

        try:
            from TTS.api import TTS
            self.tts = TTS(model_name=model_name, gpu=use_gpu)
            self.model_name = model_name
            logger.info("TTS model loaded successfully")

            # Get supported languages if available
            try:
                self.languages = self.tts.languages if hasattr(self.tts, 'languages') else []
                if self.languages:
                    logger.info(f"Supported languages: {', '.join(self.languages)}")
            except:
                self.languages = []

            # Check if model requires speaker
            self.requires_speaker = self._check_speaker_requirement()

            # Get available speakers if they exist
            self.speakers = []
            if hasattr(self.tts, 'speakers') and self.tts.speakers:
                self.speakers = self.tts.speakers
                logger.info(f"Available speakers: {len(self.speakers)}")

        except Exception as e:
            logger.error(f"Failed to load TTS model: {str(e)}")
            logger.info("Consider using gtts as fallback: pip install gtts")
            raise

    def _check_speaker_requirement(self) -> bool:
        """Check if the model requires speaker parameter"""
        # XTTS models typically require speaker_wav or speaker_id
        requires_speaker = "xtts" in self.model_name.lower()
        if requires_speaker:
            logger.info("Model requires speaker parameter (voice cloning model)")
        return requires_speaker

    def synthesize(
            self,
            text: str,
            output_path: Optional[Path] = None,
            language: str = config.TTS_LANGUAGE,
            speaker: Optional[str] = None,
            speaker_wav: Optional[str] = None
    ) -> Path:
        """
        Convert text to speech audio file.

        Args:
            text: Text to synthesize
            output_path: Output audio file path (auto-generated if None)
            language: Target language code
            speaker: Speaker voice identifier (for multi-speaker models)
            speaker_wav: Reference audio for voice cloning (for XTTS models)

        Returns:
            Path to generated audio file
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for synthesis")
            raise ValueError("Text cannot be empty")

        # Generate output path if not provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = config.TTS_OUTPUT_DIR / f"tts_output_{timestamp}.wav"

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Synthesizing text to: {output_path}")

        try:
            # Try primary synthesis method
            safe_text = clean_text_for_tts(text)
            self._synthesize_primary(safe_text, output_path, language, speaker, speaker_wav)
            logger.info(f"Audio synthesized successfully: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Primary synthesis failed: {str(e)}")
            logger.info("Attempting fallback to gTTS...")

            try:
                # Fallback to gTTS
                self._synthesize_gtts_fallback(text, output_path, language)
                logger.info(f"Fallback synthesis successful: {output_path}")
                return output_path
            except Exception as fallback_error:
                logger.error(f"Fallback synthesis also failed: {str(fallback_error)}")
                raise Exception(f"All synthesis methods failed. Original error: {str(e)}")

    def _synthesize_primary(
            self,
            text: str,
            output_path: Path,
            language: str,
            speaker: Optional[str],
            speaker_wav: Optional[str]
    ):
        """Primary synthesis using Coqui TTS"""
        # Prepare synthesis parameters
        tts_params = {
            "text": text,
            "file_path": str(output_path)
        }

        # Add language if supported
        if self.languages and language in self.languages:
            tts_params["language"] = language
        elif self.languages and len(self.languages) > 0:
            logger.warning(f"Language {language} not supported. Available: {self.languages}")
            # Try to map to closest language
            lang_prefix = language[:2]  # e.g., 'en' from 'en-US'
            matching = [l for l in self.languages if l.startswith(lang_prefix)]
            if matching:
                tts_params["language"] = matching[0]
                logger.info(f"Using closest match: {matching[0]}")

        # Handle speaker parameter for different model types
        if self.requires_speaker:
            # For XTTS models that need speaker reference
            if speaker_wav:
                tts_params["speaker_wav"] = speaker_wav
                logger.info(f"Using speaker_wav: {speaker_wav}")
            elif speaker and self.speakers and speaker in self.speakers:
                tts_params["speaker"] = speaker
                logger.info(f"Using speaker: {speaker}")
            elif self.speakers and len(self.speakers) > 0:
                # Use first available speaker as default
                default_speaker = self.speakers[0]
                tts_params["speaker"] = default_speaker
                logger.info(f"Using default speaker: {default_speaker}")
            else:
                # Try without speaker and let it fail to trigger fallback
                logger.warning("XTTS model requires speaker but none provided")
                raise ValueError("Speaker parameter required for voice cloning model")
        else:
            # Regular TTS models may support optional speaker
            if speaker and self.speakers and speaker in self.speakers:
                tts_params["speaker"] = speaker

        # Generate speech
        self.tts.tts_to_file(**tts_params)

    def _synthesize_gtts_fallback(self, text: str, output_path: Path, language: str):
        """Fallback synthesis using Google TTS"""
        try:
            from gtts import gTTS

            # Map language code (e.g., 'en' from 'en-us')
            lang_code = language.split('-')[0] if '-' in language else language

            # Generate speech with gTTS
            tts = gTTS(text=text, lang=lang_code, slow=False)

            # Save as mp3 first
            temp_mp3 = output_path.with_name(output_path.stem + "_fallback.mp3")

            tts.save(str(temp_mp3))

            # If output needs to be .wav, convert it
            if output_path.suffix.lower() == '.wav':
                self._convert_mp3_to_wav(temp_mp3, output_path)
                temp_mp3.unlink()  # Remove temp mp3
            else:
                # Just rename if output is mp3
                temp_mp3.rename(output_path)

            logger.info("gTTS fallback synthesis successful")

        except ImportError:
            raise ImportError("gTTS not installed. Install with: pip install gtts")

    def _convert_mp3_to_wav(self, mp3_path: Path, wav_path: Path):
        try:
            from pydub import AudioSegment
            import config

            AudioSegment.converter = str(config.FFMPEG_PATH / "ffmpeg.exe")
            AudioSegment.ffprobe = str(config.FFMPEG_PATH / "ffprobe.exe")

            audio = AudioSegment.from_mp3(str(mp3_path))
            audio.export(str(wav_path), format="wav")

        except Exception as e:
            logger.warning(f"Format conversion failed: {e}")
            mp3_path.rename(wav_path.with_suffix(".mp3"))

    def synthesize_streaming(
            self,
            text: str,
            language: str = config.TTS_LANGUAGE
    ) -> bytes:
        """
        Synthesize speech and return as bytes (for streaming).

        Args:
            text: Text to synthesize
            language: Target language code

        Returns:
            Audio data as bytes
        """
        # Synthesize to temporary file
        temp_path = config.TTS_OUTPUT_DIR / "temp_stream.wav"
        self.synthesize(text, temp_path, language)

        # Read and return bytes
        with open(temp_path, "rb") as f:
            audio_bytes = f.read()

        # Cleanup temp file
        temp_path.unlink()

        return audio_bytes

    def get_available_models(self) -> list:
        """Get list of available TTS models."""
        try:
            from TTS.api import TTS
            models = TTS.list_models()
            return models
        except:
            return []

    def get_available_languages(self) -> list:
        """Get list of supported languages for current model."""
        return self.languages

    def get_available_speakers(self) -> list:
        """Get list of available speakers for current model."""
        return self.speakers


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Text-to-Speech Module Test")
    print("=" * 60)

    # Initialize TTS
    try:
        tts = TextToSpeech()
        print(f"✓ TTS initialized: {tts.model_name}")
        print(f"✓ Languages: {tts.languages}")
        print(f"✓ Requires speaker: {tts.requires_speaker}")
        print(f"✓ Available speakers: {len(tts.speakers)}")
    except Exception as e:
        print(f"✗ TTS initialization failed: {e}")
        print("\nTrying with fallback configuration...")
        # You can modify config.TTS_MODEL and retry here
        exit(1)

    # Example text
    sample_text = "Hello! This is a test of the text-to-speech system."

    # Synthesize speech
    try:
        print("\nSynthesizing sample text...")
        output_file = tts.synthesize(
            text=sample_text,
            language="en"
        )
        print(f"✓ Audio generated: {output_file}")
        print(f"✓ File size: {output_file.stat().st_size} bytes")
    except Exception as e:
        print(f"✗ Synthesis failed: {e}")

    print("\n" + "=" * 60)
    print("TTS module test complete")
    print("=" * 60)