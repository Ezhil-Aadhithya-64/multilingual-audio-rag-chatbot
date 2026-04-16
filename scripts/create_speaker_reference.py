"""
Create a speaker reference audio file for XTTS voice cloning.
This generates a simple reference voice that can be used with XTTS models.
"""

import config
from pathlib import Path
from gtts import gTTS
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_speaker_reference(
        text: str = None,
        output_path: Path = None,
        language: str = "en"
) -> Path:
    """
    Create a speaker reference audio file.

    Args:
        text: Text to synthesize (5-10 seconds recommended)
        output_path: Where to save the reference
        language: Language code

    Returns:
        Path to created reference file
    """
    # Default text (about 5-7 seconds)
    if text is None:
        text = """Hello, this is a sample voice reference. 
        It will be used to clone the speaking style for text-to-speech synthesis. 
        The reference should be clear and natural."""

    # Default output path
    if output_path is None:
        output_path = config.TTS_OUTPUT_DIR / "default_speaker_reference.wav"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Creating speaker reference at: {output_path}")
    logger.info(f"Text: {text[:50]}...")

    try:
        # Generate audio using gTTS
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(str(output_path))

        logger.info(f"✓ Speaker reference created successfully!")
        logger.info(f"  File: {output_path}")
        logger.info(f"  Size: {output_path.stat().st_size} bytes")

        return output_path

    except Exception as e:
        logger.error(f"Failed to create reference: {e}")
        raise


def create_multiple_references():
    """Create reference files for different languages"""

    references = {
        "en": "Hello, this is an English voice reference for text-to-speech synthesis.",
        "es": "Hola, esta es una referencia de voz en español para síntesis de texto a voz.",
        "fr": "Bonjour, ceci est une référence vocale en français pour la synthèse vocale.",
        "de": "Hallo, dies ist eine deutsche Stimmenreferenz für Text-zu-Sprache-Synthese.",
        "it": "Ciao, questo è un riferimento vocale italiano per la sintesi vocale.",
    }

    logger.info("Creating speaker references for multiple languages...")

    created = []
    for lang, text in references.items():
        try:
            output_path = config.TTS_OUTPUT_DIR / f"speaker_reference_{lang}.wav"
            ref_path = create_speaker_reference(text, output_path, lang)
            created.append((lang, ref_path))
            logger.info(f"  ✓ {lang}: {ref_path.name}")
        except Exception as e:
            logger.error(f"  ✗ {lang}: {e}")

    return created


if __name__ == "__main__":
    print("=" * 60)
    print("Speaker Reference Generator for XTTS")
    print("=" * 60)

    print("\nOption 1: Create default English reference")
    print("Option 2: Create references for multiple languages")
    print("Option 3: Create custom reference")

    choice = input("\nChoose option (1-3) or press Enter for default: ").strip()

    if choice == "2":
        print("\nCreating multi-language references...")
        refs = create_multiple_references()
        print(f"\n✓ Created {len(refs)} reference files in {config.TTS_OUTPUT_DIR}")

    elif choice == "3":
        print("\nCreate custom reference:")
        custom_text = input("Enter text (or press Enter for default): ").strip()
        custom_lang = input("Enter language code (default: en): ").strip() or "en"

        if not custom_text:
            ref_path = create_speaker_reference(language=custom_lang)
        else:
            ref_path = create_speaker_reference(text=custom_text, language=custom_lang)

        print(f"\n✓ Created: {ref_path}")

    else:
        # Default: create English reference
        print("\nCreating default English reference...")
        ref_path = create_speaker_reference()
        print(f"\n✓ Created: {ref_path}")

    print("\n" + "=" * 60)
    print("How to use the reference file:")
    print("=" * 60)
    print("\nIn your code:")
    print("  from tts import TextToSpeech")
    print("  tts = TextToSpeech(speaker_wav='path/to/reference.wav')")
    print("  tts.synthesize('Your text here')")
    print("\nOr pass it directly:")
    print("  tts.synthesize(text, speaker_wav='path/to/reference.wav')")
    print("\n" + "=" * 60)