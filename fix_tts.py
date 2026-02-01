"""
Quick Fix Script for TTS Issues
Run this to diagnose and fix TTS problems.
"""

import sys
from pathlib import Path

print("=" * 70)
print("TTS TROUBLESHOOTING AND FIX SCRIPT")
print("=" * 70)

# Step 1: Check current TTS model
print("\n[1/5] Checking current configuration...")
try:
    import config

    print(f"✓ Current TTS model: {config.TTS_MODEL}")
    print(f"✓ TTS language: {config.TTS_LANGUAGE}")
except Exception as e:
    print(f"✗ Error loading config: {e}")
    sys.exit(1)

# Step 2: Check if TTS packages are installed
print("\n[2/5] Checking TTS packages...")
missing_packages = []

try:
    from TTS.api import TTS

    print("✓ Coqui TTS installed")
except ImportError:
    print("✗ Coqui TTS not installed")
    missing_packages.append("TTS")

try:
    from gtts import gTTS

    print("✓ gTTS installed (good for fallback)")
except ImportError:
    print("⚠ gTTS not installed (recommended as fallback)")
    missing_packages.append("gtts")

if missing_packages:
    print(f"\n⚠ Missing packages: {', '.join(missing_packages)}")
    print(f"Install with: pip install {' '.join(missing_packages)}")

# Step 3: Test current TTS model
print("\n[3/5] Testing current TTS model...")
if "xtts" in config.TTS_MODEL.lower():
    print("⚠ You're using XTTS which requires speaker parameter")
    print("\nRECOMMENDED SOLUTIONS:")
    print("\nOption A: Install gTTS for automatic fallback")
    print("  pip install gtts")
    print("  (The updated tts.py will automatically use this)")

    print("\nOption B: Switch to simpler Coqui model")
    print("  In config.py, change:")
    print('  TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"')

    print("\nOption C: Use gTTS directly")
    print("  In config.py, you could create a simpler TTS class using only gTTS")
else:
    print("✓ Using compatible TTS model")

# Step 4: Create a simple working TTS example
print("\n[4/5] Creating simple TTS test...")

test_text = "Hello, this is a test of the text to speech system."

# Try gTTS first (simplest)
print("\nTrying gTTS (simplest solution)...")
try:
    from gtts import gTTS

    output_dir = config.TTS_OUTPUT_DIR
    output_file = output_dir / "test_gtts.mp3"

    tts = gTTS(text=test_text, lang='en', slow=False)
    tts.save(str(output_file))

    print(f"✓ gTTS works! Audio saved to: {output_file}")
    print(f"✓ File size: {output_file.stat().st_size} bytes")
    print("\n→ RECOMMENDED: Use gTTS as your TTS solution")

except Exception as e:
    print(f"✗ gTTS test failed: {e}")

# Try Coqui TTS with simple model
print("\nTrying simple Coqui model...")
try:
    from TTS.api import TTS

    # Use a simple model that doesn't need speaker
    simple_model = "tts_models/en/ljspeech/tacotron2-DDC"
    print(f"Testing with: {simple_model}")
    print("(This may take a moment to download...)")

    tts = TTS(simple_model)
    output_file = config.TTS_OUTPUT_DIR / "test_coqui_simple.wav"

    tts.tts_to_file(text=test_text, file_path=str(output_file))

    print(f"✓ Simple Coqui model works! Audio saved to: {output_file}")
    print(f"✓ File size: {output_file.stat().st_size} bytes")
    print(f"\n→ UPDATE config.py with: TTS_MODEL = \"{simple_model}\"")

except Exception as e:
    print(f"✗ Coqui test failed: {e}")

# Step 5: Provide solution summary
print("\n" + "=" * 70)
print("[5/5] SOLUTION SUMMARY")
print("=" * 70)

print("\n🔧 IMMEDIATE FIX:")
print("\n1. Install gTTS (easiest):")
print("   pip install gtts")

print("\n2. Replace your tts.py with the updated version that includes:")
print("   - Automatic fallback to gTTS")
print("   - Proper speaker handling for XTTS")
print("   - Error handling and logging")

print("\n3. OR change TTS model in config.py:")
print('   TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"')

print("\n📝 ALTERNATIVE: Simple gTTS-only solution")
print("\nCreate a simple_tts.py file:")
print('''
from gtts import gTTS
from pathlib import Path
import config

def text_to_speech(text, output_path=None, language='en'):
    if output_path is None:
        output_path = config.TTS_OUTPUT_DIR / "output.mp3"

    tts = gTTS(text=text, lang=language, slow=False)
    tts.save(str(output_path))
    return output_path
''')

print("\n✅ After applying the fix:")
print("   - Your TTS will work with automatic fallback")
print("   - No more 'speaker_wav' errors")
print("   - Simple and reliable")

print("\n" + "=" * 70)
print("Run this script anytime to diagnose TTS issues!")
print("=" * 70)