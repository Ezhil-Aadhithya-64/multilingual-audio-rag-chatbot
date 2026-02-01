"""
Configuration file for Multilingual Audio RAG Chatbot
All constants, API keys, and hyperparameters are defined here.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio_samples"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
FFMPEG_PATH = BASE_DIR / "ffmpeg" / "bin"

# Inject ffmpeg into PATH (NO admin needed)
os.environ["PATH"] = str(FFMPEG_PATH) + os.pathsep + os.environ.get("PATH", "")
# Create directories if they don't exist
for directory in [DATA_DIR, AUDIO_DIR, CHROMA_DB_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Keys (load from environment variables for security)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-key-here")  # Replace with your actual key or use .env
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)

# STT Configuration
STT_MODEL = "whisper-1"  # OpenAI API model
STT_LOCAL_MODEL = "base"  # Options: tiny, base, small, medium, large
USE_LOCAL_WHISPER = False  # Set to False to use OpenAI API
AUDIO_CHUNK_LENGTH_MS = 30000  # 30 seconds per chunk

# Embedding Configuration
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIMENSION = 384
CHUNK_SIZE = 50  # tokens per chunk
CHUNK_OVERLAP = 25  # overlap between chunks

# Vector Database Configuration
CHROMA_COLLECTION_NAME = "multilingual_audio_docs"
TOP_K_RETRIEVAL = 10  # Number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.5  # Minimum similarity score

# Reranker Configuration
USE_RERANKER = True
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N = 5  # Re-rank top N from initial retrieval

# LLM Configuration
LLM_MODEL_NAME = "mosaicml/mpt-7b-instruct"
LLM_MAX_LENGTH = 512
LLM_TEMPERATURE = 0.1
LLM_TOP_P = 0.9
LLM_DEVICE = "auto"  # auto, cpu, cuda

# Grounding Prompt Template
GROUNDINGPROMPT = (
    "You are a helpful AI assistant. Using the information below, provide a concise and creative explanation. "
    "Avoid adding information not supported by the context.\n\n"
    "Context: {context}\nQuestion: {query}\nAnswer:"
)
# ============================================================================
# TTS Configuration - FIXED FOR XTTS ISSUES
# ============================================================================

# OPTION 1: Use XTTS with automatic speaker selection (current setup - now fixed with fallback)
TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"
# OPTION 2: Use simpler single-language models (no speaker required)
# TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"  # English only, no speaker needed
# TTS_MODEL = "tts_models/en/ljspeech/glow-tts"  # English only, fast

# OPTION 3: Use language-specific models
# TTS_MODEL = "tts_models/es/css10/vits"  # Spanish
# TTS_MODEL = "tts_models/fr/css10/vits"  # French
# TTS_MODEL = "tts_models/de/css10/vits"  # German

TTS_LANGUAGE = "en"  # Default language
TTS_SPEAKER = None  # Speaker voice (will auto-select if available)
TTS_OUTPUT_DIR = DATA_DIR / "tts_output"
TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# TTS Fallback Configuration
# The updated tts.py will automatically fall back to gTTS if Coqui TTS fails
# Make sure gTTS is installed: pip install gtts

# ============================================================================

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = BASE_DIR / "chatbot.log"

# Performance Settings
BATCH_SIZE = 32
MAX_WORKERS = 4  # For parallel processing

# Supported Languages
SUPPORTED_LANGUAGES = [
    "en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "ja", "ko", "zh",
    "ar", "hi", "tr", "vi", "th", "id", "ms", "tl", "sw", "am"
]

# Metadata Fields
METADATA_FIELDS = ["language", "source", "timestamp", "speaker", "confidence"]


# ============================================================================
# QUICK FIX RECOMMENDATIONS
# ============================================================================
"""
If you're experiencing TTS issues, try these options in order:

1. EASIEST: Install gTTS (automatic fallback)
   pip install gtts
   
2. Use a simpler Coqui model (no speaker needed):
   TTS_MODEL = "tts_models/en/ljspeech/tacotron2-DDC"
   
3. Use gTTS only (add to your code):
   from gtts import gTTS
   tts = gTTS(text="Hello", lang='en')
   tts.save("output.mp3")

4. For XTTS with voice cloning, provide speaker_wav:
   tts.synthesize(
       text="Hello",
       speaker_wav="path/to/reference_voice.wav"
   )
"""