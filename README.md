# 🎤 Multilingual Audio RAG Chatbot

A **production-grade Multilingual Audio Retrieval-Augmented Generation (RAG) chatbot** that supports **text and real-time voice interactions**.  
The system converts speech to text, retrieves relevant documents from an internal knowledge base, generates **grounded, hallucination-free responses**, and optionally converts answers back to speech.

This project demonstrates a **full end-to-end AI pipeline** combining Speech-to-Text, Vector Search, Reranking, Large Language Models, and Text-to-Speech in a clean, modular architecture.

---

## ✨ Key Features

- 🎙️ **Live microphone input** (real-time voice queries)
- 💬 **Text-based chat support**
- 🌍 **Multilingual speech-to-text** using Whisper
- 📚 **Semantic document retrieval** with vector embeddings (ChromaDB)
- 🔄 **Cross-encoder reranking** for high-quality context selection
- 🤖 **Grounded LLM responses** (Retrieval-Augmented Generation)
- 🛑 **Hallucination prevention** — answers only from internal documents
- 🔊 **Text-to-Speech output** using Coqui TTS
- 🧩 **Modular & extensible design**
- 🖥️ **Interactive Streamlit web interface**
- 🔐 **Admin-free FFmpeg setup (project-local)**

---

## 🧠 System Architecture

```
User (Text / Voice)
        ↓
Speech-to-Text (Whisper)
        ↓
Query Embedding (SentenceTransformers)
        ↓
Vector Database Search (ChromaDB)
        ↓
Optional Reranking (Cross-Encoder)
        ↓
LLM Response Generation (OpenRouter / Mistral)
        ↓
Text-to-Speech (Coqui TTS)
        ↓
Audio + Text Response
```

---

## 📂 Project Structure

```
multilingual-audio-rag-chatbot/
├── app.py                     # Streamlit web application
├── main.py                    # RAG pipeline orchestrator
├── stt.py                     # Speech-to-Text module (Whisper)
├── tts.py                     # Text-to-Speech module (Coqui TTS)
├── embedding.py               # Text embedding & chunking
├── vector_db.py               # ChromaDB integration
├── reranker.py                # Cross-encoder reranker
├── llm.py                     # LLM interface (OpenRouter)
├── ingest_docs.py             # Knowledge base ingestion
├── config.py                  # Central configuration
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Ezhil-Aadhithya-64/multilingual-audio-rag-chatbot.git
cd multilingual-audio-rag-chatbot
```

---

### 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Environment Variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_key
OPENAI_API_KEY=your_openai_key
HUGGINGFACE_TOKEN=your_huggingface_token
```

---

### 5️⃣ FFmpeg Setup

This project uses a **local FFmpeg binary**.

1. Download FFmpeg (Windows build):
   https://www.gyan.dev/ffmpeg/builds/

2. Extract it to:

```
ffmpeg/bin/ffmpeg.exe
ffmpeg/bin/ffprobe.exe
```

The application automatically injects this path at runtime.

---

### 6️⃣ Run the Application

```bash
streamlit run app.py
```

The web UI will open in your browser.

---

## 🧪 Sample Questions

Use these to verify correct RAG behavior:

- Does VoiceAssist Pro use the internet to answer questions?
- How do I reset my password?
- What audio formats are supported?
- What happens if the information is not in the knowledge base?
- Why is the first startup slow?

### ❌ Hallucination Test
Ask:
> What is machine learning?

Expected:
- The assistant states that the information is not available.

---

## 🛡️ Design Principles

- **Strict RAG**: No document → no answer
- **Transparency**: Clearly states when information is unavailable
- **Modularity**: Each component can be replaced or upgraded
- **Security-first**: No secrets committed, no external browsing

---

## 🧩 Tech Stack

- **Frontend**: Streamlit
- **Speech-to-Text**: OpenAI Whisper (local)
- **Embeddings**: SentenceTransformers
- **Vector DB**: ChromaDB
- **Reranking**: Cross-Encoder (MS MARCO)
- **LLM**: Mistral via OpenRouter
- **Text-to-Speech**: Coqui TTS
- **Language**: Python 3.10+

---


### 🏠 Home Interface & Configuration Panel
The main Streamlit interface where users can configure input mode, language detection, retrieval settings, and audio response options.
<img width="1915" height="897" alt="Screenshot 2026-02-01 220548" src="https://github.com/user-attachments/assets/fd436b27-0a53-4b66-8074-df40fcb3909a" />


---

### 💬 Text Query with Grounded RAG Response
A text-based question is submitted, relevant documents are retrieved from the internal knowledge base, and a **strictly grounded response** is generated.
<img width="1915" height="921" alt="Screenshot 2026-02-01 220958" src="https://github.com/user-attachments/assets/8cf1c224-eec1-4a7a-8b9f-095c57f0cb61" />

The system also supports optional **text-to-speech playback** of the answer.


---

### 🎤 Live Microphone Input → Transcription → Answer
Real-time voice interaction using the system microphone:
1. User records a voice query
2. Speech is transcribed automatically
3. Relevant documents are retrieved
4. A grounded response is generated
5. The response is optionally spoken aloud

<img width="1919" height="963" alt="Screenshot 2026-02-02 004417" src="https://github.com/user-attachments/assets/c53aa76c-dcb6-4d85-a31f-0b659e73aa28" />



---


## 👤 Author

**Ezhil Aadhithya**  
GitHub: https://github.com/Ezhil-Aadhithya-64

---

## ⭐ Acknowledgements

- OpenAI Whisper
- HuggingFace SentenceTransformers
- ChromaDB
- Coqui TTS
- Streamlit
- OpenRouter

---

If you find this project useful, consider giving it a ⭐ on GitHub!
