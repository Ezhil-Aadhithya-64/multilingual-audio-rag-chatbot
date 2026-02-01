"""
Streamlit Web Interface
Interactive UI for the Multilingual Audio RAG Chatbot
"""

import tempfile
from pathlib import Path
import logging
from datetime import datetime

import streamlit as st
from st_audiorec import st_audiorec

import config
from main import MultilingualAudioRAG

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Multilingual Audio RAG Chatbot",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# Logging
# --------------------------------------------------
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# --------------------------------------------------
# Pipeline Loader (Cached)
# --------------------------------------------------
@st.cache_resource
def load_pipeline():
    """Load and cache the RAG pipeline."""
    return MultilingualAudioRAG(
        use_local_whisper=True,
        use_reranker=True,
        initialize_llm=True
    )

# --------------------------------------------------
# Main App
# --------------------------------------------------
def main():

    # Title
    st.title("🎤 Multilingual Audio-to-Text RAG Chatbot")
    st.markdown("""
    Ask questions using **text or voice**.  
    The system will:
    1. Transcribe speech (if audio is used)
    2. Retrieve relevant documents
    3. Generate a grounded response
    4. Optionally speak the response
    """)

    # --------------------------------------------------
    # Sidebar
    # --------------------------------------------------
    with st.sidebar:
        st.header("⚙️ Configuration")

        input_mode = st.radio(
            "Input Mode",
            ["Text Query", "Live Microphone"],
            index=0
        )

        language = st.selectbox(
            "Language",
            ["Auto-detect"] + config.SUPPORTED_LANGUAGES,
            index=0
        )
        language = None if language == "Auto-detect" else language

        st.subheader("Advanced Settings")

        top_k = st.slider(
            "Documents to Retrieve",
            min_value=1,
            max_value=20,
            value=config.TOP_K_RETRIEVAL
        )

        use_reranking = st.checkbox(
            "Enable Reranking",
            value=config.USE_RERANKER
        )

        generate_audio = st.checkbox(
            "Generate Audio Response",
            value=True
        )

        st.subheader("📚 Database")

        if st.button("View Database Stats"):
            try:
                pipeline = load_pipeline()
                st.json(pipeline.vector_db.get_collection_stats())
            except Exception as e:
                st.error(e)

        if st.button("🔄 Reset Session"):
            st.cache_resource.clear()
            st.rerun()

    # --------------------------------------------------
    # Layout
    # --------------------------------------------------
    col1, col2 = st.columns([2, 1])

    with col1:
        try:
            with st.spinner("Loading pipeline..."):
                pipeline = load_pipeline()

            pipeline.use_reranker = use_reranking

            # ------------------------------------------
            # TEXT QUERY MODE
            # ------------------------------------------
            if input_mode == "Text Query":
                st.subheader("💬 Ask a Question")

                query = st.text_area(
                    "Enter your question:",
                    height=100,
                    placeholder="Type your question here..."
                )

                if st.button("Submit Query", type="primary"):
                    if not query.strip():
                        st.warning("Please enter a question.")
                    else:
                        with st.spinner("Processing query..."):
                            result = pipeline.process_text_query(
                                query,
                                language=language,
                                return_audio=generate_audio
                            )

                        if result["success"]:
                            st.success("✅ Response generated!")

                            st.subheader("📝 Response")
                            st.write(result["response"])

                            if result.get("audio_path"):
                                st.subheader("🔊 Audio Response")
                                st.audio(result["audio_path"])

                            with st.expander("📚 Retrieved Sources"):
                                st.write(f"**Sources used:** {result['sources_used']}")
                                for i, doc in enumerate(result["retrieved_documents"][:5], 1):
                                    st.markdown(f"**Source {i}:**")
                                    st.text(doc[:300] + "..." if len(doc) > 300 else doc)
                        else:
                            st.error(result.get("error"))

            # ------------------------------------------
            # LIVE MICROPHONE MODE
            # ------------------------------------------
            else:
                st.subheader("🎤 Speak Your Question")
                st.caption("Click record, speak, then stop.")

                wav_audio_data = st_audiorec()

                if wav_audio_data is not None:
                    st.success("🎙️ Audio captured")

                    audio_filename = f"mic_input_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                    temp_audio_path = config.AUDIO_DIR / audio_filename

                    config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)

                    with open(temp_audio_path, "wb") as f:
                        f.write(wav_audio_data)

                    st.audio(wav_audio_data, format="audio/wav")

                    if st.button("Process Voice Query", type="primary"):
                        try:
                            with st.spinner("Transcribing and processing..."):
                                result = pipeline.process_audio_query(
                                    temp_audio_path,
                                    language=language,
                                    return_audio=generate_audio
                                )

                            if result["success"]:
                                st.success("✅ Processing complete!")

                                st.subheader("📝 Transcription")
                                st.info(result["transcription"])
                                st.caption(f"Detected language: {result['detected_language']}")

                                st.subheader("💬 Response")
                                st.write(result["response"])

                                if result.get("audio_path"):
                                    st.subheader("🔊 Audio Response")
                                    st.audio(result["audio_path"])

                                with st.expander("📚 Retrieved Sources"):
                                    st.write(f"**Sources used:** {result['sources_used']}")
                                    for i, doc in enumerate(result["retrieved_documents"][:5], 1):
                                        st.markdown(f"**Source {i}:**")
                                        st.text(doc[:300] + "..." if len(doc) > 300 else doc)
                            else:
                                st.error(result.get("error"))

                        finally:
                            if temp_audio_path.exists():
                                temp_audio_path.unlink()

        except Exception as e:
            st.error(f"Pipeline error: {e}")
            logger.error(e)

    # --------------------------------------------------
    # Info Panel
    # --------------------------------------------------
    with col2:
        st.subheader("ℹ️ System Info")
        st.markdown("""
        **Features**
        - 🌍 Multilingual speech & text input
        - 🎤 Live microphone support
        - 📚 Semantic document retrieval
        - 🔄 Cross-encoder reranking
        - 🤖 Grounded RAG responses
        - 🔊 Text-to-speech output
        """)

        st.subheader("➕ Add Document")
        with st.expander("Manual Ingestion"):
            doc_text = st.text_area("Document text", height=150)
            doc_language = st.selectbox("Language", config.SUPPORTED_LANGUAGES)
            doc_source = st.text_input("Source name")

            if st.button("Add to Database"):
                if doc_text.strip():
                    metadata = {
                        "language": doc_language,
                        "source": doc_source or "manual_upload",
                        "topic": "user_added"
                    }
                    pipeline = load_pipeline()
                    pipeline.add_documents_to_db([doc_text], [metadata])
                    st.success("✅ Document added")
                else:
                    st.warning("Please enter document text")

    # --------------------------------------------------
    # Footer
    # --------------------------------------------------
    st.divider()
    st.caption(
        "Multilingual Audio RAG Chatbot | Built with Whisper, SentenceTransformers, "
        "ChromaDB, Mistral (OpenRouter), and Coqui TTS"
    )


if __name__ == "__main__":
    main()
