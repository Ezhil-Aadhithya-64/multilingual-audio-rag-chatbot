"""
Simple Clean Chatbot Interface
"""

import logging
from datetime import datetime
import uuid
from pathlib import Path

import streamlit as st
from st_audiorec import st_audiorec

import config
from services.orchestrator import AgenticOrchestrator

# Page config
st.set_page_config(
    page_title="FAQ Assistant",
    page_icon="🤖",
    layout="centered"
)

# Logging
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Load orchestrator using session state (not cache_resource to avoid ChromaDB corruption)
def load_orchestrator():
    """
    Load orchestrator with error handling and session state management.
    
    CRITICAL: Uses session_state instead of @st.cache_resource to prevent ChromaDB corruption.
    The singleton pattern in VectorDatabase handles instance reuse internally.
    """
    # Use session state instead of cache_resource to avoid ChromaDB corruption
    if "orchestrator" not in st.session_state:
        try:
            logger.info("Initializing orchestrator in session state...")
            
            # Show initialization progress
            with st.spinner("🔧 Initializing AI system (first time may take longer)..."):
                st.session_state.orchestrator = AgenticOrchestrator()
            
            logger.info("✅ Orchestrator initialized successfully")
            
        except RuntimeError as e:
            # Handle ChromaDB corruption errors specifically
            error_str = str(e)
            logger.error(f"Failed to initialize orchestrator: {e}")
            
            if "ChromaDB initialization failed" in error_str or "corrupted" in error_str.lower():
                st.error(
                    "🚨 **Database Corruption Detected**\n\n"
                    "The vector database is corrupted and needs to be reset.\n\n"
                    "**To fix this issue:**\n\n"
                    "1. Open a terminal in the project directory\n"
                    "2. Run: `python scripts/quick_fix_db.py`\n"
                    "3. Run: `python init_db_simple.py`\n"
                    "4. Restart this application\n\n"
                    "**Technical details:**\n"
                    f"```\n{error_str}\n```"
                )
            else:
                st.error(
                    f"❌ **Failed to initialize AI system**\n\n"
                    f"Error: {error_str}\n\n"
                    "Please check the logs for more details."
                )
            
            st.stop()
            
        except Exception as e:
            logger.error(f"Unexpected error during initialization: {e}", exc_info=True)
            st.error(
                f"❌ **Unexpected error during initialization**\n\n"
                f"Error: {e}\n\n"
                "Please check the logs and try restarting the application."
            )
            st.stop()
    
    return st.session_state.orchestrator

def main():
    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        input_mode = st.radio(
            "Input Mode",
            ["Text", "Voice"],
            index=0
        )
        
        st.divider()
        
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()
    
    # Main title
    st.title("🤖 Chat Assistant")
    
    # Load orchestrator with spinner
    try:
        with st.spinner("Initializing AI system..."):
            orchestrator = load_orchestrator()
    except Exception as e:
        st.error(f"Failed to load system: {e}")
        st.stop()
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show audio if available
            if message.get("audio_path"):
                st.audio(message["audio_path"])
    
    # Chat input
    if input_mode == "Text":
        # Text input using chat_input (fixed at bottom by default)
        if prompt := st.chat_input("What is up?"):
            # Add user message and display immediately
            st.session_state.messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Display user message immediately
            with st.chat_message("user"):
                st.write(prompt)
            
            # Get assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        result = orchestrator.process_query(
                            query=prompt,
                            session_id=st.session_state.session_id
                        )
                        
                        response = result["response"]
                        
                        # Display response
                        st.write(response)
                        
                        # Add assistant message to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })
                        
                    except Exception as e:
                        error_msg = f"Error: {e}"
                        st.error(error_msg)
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })
                        logger.error(f"Error processing query: {e}")
    
    else:
        # Voice input mode
        st.divider()
        st.subheader("🎤 Voice Input")
        
        wav_audio = st_audiorec()
        
        if wav_audio is not None:
            st.audio(wav_audio, format="audio/wav")
            
            if st.button("Send Voice Message", type="primary", use_container_width=True):
                # Save audio file
                audio_filename = f"voice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
                audio_path = config.AUDIO_DIR / audio_filename
                config.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
                
                with open(audio_path, "wb") as f:
                    f.write(wav_audio)
                
                try:
                    with st.spinner("Processing voice..."):
                        result = orchestrator.process_audio_query(
                            audio_path=audio_path,
                            session_id=st.session_state.session_id
                        )
                    
                    # Add user message (transcription)
                    transcription = result["transcription"]
                    st.session_state.messages.append({
                        "role": "user",
                        "content": f"🎤 {transcription}"
                    })
                    
                    # Add assistant response
                    response = result["response"]
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing voice: {e}")
                    logger.error(f"Voice processing error: {e}")
                
                finally:
                    # Clean up audio file
                    if audio_path.exists():
                        audio_path.unlink()


if __name__ == "__main__":
    main()
