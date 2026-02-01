"""
Main Pipeline Orchestrator
Integrates all modules: STT → Embedding → Vector Search → Reranking → LLM → TTS
"""

import logging
from typing import Dict, List, Optional, Union
from pathlib import Path
import config

# Import all modules
from stt import SpeechToText
from embedding import TextEmbedder
from vector_db import VectorDatabase
from reranker import DocumentReranker
from llm import MistralInstructModel
from tts import TextToSpeech

# Configure logging
logging.basicConfig(
    level=config.LOG_LEVEL,
    format=config.LOG_FORMAT,
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MultilingualAudioRAG:
    """
    Complete RAG pipeline for audio-to-text multilingual chatbot.
    Orchestrates: Audio → STT → Embeddings → Retrieval → Reranking → LLM → TTS
    """

    def __init__(
            self,
            use_local_whisper: bool = True,
            use_reranker: bool = config.USE_RERANKER,
            initialize_llm: bool = True  # Set False by default due to model size
    ):
        """
        Initialize all pipeline components.

        Args:
            use_local_whisper: Use local Whisper model vs API
            use_reranker: Enable cross-encoder reranking
            initialize_llm: Load LLM model (requires API key and network)
        """
        logger.info("Initializing Multilingual Audio RAG Pipeline...")

        # Initialize components
        self.stt = SpeechToText(use_local=use_local_whisper)
        self.embedder = TextEmbedder()
        self.vector_db = VectorDatabase(embedding_model=config.EMBEDDING_MODEL)
        self.tts = TextToSpeech()

        # Optional components
        self.use_reranker = use_reranker
        if use_reranker:
            self.reranker = DocumentReranker()
        else:
            self.reranker = None

        # LLM (load on demand or at init)
        if initialize_llm:
            self.llm = MistralInstructModel()
        else:
            self.llm = None
            logger.info("LLM not initialized. Call load_llm() when needed.")

        logger.info("Pipeline initialized successfully")

    def load_llm(self):
        """Load LLM model on demand."""
        if self.llm is None:
            logger.info("Loading LLM model...")
            self.llm = MistralInstructModel()

    def process_audio_query(
            self,
            audio_path: Union[str, Path],
            language: Optional[str] = None,
            return_audio: bool = True
    ) -> Dict[str, any]:
        """
        Complete pipeline: process audio query and generate response.

        Args:
            audio_path: Path to audio file
            language: Source language (auto-detected if None)
            return_audio: Generate TTS audio response

        Returns:
            Dictionary with transcription, response, and optional audio
        """
        logger.info(f"Processing audio query: {audio_path}")

        try:
            # Step 1: Speech-to-Text
            logger.info("Step 1: Transcribing audio...")
            stt_result = self.stt.transcribe_audio(audio_path, language)
            query_text = stt_result["text"]
            detected_language = stt_result.get("language", "unknown")

            logger.info(f"Transcription: {query_text[:100]}...")
            logger.info(f"Detected language: {detected_language}")

            # Step 2: Process text query
            result = self.process_text_query(
                query_text,
                language=detected_language,
                return_audio=return_audio
            )

            # Add STT metadata
            result["transcription"] = query_text
            result["detected_language"] = detected_language
            result["audio_duration"] = stt_result.get("duration", 0)

            return result

        except Exception as e:
            logger.error(f"Audio query processing failed: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }

    def process_text_query(
            self,
            query: str,
            language: Optional[str] = None,
            return_audio: bool = True
    ) -> Dict[str, any]:
        """
        Process text query through RAG pipeline.

        Args:
            query: Query text
            language: Query language
            return_audio: Generate TTS audio response

        Returns:
            Dictionary with response and metadata
        """
        logger.info(f"Processing text query: {query[:100]}...")

        try:
            # Step 1: Embed query
            logger.info("Step 1: Embedding query...")
            query_embedding = self.embedder.embed_text(query)

            # Step 2: Retrieve from vector database
            logger.info("Step 2: Retrieving relevant documents...")
            retrieval_params = {
                "query_embedding": query_embedding,
                "n_results": config.TOP_K_RETRIEVAL
            }

            # Add language filter if provided
            if language:
                retrieval_params["where"] = {"language": language}

            results = self.vector_db.query(**retrieval_params)

            retrieved_docs = results["documents"]
            retrieved_metadata = results["metadatas"]

            logger.info(f"Retrieved {len(retrieved_docs)} documents")

            # Step 3: Optional reranking
            if self.use_reranker and self.reranker and retrieved_docs:
                logger.info("Step 3: Reranking documents...")
                reranked = self.reranker.rerank(
                    query,
                    retrieved_docs,
                    retrieved_metadata
                )

                # Extract reranked documents
                retrieved_docs = [r["document"] for r in reranked]
                retrieved_metadata = [r.get("metadata", {}) for r in reranked]

                logger.info(f"Reranked to top {len(retrieved_docs)} documents")

            # Step 4: Generate response with Mistral LLM
            logger.info("Step 4: Generating response...")

            if self.llm is None:
                # If LLM not loaded, return simple summary
                response_text = self._generate_simple_response(query, retrieved_docs)
                llm_result = {
                    "response": response_text,
                    "sources_used": len(retrieved_docs)
                }
            else:
                llm_result = self.llm.generate_response(query, retrieved_docs)

            response_text = llm_result["response"]
            # 🚨 Detect LLM failure
            if response_text.lower().startswith("http error") or "error contacting api" in response_text.lower():
                logger.warning("LLM failed. Falling back to RAG-only response.")
                response_text = self._generate_simple_response(query, retrieved_docs)
                return_audio = False  # ❌ DO NOT speak errors

            # Step 5: Optional Text-to-Speech
            audio_path = None
            if return_audio:
                logger.info("Step 5: Generating audio response...")
                audio_path = self.tts.synthesize(
                    response_text,
                    language=language or "en"
                )
                logger.info(f"Audio generated: {audio_path}")

            # Compile results
            return {
                "query": query,
                "response": response_text,
                "audio_path": str(audio_path) if audio_path else None,
                "sources_used": llm_result.get("sources_used", 0),
                "retrieved_documents": retrieved_docs,
                "metadata": retrieved_metadata,
                "success": True
            }

        except Exception as e:
            logger.error(f"Text query processing failed: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }

    def _generate_simple_response(
            self,
            query: str,
            documents: List[str]
    ) -> str:
        """Generate simple response when LLM is not loaded."""
        if not documents:
            return "I couldn't find relevant information to answer your question."

        response = f"Based on the retrieved information:\n\n"
        for i, doc in enumerate(documents[:3], 1):
            response += f"{i}. {doc[:200]}...\n\n"

        return response.strip()

    def add_documents_to_db(
            self,
            documents: List[str],
            metadatas: Optional[List[Dict]] = None,
            chunk_documents: bool = True
    ) -> None:
        """
        Add documents to vector database.

        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
            chunk_documents: Split documents into chunks
        """
        logger.info(f"Adding {len(documents)} documents to database...")

        try:
            if chunk_documents:
                all_chunks = []
                all_embeddings = []
                all_metadata = []

                for i, doc in enumerate(documents):
                    chunks = self.embedder.chunk_text(doc)
                    chunks_with_emb = self.embedder.embed_chunks(chunks)

                    for chunk in chunks_with_emb:
                        all_chunks.append(chunk["text"])
                        all_embeddings.append(chunk["embedding"])

                        chunk_metadata = metadatas[i].copy() if metadatas and i < len(metadatas) else {}
                        chunk_metadata["chunk_id"] = chunk["chunk_id"]
                        chunk_metadata["original_doc_id"] = i
                        all_metadata.append(chunk_metadata)

                self.vector_db.add_documents(
                    documents=all_chunks,
                    embeddings=all_embeddings,
                    metadatas=all_metadata
                )

                logger.info(f"Added {len(all_chunks)} chunks from {len(documents)} documents")
            else:
                embeddings = self.embedder.embed_batch(documents)
                self.vector_db.add_documents(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )

                logger.info(f"Added {len(documents)} documents")

        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    rag_pipeline = MultilingualAudioRAG(
        use_local_whisper=True,
        use_reranker=True,
        initialize_llm=True
    )

    application_docs = [

        """
        VoiceAssist Pro is an AI-powered product support assistant designed to help users
        access product documentation using voice or text-based queries.

        The system supports multilingual speech-to-text input, semantic document retrieval,
        and grounded response generation. VoiceAssist Pro does not generate answers from
        general AI knowledge or external sources.

        VoiceAssist Pro is commonly used in customer support environments, internal company
        helpdesks, and technical documentation platforms where accuracy, consistency, and
        traceability of information are critical.
        """,

        """
        Internet Usage Policy for VoiceAssist Pro

        VoiceAssist Pro does not use the internet to answer user questions.
        The system does not perform live web searches, access external websites,
        or retrieve information from third-party services.

        All answers generated by VoiceAssist Pro are based solely on the internal
        knowledge base that has been explicitly provided and maintained by administrators.

        If requested information is not available in the internal knowledge base,
        the assistant will clearly state that it cannot find the requested information.
        """,

        """
        Account Registration and Password Reset

        To use VoiceAssist Pro, users must register with a valid email address.
        After registration, a verification email is sent and must be confirmed
        before the account can be accessed.

        Password Reset Procedure:
        1. Open the VoiceAssist Pro login page.
        2. Click the "Forgot Password" link below the password field.
        3. Enter the email address associated with the account.
        4. Submit the request and wait for the password reset email.
        5. Open the email and click the password reset link.
        6. Create a new password with at least eight characters, including one
           uppercase letter, one number, and one special character.
        7. Save the new password and log in using the updated credentials.

        For security reasons, accounts are temporarily locked after five consecutive
        failed login attempts and are automatically unlocked after thirty minutes.
        If issues persist, users should contact product support.
        """,

        """
        Audio Input and Speech-to-Text Support

        VoiceAssist Pro allows users to submit queries using uploaded audio files
        or live microphone input.

        Supported audio formats include WAV, MP3, M4A, FLAC, and OGG.
        The recommended maximum audio length per request is five minutes.

        The system automatically detects the spoken language in the audio input.
        Users may optionally specify the language manually to improve transcription
        accuracy.

        For best results, audio recordings should be clear, free of background noise,
        and spoken at a normal pace. If transcription fails, users should verify that
        the audio file is not corrupted and that required audio dependencies such as
        FFmpeg are properly installed.
        """,

        """
        Answer Generation and Knowledge Base Behavior

        VoiceAssist Pro generates answers exclusively from documents stored in its
        internal knowledge base.

        When a user submits a query, the system retrieves the most relevant document
        sections using semantic similarity search and generates a response grounded
        in that retrieved information.

        If the requested information is not present in the knowledge base, the
        assistant will explicitly state that the information is not available.
        This behavior prevents hallucinated or misleading responses and ensures
        transparent and reliable support interactions.

        Administrators are responsible for keeping the knowledge base accurate
        and up to date by adding or modifying documentation as needed.
        """,

        """
        Troubleshooting and Performance Information

        Users may experience slower responses during the first application startup.
        This behavior is expected, as AI models may need to download and initialize
        during the initial run.

        Audio playback issues may occur if the text-to-speech engine fails to load.
        In such cases, VoiceAssist Pro automatically attempts a fallback speech
        synthesis method to continue providing audio output.

        If the assistant consistently fails to answer questions, users should verify
        that the knowledge base has been populated with relevant and current documents.
        An empty or outdated knowledge base will limit the system’s ability to respond.
        """,

        """
        Support and Escalation Policy

        VoiceAssist Pro provides automated assistance but does not replace human
        support for critical issues such as billing, account recovery, or service
        outages.

        Users can contact the product support team through the official support
        email address or helpdesk portal. Standard support requests are typically
        answered within one business day.

        Issues that significantly impact business operations should be marked
        as high priority to ensure faster escalation and resolution.
        """
    ]

    application_metadata = [
        {"language": "en", "source": "product_overview", "topic": "overview", "type": "description"},
        {"language": "en", "source": "policy", "topic": "internet_usage", "type": "policy"},
        {"language": "en", "source": "account_support", "topic": "password_reset", "type": "procedure"},
        {"language": "en", "source": "audio_support", "topic": "speech_to_text", "type": "usage"},
        {"language": "en", "source": "knowledge_base", "topic": "rag_behavior", "type": "explanation"},
        {"language": "en", "source": "troubleshooting", "topic": "performance", "type": "support"},
        {"language": "en", "source": "support_policy", "topic": "escalation", "type": "policy"}
    ]

    print("Adding sample documents to database...")
    rag_pipeline.add_documents_to_db(application_docs, application_metadata)

    print("\nProcessing text query...")
    query = "What is machine learning?"
    result = rag_pipeline.process_text_query(query, return_audio=False)

    if result["success"]:
        print(f"\nQuery: {result['query']}")
        print(f"Response: {result['response']}")
        print(f"Sources used: {result['sources_used']}")
    else:
        print(f"Error: {result.get('error')}")

    print("\nPipeline test completed successfully!")
