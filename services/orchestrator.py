"""
Agentic Orchestrator - Main system coordinator
Integrates router, agents, tools, memory, guardrails, and evaluation
"""

import logging
from typing import Dict, Optional, List
from pathlib import Path
import uuid
import time

# Import core components
from core.router import IntentRouter, Intent
from core.guardrails import GuardrailsEngine
from core.session import SessionManager

# Import agents
from agents.rag_agent import RAGAgent
from agents.tool_agent import ToolAgent

# Import tools
from tools.summarization_tool import SummarizationTool
from tools.document_search_tool import DocumentSearchTool

# Import evaluation
from evaluation.metrics import MetricsCollector

# Import existing modules
from stt import SpeechToText
from embedding import TextEmbedder
from vector_db import VectorDatabase
from reranker import DocumentReranker
from llm import MistralInstructModel
from tts import TextToSpeech
import config

logger = logging.getLogger(__name__)


class AgenticOrchestrator:
    """
    Main orchestrator for the agentic AI system.
    Coordinates routing, agent execution, guardrails, and evaluation.
    """

    def __init__(
        self,
        session_manager: Optional[SessionManager] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        use_local_whisper: bool = True,
        use_reranker: bool = True
    ):
        """
        Initialize orchestrator with all system components.

        Args:
            session_manager: Session manager instance
            metrics_collector: Metrics collector instance
            use_local_whisper: Use local Whisper model
            use_reranker: Enable reranking
        """
        logger.info("Initializing Agentic Orchestrator...")

        # Initialize core components
        self.session_manager = session_manager or SessionManager()
        self.metrics_collector = metrics_collector or MetricsCollector()

        # Initialize existing modules
        self.stt = SpeechToText(use_local=use_local_whisper)
        self.embedder = TextEmbedder()
        self.vector_db = VectorDatabase(embedding_model=config.EMBEDDING_MODEL)
        self.reranker = DocumentReranker() if use_reranker else None
        self.llm = MistralInstructModel()
        self.tts = TextToSpeech()

        # Initialize tools
        self.tools = self._initialize_tools()

        # Initialize agents
        self.agents = self._initialize_agents()

        # Initialize router and guardrails
        self.router = IntentRouter(confidence_threshold=0.5)
        self.guardrails = GuardrailsEngine(
            confidence_threshold=0.5,
            grounding_threshold=0.7
        )

        logger.info("Agentic Orchestrator initialized successfully")

    def _initialize_tools(self) -> Dict:
        """Initialize all available tools."""
        tools = {
            "summarization": SummarizationTool(
                vector_db=self.vector_db,
                llm=self.llm
            ),
            "document_search": DocumentSearchTool(
                vector_db=self.vector_db,
                embedder=self.embedder
            )
        }

        logger.info(f"Initialized {len(tools)} tools")
        return tools

    def _initialize_agents(self) -> Dict:
        """Initialize all agents."""
        agents = {
            Intent.RAG_QUERY: RAGAgent(
                embedder=self.embedder,
                vector_db=self.vector_db,
                reranker=self.reranker,
                llm=self.llm
            ),
            Intent.TOOL_EXECUTION: ToolAgent(
                tool_registry=self.tools
            )
        }

        logger.info(f"Initialized {len(agents)} agents")
        return agents

    def process_query(
        self,
        query: str,
        session_id: str,
        language: Optional[str] = None,
        return_audio: bool = False,
        use_cache: bool = True
    ) -> Dict:
        """
        Process text query through complete agentic pipeline.

        Args:
            query: User query
            session_id: Session identifier
            language: Query language
            return_audio: Generate audio response
            use_cache: Use cached context

        Returns:
            Complete result dictionary
        """
        start_time = time.time()
        logger.info(f"Processing query: {query[:100]}...")

        try:
            # Step 1: Get session context
            session_context = self.session_manager.get_context(session_id)

            # Step 2: Route query
            intent, confidence, routing_metadata = self.router.route(
                query, session_context
            )

            logger.info(f"Routed to {intent.value} with confidence {confidence:.2f}")

            # Step 3: Handle different intents
            if intent == Intent.CLARIFICATION:
                result = self._handle_clarification(query, routing_metadata)

            elif intent == Intent.REJECTION:
                result = self._handle_rejection(query)

            elif intent == Intent.FOLLOW_UP:
                # Process as RAG with enhanced context
                result = self._execute_agent(
                    Intent.RAG_QUERY,
                    query,
                    session_context,
                    language=language,
                    use_cache=True
                )

            else:
                # Execute appropriate agent
                result = self._execute_agent(
                    intent,
                    query,
                    session_context,
                    language=language,
                    use_cache=use_cache,
                    suggested_tools=routing_metadata.get("suggested_tools")
                )

            # Step 4: Apply guardrails
            is_valid, final_response, guardrails_metadata = self.guardrails.validate(
                response=result["response"],
                confidence=result["confidence"],
                retrieved_context=result.get("metadata", {}).get("retrieved_documents", []),
                query=query
            )

            if not is_valid:
                result["response"] = final_response
                result["confidence"] = 0.3
                result["metadata"]["guardrails_failed"] = True
                result["metadata"]["guardrails_metadata"] = guardrails_metadata

            # Step 5: Generate audio if requested
            audio_path = None
            if return_audio and is_valid:
                try:
                    audio_path = self.tts.synthesize(
                        final_response,
                        language=language or "en"
                    )
                except Exception as e:
                    logger.warning(f"TTS failed: {e}")

            # Step 6: Update session memory
            self.session_manager.add_interaction(
                session_id=session_id,
                query=query,
                intent=intent.value,
                response=final_response,
                confidence=result["confidence"],
                retrieved_context=result.get("metadata", {}).get("retrieved_documents"),
                metadata={
                    "routing_metadata": routing_metadata,
                    "guardrails_metadata": guardrails_metadata
                }
            )

            # Step 7: Calculate total latency
            total_latency = (time.time() - start_time) * 1000

            # Build final result
            return {
                "success": result["success"],
                "response": final_response,
                "confidence": result["confidence"],
                "intent": intent.value,
                "agent": result.get("agent", "unknown"),
                "audio_path": str(audio_path) if audio_path else None,
                "retrieved_documents": result.get("metadata", {}).get("retrieved_documents"),
                "metadata": {
                    **result.get("metadata", {}),
                    "total_latency_ms": total_latency,
                    "routing_confidence": confidence,
                    "guardrails_passed": is_valid
                }
            }

        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            return {
                "success": False,
                "response": "An error occurred while processing your query.",
                "confidence": 0.0,
                "intent": "error",
                "agent": "none",
                "metadata": {"error": str(e)}
            }

    def process_audio_query(
        self,
        audio_path: Path,
        session_id: str,
        language: Optional[str] = None,
        return_audio: bool = False
    ) -> Dict:
        """
        Process audio query through complete pipeline.

        Args:
            audio_path: Path to audio file
            session_id: Session identifier
            language: Audio language
            return_audio: Generate audio response

        Returns:
            Complete result with transcription
        """
        logger.info(f"Processing audio query: {audio_path}")

        try:
            # Step 1: Transcribe audio
            stt_result = self.stt.transcribe_audio(audio_path, language)
            query_text = stt_result["text"]
            detected_language = stt_result.get("language", "unknown")

            logger.info(f"Transcribed: {query_text[:100]}...")

            # Step 2: Process as text query
            result = self.process_query(
                query=query_text,
                session_id=session_id,
                language=detected_language,
                return_audio=return_audio
            )

            # Add STT metadata
            result["transcription"] = query_text
            result["detected_language"] = detected_language
            result["audio_duration"] = stt_result.get("duration", 0)

            return result

        except Exception as e:
            logger.error(f"Audio query processing failed: {e}")
            return {
                "success": False,
                "response": "Failed to process audio query.",
                "confidence": 0.0,
                "error": str(e)
            }

    def _execute_agent(
        self,
        intent: Intent,
        query: str,
        context: Dict,
        **kwargs
    ) -> Dict:
        """
        Execute appropriate agent based on intent.

        Args:
            intent: Query intent
            query: User query
            context: Session context
            **kwargs: Additional parameters

        Returns:
            Agent execution result
        """
        agent = self.agents.get(intent)

        if not agent:
            logger.warning(f"No agent found for intent: {intent.value}")
            return {
                "success": False,
                "response": "Unable to process this type of query.",
                "confidence": 0.0,
                "agent": "none",
                "metadata": {}
            }

        # Execute agent
        result = agent.execute(query, context, **kwargs)

        return result

    def _handle_clarification(self, query: str, metadata: Dict) -> Dict:
        """
        Handle clarification requests.

        Args:
            query: User query
            metadata: Routing metadata

        Returns:
            Clarification response
        """
        reason = metadata.get("clarification_reason", "insufficient_context")

        if reason == "query_too_short":
            response = (
                "Your question seems quite brief. Could you provide more details "
                "about what you'd like to know?"
            )
        elif reason == "ambiguous_reference":
            response = (
                "I'm not sure what you're referring to. Could you please clarify "
                "or provide more context?"
            )
        else:
            response = (
                "I need more information to answer your question accurately. "
                "Could you please rephrase or provide additional context?"
            )

        return {
            "success": True,
            "response": response,
            "confidence": 0.3,
            "agent": "ClarificationAgent",
            "metadata": {"reason": reason}
        }

    def _handle_rejection(self, query: str) -> Dict:
        """
        Handle out-of-scope queries.

        Args:
            query: User query

        Returns:
            Rejection response
        """
        response = (
            "I'm designed to help with VoiceAssist Pro documentation and support. "
            "Your question appears to be outside my area of expertise. "
            "Please ask about password reset, audio input, account management, "
            "or other VoiceAssist Pro features."
        )

        return {
            "success": True,
            "response": response,
            "confidence": 0.9,
            "agent": "RejectionAgent",
            "metadata": {"reason": "out_of_scope"}
        }

    def ingest_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        chunk: bool = True
    ) -> Dict:
        """
        Ingest documents into knowledge base.

        Args:
            documents: List of document texts
            metadatas: Optional metadata
            chunk: Whether to chunk documents

        Returns:
            Ingestion result
        """
        logger.info(f"Ingesting {len(documents)} documents...")

        try:
            if chunk:
                all_chunks = []
                all_embeddings = []
                all_metadata = []

                for i, doc in enumerate(documents):
                    chunks = self.embedder.chunk_text(doc)
                    chunks_with_emb = self.embedder.embed_chunks(chunks)

                    for chunk_data in chunks_with_emb:
                        all_chunks.append(chunk_data["text"])
                        all_embeddings.append(chunk_data["embedding"])

                        chunk_metadata = metadatas[i].copy() if metadatas and i < len(metadatas) else {}
                        chunk_metadata["chunk_id"] = chunk_data["chunk_id"]
                        chunk_metadata["original_doc_id"] = i
                        all_metadata.append(chunk_metadata)

                self.vector_db.add_documents(
                    documents=all_chunks,
                    embeddings=all_embeddings,
                    metadatas=all_metadata
                )

                logger.info(f"Ingested {len(all_chunks)} chunks from {len(documents)} documents")

                return {
                    "success": True,
                    "num_documents": len(documents),
                    "num_chunks": len(all_chunks)
                }

            else:
                embeddings = self.embedder.embed_batch(documents)
                self.vector_db.add_documents(
                    documents=documents,
                    embeddings=embeddings,
                    metadatas=metadatas
                )

                logger.info(f"Ingested {len(documents)} documents")

                return {
                    "success": True,
                    "num_documents": len(documents),
                    "num_chunks": len(documents)
                }

        except Exception as e:
            logger.error(f"Document ingestion failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Example usage
if __name__ == "__main__":
    orchestrator = AgenticOrchestrator()

    # Test query
    session_id = orchestrator.session_manager.create_session()

    result = orchestrator.process_query(
        query="How do I reset my password?",
        session_id=session_id
    )

    print(f"Response: {result['response']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Intent: {result['intent']}")
    print(f"Agent: {result['agent']}")
