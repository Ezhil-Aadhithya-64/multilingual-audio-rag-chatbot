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
        use_reranker: bool = True,
        max_retries: int = 2
    ):
        """
        Initialize orchestrator with all system components.

        Args:
            session_manager: Session manager instance
            metrics_collector: Metrics collector instance
            use_local_whisper: Use local Whisper model
            use_reranker: Enable reranking
            max_retries: Maximum number of retry attempts for self-correction
        """
        logger.info("Initializing Agentic Orchestrator...")

        # Initialize core components
        self.session_manager = session_manager or SessionManager()
        self.metrics_collector = metrics_collector or MetricsCollector()
        self.max_retries = max_retries

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
        self.router = IntentRouter(confidence_threshold=0.5, llm_classifier=self.llm)
        self.guardrails = GuardrailsEngine(
            confidence_threshold=0.4,  # Lowered from 0.5 for better usability
            grounding_threshold=0.5    # Lowered from 0.7 (token overlap is strict)
        )
        
        # Self-correction statistics
        self.correction_stats = {
            "total_retries": 0,
            "successful_corrections": 0,
            "failed_corrections": 0
        }

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
                tool_registry=self.tools,
                llm=self.llm  # Pass LLM for intelligent tool selection
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

            # Step 4: Apply guardrails with self-correction loop
            retrieval_scores = result.get("metadata", {}).get("retrieval_scores", [])
            reranker_scores = result.get("metadata", {}).get("reranker_scores", [])
            retrieved_docs = result.get("metadata", {}).get("retrieved_documents", [])
            
            # Self-correction loop
            retry_count = 0
            initial_confidence = result["confidence"]
            initial_response = result["response"]
            correction_history = []
            
            while retry_count <= self.max_retries:
                is_valid, final_response, guardrails_metadata = self.guardrails.validate(
                    response=result["response"],
                    confidence=result["confidence"],
                    retrieved_context=retrieved_docs,
                    query=query,
                    retrieval_scores=retrieval_scores,
                    reranker_scores=reranker_scores,
                    retry_count=retry_count
                )
                
                if is_valid:
                    # Response passed guardrails
                    if retry_count > 0:
                        logger.info(f"Self-correction successful after {retry_count} retries")
                        self.correction_stats["successful_corrections"] += 1
                    break
                
                # Response failed guardrails
                failure_reason = guardrails_metadata.get("failure_reason", "unknown")
                retry_recommended = guardrails_metadata.get("retry_recommended", False)
                feedback = guardrails_metadata.get("feedback", "")
                
                logger.warning(f"Guardrails failed: {failure_reason} (retry: {retry_count}/{self.max_retries})")
                
                # Check if retry is recommended and we haven't exceeded max retries
                if not retry_recommended or retry_count >= self.max_retries:
                    logger.info(f"No retry recommended or max retries reached. Using fallback.")
                    if retry_count > 0:
                        self.correction_stats["failed_corrections"] += 1
                    break
                
                # Attempt self-correction
                retry_count += 1
                self.correction_stats["total_retries"] += 1
                
                # Store correction attempt
                correction_history.append({
                    "attempt": retry_count,
                    "failure_reason": failure_reason,
                    "feedback": feedback,
                    "previous_response": result["response"],
                    "confidence": result["confidence"]
                })
                
                logger.info(f"Attempting self-correction (retry {retry_count}/{self.max_retries})")
                logger.info(f"Feedback: {feedback}")
                
                # Regenerate response with feedback
                corrected_result = self._retry_with_feedback(
                    intent=intent,
                    query=query,
                    context=session_context,
                    previous_response=result["response"],
                    feedback=feedback,
                    failure_reason=failure_reason,
                    retrieved_docs=retrieved_docs,
                    retry_count=retry_count
                )
                
                if corrected_result:
                    result = corrected_result
                else:
                    # Retry failed, use fallback
                    logger.warning("Retry generation failed, using fallback")
                    break

            # Update result with guardrails outcome
            if not is_valid:
                result["response"] = final_response
                result["confidence"] = guardrails_metadata.get("enhanced_confidence", 0.3)
                result["metadata"]["guardrails_failed"] = True
            
            result["metadata"]["guardrails_metadata"] = guardrails_metadata
            result["metadata"]["retry_count"] = retry_count
            result["metadata"]["initial_confidence"] = initial_confidence
            result["metadata"]["final_confidence"] = result["confidence"]
            result["metadata"]["improvement_delta"] = result["confidence"] - initial_confidence
            result["metadata"]["correction_history"] = correction_history
            
            if retry_count > 0:
                result["metadata"]["self_corrected"] = True
                logger.info(
                    f"Self-correction summary: "
                    f"retries={retry_count}, "
                    f"initial_conf={initial_confidence:.2f}, "
                    f"final_conf={result['confidence']:.2f}, "
                    f"improvement={result['confidence'] - initial_confidence:+.2f}"
                )

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
    
    def _retry_with_feedback(
        self,
        intent: Intent,
        query: str,
        context: Dict,
        previous_response: str,
        feedback: str,
        failure_reason: str,
        retrieved_docs: List[str],
        retry_count: int
    ) -> Optional[Dict]:
        """
        Retry agent execution with feedback from guardrails.
        
        Args:
            intent: Query intent
            query: Original user query
            context: Session context
            previous_response: Previous failed response
            feedback: Feedback from guardrails
            failure_reason: Reason for failure
            retrieved_docs: Retrieved context documents
            retry_count: Current retry attempt number
            
        Returns:
            Corrected result dictionary or None on failure
        """
        logger.info(f"Retrying with feedback (attempt {retry_count})")
        
        try:
            # Build retry context with feedback
            retry_context = {
                **context,
                "retry_attempt": retry_count,
                "previous_response": previous_response,
                "failure_reason": failure_reason,
                "feedback": feedback,
                "retrieved_docs": retrieved_docs,
                "is_retry": True
            }
            
            # Execute agent with retry context
            if intent == Intent.RAG_QUERY:
                # For RAG queries, regenerate with stricter grounding
                result = self._execute_rag_retry(
                    query=query,
                    context=retry_context,
                    retrieved_docs=retrieved_docs,
                    feedback=feedback
                )
            elif intent == Intent.TOOL_EXECUTION:
                # For tool execution, retry with same tools but better synthesis
                result = self._execute_tool_retry(
                    query=query,
                    context=retry_context,
                    feedback=feedback
                )
            else:
                # For other intents, use standard execution
                result = self._execute_agent(intent, query, retry_context)
            
            return result
            
        except Exception as e:
            logger.error(f"Retry execution failed: {e}")
            return None
    
    def _execute_rag_retry(
        self,
        query: str,
        context: Dict,
        retrieved_docs: List[str],
        feedback: str
    ) -> Dict:
        """
        Execute RAG agent with retry-specific prompt enhancement.
        
        Args:
            query: User query
            context: Retry context with feedback
            retrieved_docs: Retrieved documents
            feedback: Feedback from guardrails
            
        Returns:
            Retry result dictionary
        """
        agent = self.agents.get(Intent.RAG_QUERY)
        
        if not agent:
            return {
                "success": False,
                "response": "RAG agent not available",
                "confidence": 0.0,
                "agent": "none",
                "metadata": {}
            }
        
        # Build enhanced prompt with feedback
        enhanced_query = self._build_retry_prompt(
            query=query,
            previous_response=context.get("previous_response", ""),
            feedback=feedback,
            failure_reason=context.get("failure_reason", "")
        )
        
        # Execute with enhanced prompt and existing context
        result = agent.execute(
            query=enhanced_query,
            context=context,
            use_cache=False,  # Don't use cache for retries
            retrieved_docs_override=retrieved_docs  # Use same retrieved docs
        )
        
        return result
    
    def _execute_tool_retry(
        self,
        query: str,
        context: Dict,
        feedback: str
    ) -> Dict:
        """
        Execute tool agent with retry-specific improvements.
        
        Args:
            query: User query
            context: Retry context with feedback
            feedback: Feedback from guardrails
            
        Returns:
            Retry result dictionary
        """
        agent = self.agents.get(Intent.TOOL_EXECUTION)
        
        if not agent:
            return {
                "success": False,
                "response": "Tool agent not available",
                "confidence": 0.0,
                "agent": "none",
                "metadata": {}
            }
        
        # Add feedback to context for tool agent
        retry_context = {
            **context,
            "retry_feedback": feedback
        }
        
        # Execute with retry context
        result = agent.execute(query, retry_context)
        
        return result
    
    def _build_retry_prompt(
        self,
        query: str,
        previous_response: str,
        feedback: str,
        failure_reason: str
    ) -> str:
        """
        Build enhanced prompt for retry with feedback.
        
        Args:
            query: Original user query
            previous_response: Previous failed response
            feedback: Feedback from guardrails
            failure_reason: Reason for failure
            
        Returns:
            Enhanced prompt string
        """
        retry_prompt = f"""RETRY REQUEST - Previous response was rejected.

Original Question: {query}

Previous Response (REJECTED):
{previous_response}

Failure Reason: {failure_reason}

Feedback for Improvement:
{feedback}

INSTRUCTIONS FOR RETRY:
1. Carefully review the feedback above
2. Address the specific issues identified
3. Use ONLY information from the provided context
4. Be more thorough and detailed in your response
5. Ensure all facts are directly supported by the context
6. Do not add external knowledge or assumptions

Please regenerate a better response that addresses the feedback:"""
        
        return retry_prompt

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

    def get_session_stats(self, session_id: str) -> Dict:
        """
        Get statistics for a session.

        Args:
            session_id: Session identifier

        Returns:
            Session statistics
        """
        try:
            session = self.session_manager.get_session(session_id)
            
            if not session:
                return {
                    "error": "Session not found",
                    "session_id": session_id
                }
            
            history = session.get("history", [])
            
            # Calculate stats
            total_queries = len([h for h in history if h.get("role") == "user"])
            total_responses = len([h for h in history if h.get("role") == "assistant"])
            
            intents = [h.get("metadata", {}).get("intent") for h in history if h.get("metadata", {}).get("intent")]
            intent_counts = {}
            for intent in intents:
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            agents = [h.get("metadata", {}).get("agent") for h in history if h.get("metadata", {}).get("agent")]
            agent_counts = {}
            for agent in agents:
                agent_counts[agent] = agent_counts.get(agent, 0) + 1
            
            return {
                "session_id": session_id,
                "created_at": session.get("created_at"),
                "total_queries": total_queries,
                "total_responses": total_responses,
                "intent_distribution": intent_counts,
                "agent_distribution": agent_counts,
                "conversation_length": len(history)
            }
            
        except Exception as e:
            logger.error(f"Failed to get session stats: {e}")
            return {
                "error": str(e),
                "session_id": session_id
            }
    
    def get_correction_stats(self) -> Dict:
        """
        Get self-correction statistics.
        
        Returns:
            Dictionary with correction statistics
        """
        total_attempts = (
            self.correction_stats["successful_corrections"] +
            self.correction_stats["failed_corrections"]
        )
        
        return {
            **self.correction_stats,
            "total_correction_attempts": total_attempts,
            "success_rate": (
                self.correction_stats["successful_corrections"] / total_attempts
                if total_attempts > 0 else 0.0
            )
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
