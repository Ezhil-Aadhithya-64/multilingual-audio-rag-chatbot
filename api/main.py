"""
FastAPI Backend - Production REST API for VoiceAssist Pro
Exposes endpoints for query processing, ingestion, and metrics
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import logging
import uuid
from datetime import datetime
from pathlib import Path
import base64

# Import system components
from services.orchestrator import AgenticOrchestrator
from core.session import SessionManager
from evaluation.metrics import MetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="VoiceAssist Pro API",
    description="Production-grade Agentic AI System for Multilingual Document QA",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize system components (would be dependency injection in production)
orchestrator = None
session_manager = None
metrics_collector = None


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Text query request model"""
    query: str = Field(..., description="User query text")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    language: Optional[str] = Field(None, description="Query language (auto-detect if None)")
    return_audio: bool = Field(False, description="Generate audio response")
    use_cache: bool = Field(True, description="Use cached context if available")


class AudioQueryRequest(BaseModel):
    """Audio query request model"""
    audio_data: str = Field(..., description="Base64-encoded audio data")
    session_id: Optional[str] = Field(None, description="Session ID")
    language: Optional[str] = Field(None, description="Audio language (auto-detect if None)")
    return_audio: bool = Field(False, description="Generate audio response")


class IngestRequest(BaseModel):
    """Document ingestion request model"""
    documents: List[str] = Field(..., description="List of document texts")
    metadatas: Optional[List[Dict]] = Field(None, description="Document metadata")
    chunk: bool = Field(True, description="Chunk documents before ingestion")


class QueryResponse(BaseModel):
    """Query response model"""
    query_id: str
    session_id: str
    success: bool
    response: str
    confidence: float
    intent: str
    agent: str
    audio_path: Optional[str] = None
    retrieved_documents: Optional[List[str]] = None
    metadata: Dict
    timestamp: str


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize system components on startup"""
    global orchestrator, session_manager, metrics_collector

    logger.info("Initializing VoiceAssist Pro API...")

    try:
        # Initialize components
        session_manager = SessionManager(storage_path=Path("data/sessions"))
        metrics_collector = MetricsCollector(storage_path=Path("data/metrics"))

        # Initialize orchestrator (would load all agents, tools, etc.)
        from services.orchestrator import AgenticOrchestrator
        orchestrator = AgenticOrchestrator(
            session_manager=session_manager,
            metrics_collector=metrics_collector
        )

        logger.info("VoiceAssist Pro API initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise


# Health check endpoint
@app.get("/api/v1/health")
async def health_check():
    """
    Health check endpoint.
    Returns system status and component health.
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0",
        "components": {
            "orchestrator": orchestrator is not None,
            "session_manager": session_manager is not None,
            "metrics_collector": metrics_collector is not None
        }
    }


# Query endpoint
@app.post("/api/v1/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    Process text query through agentic system.

    Args:
        request: Query request
        background_tasks: FastAPI background tasks

    Returns:
        Query response with result and metadata
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        # Generate query ID
        query_id = str(uuid.uuid4())

        # Get or create session
        session_id = request.session_id
        if not session_id:
            session_id = session_manager.create_session()
        elif not session_manager.get_session(session_id):
            session_id = session_manager.create_session(session_id)

        logger.info(f"Processing query {query_id}: {request.query[:100]}...")

        # Process query through orchestrator
        result = orchestrator.process_query(
            query=request.query,
            session_id=session_id,
            language=request.language,
            return_audio=request.return_audio,
            use_cache=request.use_cache
        )

        # Build response
        response = QueryResponse(
            query_id=query_id,
            session_id=session_id,
            success=result["success"],
            response=result["response"],
            confidence=result["confidence"],
            intent=result["intent"],
            agent=result["agent"],
            audio_path=result.get("audio_path"),
            retrieved_documents=result.get("retrieved_documents"),
            metadata=result.get("metadata", {}),
            timestamp=datetime.now().isoformat()
        )

        # Record metrics in background
        background_tasks.add_task(
            _record_metrics_async,
            query_id,
            request.query,
            result
        )

        return response

    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Audio query endpoint
@app.post("/api/v1/audio-query")
async def process_audio_query(request: AudioQueryRequest, background_tasks: BackgroundTasks):
    """
    Process audio query through agentic system.

    Args:
        request: Audio query request
        background_tasks: FastAPI background tasks

    Returns:
        Query response with transcription and result
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        # Generate query ID
        query_id = str(uuid.uuid4())

        # Decode audio data
        audio_bytes = base64.b64decode(request.audio_data)

        # Save audio temporarily
        audio_path = Path(f"data/audio_samples/query_{query_id}.wav")
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        # Get or create session
        session_id = request.session_id or session_manager.create_session()

        logger.info(f"Processing audio query {query_id}")

        # Process audio query
        result = orchestrator.process_audio_query(
            audio_path=audio_path,
            session_id=session_id,
            language=request.language,
            return_audio=request.return_audio
        )

        # Cleanup audio file in background
        background_tasks.add_task(_cleanup_file, audio_path)

        return {
            "query_id": query_id,
            "session_id": session_id,
            "transcription": result.get("transcription"),
            "detected_language": result.get("detected_language"),
            **result
        }

    except Exception as e:
        logger.error(f"Audio query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Ingestion endpoint
@app.post("/api/v1/ingest")
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents into knowledge base.

    Args:
        request: Ingestion request

    Returns:
        Ingestion result
    """
    if not orchestrator:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        logger.info(f"Ingesting {len(request.documents)} documents")

        # Ingest documents
        result = orchestrator.ingest_documents(
            documents=request.documents,
            metadatas=request.metadatas,
            chunk=request.chunk
        )

        return {
            "success": True,
            "num_documents": len(request.documents),
            "num_chunks": result.get("num_chunks", 0),
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Document ingestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Metrics endpoint
@app.get("/api/v1/metrics/{query_id}")
async def get_query_metrics(query_id: str):
    """
    Get metrics for specific query.

    Args:
        query_id: Query identifier

    Returns:
        Query metrics
    """
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics system not initialized")

    try:
        # Load metrics from storage
        metrics_file = Path(f"data/metrics/{query_id}.json")
        if not metrics_file.exists():
            raise HTTPException(status_code=404, detail="Metrics not found")

        import json
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        return metrics

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Aggregate metrics endpoint
@app.get("/api/v1/metrics")
async def get_aggregate_metrics(time_window: Optional[int] = None):
    """
    Get aggregate metrics across all queries.

    Args:
        time_window: Optional time window in seconds

    Returns:
        Aggregate metrics
    """
    if not metrics_collector:
        raise HTTPException(status_code=503, detail="Metrics system not initialized")

    try:
        metrics = metrics_collector.get_aggregate_metrics(time_window=time_window)
        return metrics

    except Exception as e:
        logger.error(f"Failed to retrieve aggregate metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Session endpoint
@app.get("/api/v1/session/{session_id}")
async def get_session(session_id: str):
    """
    Get session data and conversation history.

    Args:
        session_id: Session identifier

    Returns:
        Session data
    """
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session system not initialized")

    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": session_id,
        "created_at": session["created_at"],
        "last_activity": session["last_activity"],
        "conversation_history": session["conversation_history"],
        "stats": session_manager.get_session_stats(session_id)
    }


# Delete session endpoint
@app.delete("/api/v1/session/{session_id}")
async def delete_session(session_id: str):
    """
    Delete session and clear history.

    Args:
        session_id: Session identifier

    Returns:
        Deletion result
    """
    if not session_manager:
        raise HTTPException(status_code=503, detail="Session system not initialized")

    success = session_manager.clear_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "success": True,
        "session_id": session_id,
        "timestamp": datetime.now().isoformat()
    }


# Background task helpers
async def _record_metrics_async(query_id: str, query: str, result: Dict):
    """Record metrics in background"""
    try:
        metadata = result.get("metadata", {})
        guardrails_metadata = metadata.get("guardrails_metadata", {})
        
        metrics_collector.record_query_metrics(
            query_id=query_id,
            query=query,
            intent=result.get("intent", "unknown"),
            response=result.get("response", ""),
            confidence=result.get("confidence", 0.0),
            retrieved_docs=result.get("retrieved_documents", []),
            latency_breakdown=metadata.get("latency", {}),
            agent_name=result.get("agent", "unknown"),
            success=result.get("success", False),
            retrieval_scores=metadata.get("retrieval_scores", []),
            reranker_scores=metadata.get("reranker_scores", []),
            enhanced_confidence=guardrails_metadata.get("enhanced_confidence"),
            context_quality=guardrails_metadata.get("context_quality")
        )
    except Exception as e:
        logger.error(f"Failed to record metrics: {e}")


async def _cleanup_file(file_path: Path):
    """Cleanup temporary file"""
    try:
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        logger.error(f"Failed to cleanup file {file_path}: {e}")


# Run with: uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
