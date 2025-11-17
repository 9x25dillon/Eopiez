#!/usr/bin/env python3
"""
Eopiez Unified API Gateway

This gateway provides a single entry point for all Eopiez services:
- Motif Detection
- Message Vectorization
- LiMps Symbolic Memory
- QVNM (Quantum Vector Neural Memory)
- AL-ULS Evolution
- Context Selection
- Neutronics Surrogate

All endpoints follow the pattern: /api/v1/{service}/{action}
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
import httpx
import os
import logging

# ===========================================
# Configuration
# ===========================================

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Service URLs
JULIA_BACKEND_URL = os.environ.get("JULIA_BACKEND_URL", "http://localhost:8080")
AL_ULS_BACKEND_URL = os.environ.get("AL_ULS_BACKEND_URL", "http://localhost:8001")

# Create FastAPI app
app = FastAPI(
    title="Eopiez API Gateway",
    description="Unified API for Eopiez symbolic computation platform",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files if available
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(STATIC_DIR):
    app.mount("/ui", StaticFiles(directory=STATIC_DIR), name="ui")

# ===========================================
# Request/Response Models
# ===========================================

class MotifDetectRequest(BaseModel):
    text: str = Field(..., description="Text to analyze for motifs")
    categories: Optional[List[str]] = Field(None, description="Specific motif categories to detect")

class MotifDetectResponse(BaseModel):
    motifs: List[Dict[str, Any]] = Field(..., description="Detected motifs with scores")
    document_analysis: Dict[str, Any] = Field(..., description="Overall document analysis")

class VectorizeRequest(BaseModel):
    motifs: List[Dict[str, Any]] = Field(..., description="Motif tokens to vectorize")
    embedding_dim: int = Field(64, ge=1, le=512, description="Embedding dimension")
    entropy_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Entropy threshold")
    compression_ratio: float = Field(0.8, ge=0.0, le=1.0, description="Compression ratio")

class VectorizeResponse(BaseModel):
    vector: List[float] = Field(..., description="Generated vector representation")
    message_state: Dict[str, Any] = Field(..., description="Message state information")

class LiMpsStoreRequest(BaseModel):
    vector: List[float] = Field(..., description="Vector to store")
    text: str = Field(..., description="Original text")
    context: Dict[str, Any] = Field(default_factory=dict, description="Contextual metadata")

class LiMpsRetrieveRequest(BaseModel):
    query: str = Field(..., description="Query text")
    limit: int = Field(5, ge=1, le=100, description="Number of results")
    threshold: float = Field(0.0, ge=0.0, le=1.0, description="Similarity threshold")

class ContextSelectionRequest(BaseModel):
    candidates: List[Dict[str, str]] = Field(..., description="Candidate contexts with id and text")
    embedding_dim: int = Field(64, ge=1, le=512, description="Embedding dimension")

class HealthResponse(BaseModel):
    status: str
    services: Dict[str, str]
    version: str

# ===========================================
# Helper Functions
# ===========================================

async def call_service(url: str, method: str = "POST", json: Optional[Dict] = None, timeout: float = 120.0) -> Dict:
    """Call backend service with error handling"""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if method == "POST":
                response = await client.post(url, json=json)
            else:
                response = await client.get(url, params=json)
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Service call failed: {url} - {str(e)}")
        raise HTTPException(status_code=502, detail=f"Backend service error: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error calling service: {url} - {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

async def check_service_health(url: str) -> bool:
    """Check if a service is healthy"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{url}/health")
            return response.status_code == 200
    except:
        return False

# ===========================================
# Root & Health Endpoints
# ===========================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint - redirects to docs"""
    return """
    <html>
        <head>
            <meta http-equiv="refresh" content="0;url=/docs" />
        </head>
        <body>
            <h1>Eopiez API Gateway</h1>
            <p>Redirecting to <a href="/docs">API documentation</a>...</p>
        </body>
    </html>
    """

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check"""
    services = {
        "julia_backend": "ready" if await check_service_health(JULIA_BACKEND_URL) else "unavailable",
        "al_uls": "ready" if await check_service_health(AL_ULS_BACKEND_URL) else "unavailable",
    }

    overall_status = "healthy" if all(s == "ready" for s in services.values()) else "degraded"

    return {
        "status": overall_status,
        "services": services,
        "version": "1.0.0"
    }

# ===========================================
# API v1 - Motif Detection
# ===========================================

@app.post("/api/v1/motif/detect", response_model=MotifDetectResponse, tags=["Motif Detection"])
async def detect_motifs(request: MotifDetectRequest):
    """
    Detect symbolic motifs in text

    Identifies thematic patterns across categories:
    - Isolation & Loneliness
    - Snake Symbolism
    - Memory & Identity
    - Strands of Connection
    - Technology & Humanity
    - Phantom Pain
    - War & Peace
    - Boss Themes
    - Meta-Narrative
    """
    payload = {
        "documents": [request.text],
        "categories": request.categories
    }
    result = await call_service(f"{JULIA_BACKEND_URL}/motif/batch", json=payload)

    if "results" in result and len(result["results"]) > 0:
        return result["results"][0]
    else:
        return {"motifs": [], "document_analysis": {}}

@app.post("/api/v1/motif/batch", tags=["Motif Detection"])
async def detect_motifs_batch(texts: List[str]):
    """Batch motif detection for multiple texts"""
    payload = {"documents": texts}
    return await call_service(f"{JULIA_BACKEND_URL}/motif/batch", json=payload)

# ===========================================
# API v1 - Message Vectorization
# ===========================================

@app.post("/api/v1/vector/encode", response_model=VectorizeResponse, tags=["Vectorization"])
async def vectorize_motifs(request: VectorizeRequest):
    """
    Convert motifs to symbolic vector representation

    Transforms detected motifs into dense vector embeddings
    with entropy-based compression.
    """
    payload = {
        "motif_tokens": request.motifs,
        "embedding_dim": request.embedding_dim,
        "entropy_threshold": request.entropy_threshold,
        "compression_ratio": request.compression_ratio
    }
    return await call_service(f"{JULIA_BACKEND_URL}/motif/vectorize", json=payload)

@app.post("/api/v1/vector/pipeline", tags=["Vectorization"])
async def full_vectorization_pipeline(text: str, embedding_dim: int = 64):
    """
    Complete pipeline: text → motif detection → vectorization

    Convenience endpoint that combines detection and vectorization.
    """
    # Step 1: Detect motifs
    motif_result = await detect_motifs(MotifDetectRequest(text=text))

    # Step 2: Vectorize
    vector_result = await vectorize_motifs(VectorizeRequest(
        motifs=motif_result["motifs"],
        embedding_dim=embedding_dim
    ))

    return {
        "text": text,
        "motifs": motif_result["motifs"],
        "document_analysis": motif_result["document_analysis"],
        "vector": vector_result["vector"],
        "message_state": vector_result["message_state"]
    }

# ===========================================
# API v1 - LiMps Symbolic Memory
# ===========================================

@app.post("/api/v1/limps/store", tags=["LiMps Memory"])
async def store_memory(request: LiMpsStoreRequest):
    """
    Store symbolic memory in LiMps graph

    Creates a memory node with temporal and contextual links.
    """
    payload = {
        "vector": request.vector,
        "text": request.text,
        "context": request.context
    }
    return await call_service(f"{JULIA_BACKEND_URL}/limps/store", json=payload)

@app.get("/api/v1/limps/retrieve", tags=["LiMps Memory"])
async def retrieve_memories(
    query: str = Query(..., description="Query text"),
    limit: int = Query(5, ge=1, le=100, description="Number of results"),
    threshold: float = Query(0.0, ge=0.0, le=1.0, description="Similarity threshold")
):
    """
    Retrieve similar memories from LiMps graph

    Finds memories based on semantic similarity and temporal relevance.
    """
    payload = {
        "query": query,
        "limit": limit,
        "threshold": threshold
    }
    return await call_service(f"{JULIA_BACKEND_URL}/limps/retrieve", method="GET", json=payload)

@app.get("/api/v1/limps/graph", tags=["LiMps Memory"])
async def get_memory_graph():
    """Get the full memory graph structure"""
    return await call_service(f"{JULIA_BACKEND_URL}/limps/graph", method="GET")

@app.post("/api/v1/limps/relate", tags=["LiMps Memory"])
async def relate_memories(memory_id1: str, memory_id2: str, relationship: str):
    """Create a symbolic relationship between two memories"""
    payload = {
        "id1": memory_id1,
        "id2": memory_id2,
        "relationship": relationship
    }
    return await call_service(f"{JULIA_BACKEND_URL}/limps/relate", json=payload)

# ===========================================
# API v1 - QVNM (Quantum Vector Neural Memory)
# ===========================================

@app.post("/api/v1/qvnm/upload", tags=["QVNM"])
async def upload_vectors(file: UploadFile = File(...)):
    """
    Upload vector dataset (.npz or .jsonl)

    Creates a new QVNM session for analysis.
    """
    # Import the original QVNM upload logic
    from api import qvnm_upload_vectors as original_upload
    return await original_upload(file)

@app.post("/api/v1/qvnm/estimate/{session_id}", tags=["QVNM"])
async def estimate_intrinsic_dimension(
    session_id: str,
    k: int = 10,
    gamma: float = 0.5,
    alpha: float = 0.5,
    boots: int = 8,
    mode: str = "local"
):
    """
    Estimate intrinsic dimension using Costa-Hero algorithm

    Returns local or global dimension estimates.
    """
    payload = {
        "sid": session_id,
        "k": k,
        "gamma": gamma,
        "alpha": alpha,
        "boots": boots,
        "mode": mode
    }
    from api import proxy_estimate_id
    return await proxy_estimate_id(payload)

@app.post("/api/v1/qvnm/query/{session_id}", tags=["QVNM"])
async def query_nearest_neighbors(
    session_id: str,
    seed_id: str,
    topk: int = 5,
    steps: int = 10,
    alpha: float = 0.85
):
    """
    Query for nearest neighbors using manifold-aware search
    """
    payload = {
        "sid": session_id,
        "seed_id": seed_id,
        "topk": topk,
        "steps": steps,
        "alpha": alpha
    }
    from api import proxy_query
    return await proxy_query(payload)

@app.post("/api/v1/qvnm/preview/{session_id}", tags=["QVNM"])
async def build_manifold_preview(
    session_id: str,
    r: int = 2,
    k_eval: int = 10,
    bins: int = 20,
    lambda_m: float = 0.3,
    lambda_h: float = 0.3
):
    """
    Build 2D preview of high-dimensional manifold
    """
    payload = {
        "sid": session_id,
        "r": r,
        "k_eval": k_eval,
        "bins": bins,
        "lambda_m": lambda_m,
        "lambda_h": lambda_h
    }
    from api import proxy_build_preview
    return await proxy_build_preview(payload)

# ===========================================
# API v1 - AL-ULS Evolution
# ===========================================

@app.post("/api/v1/al-uls/train", tags=["AL-ULS"])
async def train_adaptive_model(
    training_data: List[Dict[str, Any]],
    constraints: Optional[Dict[str, Any]] = None
):
    """
    Train AL-ULS adaptive learning model

    Combines neural networks with symbolic constraints
    for self-evolving learning.
    """
    payload = {
        "training_data": training_data,
        "constraints": constraints or {}
    }
    return await call_service(f"{AL_ULS_BACKEND_URL}/train", json=payload)

@app.post("/api/v1/al-uls/infer/{model_id}", tags=["AL-ULS"])
async def run_inference(model_id: str, input_data: Dict[str, Any]):
    """
    Run inference with trained AL-ULS model
    """
    payload = {
        "model_id": model_id,
        "input": input_data
    }
    return await call_service(f"{AL_ULS_BACKEND_URL}/infer", json=payload)

@app.get("/api/v1/al-uls/models", tags=["AL-ULS"])
async def list_models():
    """List available AL-ULS models"""
    return await call_service(f"{AL_ULS_BACKEND_URL}/models", method="GET")

@app.post("/api/v1/al-uls/evolve/{model_id}", tags=["AL-ULS"])
async def evolve_constraints(model_id: str):
    """
    Trigger constraint evolution for a model

    Uses symbolic evolution to adapt constraints.
    """
    payload = {"model_id": model_id}
    return await call_service(f"{AL_ULS_BACKEND_URL}/evolve", json=payload)

# ===========================================
# API v1 - Context Selection
# ===========================================

@app.post("/api/v1/context/select", tags=["Context Selection"])
async def select_contexts(request: ContextSelectionRequest):
    """
    Select optimal contexts using motif-based entropy scoring

    Ranks candidate contexts by information density and symbolic richness.
    """
    payload = {
        "candidates": request.candidates,
        "embedding_dim": request.embedding_dim
    }

    # Use the original dual context selection logic
    from api import dual_select_contexts
    return await dual_select_contexts(payload)

@app.post("/api/v1/context/compare", tags=["Context Selection"])
async def compare_contexts(context1: str, context2: str):
    """
    Compare two contexts and explain the differences
    """
    candidates = [
        {"id": "context1", "text": context1},
        {"id": "context2", "text": context2}
    ]
    result = await select_contexts(ContextSelectionRequest(candidates=candidates))

    return {
        "winner": result["ranked"][0]["id"],
        "context1_score": next(r["score"] for r in result["ranked"] if r["id"] == "context1"),
        "context2_score": next(r["score"] for r in result["ranked"] if r["id"] == "context2"),
        "analysis": result["ranked"]
    }

# ===========================================
# API v1 - Neutronics (if available)
# ===========================================

@app.post("/api/v1/neutronics/predict", tags=["Neutronics"])
async def predict_flux(parameters: Dict[str, float]):
    """
    Predict neutron flux using surrogate model

    Input: Reactor parameters
    Output: Flux predictions
    """
    try:
        return await call_service(f"{JULIA_BACKEND_URL}/neutronics/predict", json=parameters)
    except HTTPException as e:
        if e.status_code == 502:
            raise HTTPException(status_code=404, detail="Neutronics service not available")
        raise

# ===========================================
# Legacy Compatibility Endpoints
# ===========================================

@app.post("/dual/select_contexts", tags=["Legacy"])
async def legacy_dual_select(payload: Dict[str, Any]):
    """Legacy endpoint for backwards compatibility"""
    from api import dual_select_contexts
    return await dual_select_contexts(payload)

@app.post("/qvnm/upload_vectors", tags=["Legacy"])
async def legacy_qvnm_upload(file: UploadFile = File(...)):
    """Legacy QVNM upload endpoint"""
    from api import qvnm_upload_vectors
    return await qvnm_upload_vectors(file)

# ===========================================
# Main Entry Point
# ===========================================

if __name__ == "__main__":
    port = int(os.environ.get("API_PORT", "8000"))
    workers = int(os.environ.get("API_WORKERS", "4"))

    logger.info(f"Starting Eopiez API Gateway on port {port}")
    logger.info(f"Julia Backend: {JULIA_BACKEND_URL}")
    logger.info(f"AL-ULS Backend: {AL_ULS_BACKEND_URL}")

    uvicorn.run(
        "api_gateway:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        log_level=os.environ.get("LOG_LEVEL", "info").lower()
    )
