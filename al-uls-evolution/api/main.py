"""
AL-ULS Evolution API

FastAPI application for the AL-ULS Evolution system with Julia symbolic integration.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import torch
import numpy as np
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.al_uls_orchestrator import ALULSEvolutionSystem, ALULSConfig
from services.al_uls_client import ALULSClient

app = FastAPI(
    title="AL-ULS Evolution API",
    description="Adaptive Learning with Universal Symbolic Constraints - Evolution System",
    version="1.0.0"
)

# Global system instance
al_uls_system: Optional[ALULSEvolutionSystem] = None
julia_client: Optional[ALULSClient] = None


# Request/Response Models
class InitSystemRequest(BaseModel):
    input_dim: int = 64
    hidden_dims: List[int] = [128, 256, 128]
    output_dim: int = 10
    base_lr: float = 1e-3
    entropy_target: float = 3.5
    use_chaos_rag: bool = True
    use_symbolic_reasoning: bool = True


class TrainStepRequest(BaseModel):
    batch_data: List[List[float]]
    batch_targets: List[float]  # or List[List[float]] for regression


class EvaluateRequest(BaseModel):
    eval_data: List[List[float]]
    eval_targets: List[float]


class ChaosQueryRequest(BaseModel):
    query_vector: List[float]
    top_k: int = 10


class SymbolicConstraintRequest(BaseModel):
    expression: str
    variables: List[str]
    initial_strength: float = 0.5


class SymbolicEvalRequest(BaseModel):
    expression: str


# API Endpoints

@app.on_event("startup")
async def startup_event():
    """Initialize Julia client on startup"""
    global julia_client
    julia_client = ALULSClient()
    # Test connection
    health = await julia_client.health()
    if health.get('ok'):
        print("✅ Connected to Julia symbolic server")
    else:
        print("⚠️ Julia symbolic server not available")


@app.get("/")
async def root():
    return {
        "service": "AL-ULS Evolution API",
        "version": "1.0.0",
        "status": "operational",
        "components": {
            "symbolic_memory": True,
            "neural_symbolic_hybrid": True,
            "entropy_optimizer": True,
            "chaos_rag": True,
            "matrix_optimizer": True,
            "julia_integration": julia_client is not None
        }
    }


@app.post("/system/initialize")
async def initialize_system(config: InitSystemRequest):
    """Initialize the AL-ULS Evolution system"""
    global al_uls_system
    
    try:
        al_uls_config = ALULSConfig(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.output_dim,
            base_lr=config.base_lr,
            entropy_target=config.entropy_target,
            use_chaos_rag=config.use_chaos_rag,
            use_symbolic_reasoning=config.use_symbolic_reasoning
        )
        
        al_uls_system = ALULSEvolutionSystem(al_uls_config)
        
        return {
            "status": "initialized",
            "config": {
                "input_dim": config.input_dim,
                "hidden_dims": config.hidden_dims,
                "output_dim": config.output_dim,
                "entropy_target": config.entropy_target
            },
            "system_snapshot": al_uls_system.get_system_snapshot()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")


@app.post("/train/step")
async def training_step(request: TrainStepRequest):
    """Perform a single training step"""
    if al_uls_system is None:
        raise HTTPException(status_code=400, detail="System not initialized. Call /system/initialize first.")
    
    try:
        # Convert to tensors
        batch_data = torch.tensor(request.batch_data, dtype=torch.float32)
        batch_targets = torch.tensor(request.batch_targets, dtype=torch.long if len(np.array(request.batch_targets).shape) == 1 else torch.float32)
        
        # Perform training step
        metrics = al_uls_system.training_step(batch_data, batch_targets)
        
        return {
            "status": "success",
            "metrics": metrics,
            "step": al_uls_system.current_step,
            "epoch": al_uls_system.current_epoch
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training step failed: {str(e)}")


@app.post("/evaluate")
async def evaluate(request: EvaluateRequest):
    """Evaluate the model"""
    if al_uls_system is None:
        raise HTTPException(status_code=400, detail="System not initialized.")
    
    try:
        eval_data = torch.tensor(request.eval_data, dtype=torch.float32)
        eval_targets = torch.tensor(request.eval_targets, dtype=torch.long if len(np.array(request.eval_targets).shape) == 1 else torch.float32)
        
        metrics = al_uls_system.evaluate(eval_data, eval_targets)
        
        return {
            "status": "success",
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@app.post("/chaos/query")
async def chaos_query(request: ChaosQueryRequest):
    """Perform chaos-augmented retrieval"""
    if al_uls_system is None:
        raise HTTPException(status_code=400, detail="System not initialized.")
    
    if not al_uls_system.chaos_rag:
        raise HTTPException(status_code=400, detail="Chaos RAG not enabled in this system.")
    
    try:
        query_vector = np.array(request.query_vector)
        result = await al_uls_system.chaos_augmented_query(query_vector, top_k=request.top_k)
        
        return {
            "status": "success",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chaos query failed: {str(e)}")


@app.get("/symbolic/constraints")
async def get_symbolic_constraints():
    """Get current symbolic constraints"""
    if al_uls_system is None:
        raise HTTPException(status_code=400, detail="System not initialized.")
    
    constraints_state = al_uls_system.symbolic_memory.get_constraint_state()
    top_constraints = al_uls_system.symbolic_memory.get_top_constraints(n=20)
    
    return {
        "status": "success",
        "num_constraints": len(constraints_state.constraints),
        "entropy": constraints_state.entropy,
        "coherence": constraints_state.coherence,
        "top_constraints": [
            {
                "id": cid,
                "expression": c.expression,
                "variables": c.variables,
                "strength": c.strength,
                "confidence": c.confidence
            }
            for cid, c in top_constraints
        ]
    }


@app.post("/symbolic/add_constraint")
async def add_symbolic_constraint(request: SymbolicConstraintRequest):
    """Add a new symbolic constraint"""
    if al_uls_system is None:
        raise HTTPException(status_code=400, detail="System not initialized.")
    
    try:
        constraint = al_uls_system.symbolic_memory.create_constraint(
            expression=request.expression,
            variables=request.variables,
            initial_strength=request.initial_strength
        )
        
        return {
            "status": "success",
            "constraint": {
                "expression": constraint.expression,
                "variables": constraint.variables,
                "strength": constraint.strength,
                "confidence": constraint.confidence
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add constraint: {str(e)}")


@app.post("/symbolic/eval")
async def symbolic_eval(request: SymbolicEvalRequest):
    """Evaluate symbolic expression using Julia backend"""
    if julia_client is None:
        raise HTTPException(status_code=503, detail="Julia symbolic server not available")
    
    try:
        # Parse the expression
        parse_result = await julia_client.parse(request.expression)
        
        if not parse_result.get('ok'):
            return {
                "status": "error",
                "error": parse_result.get('error', 'Unknown parse error')
            }
        
        parsed = parse_result.get('parsed', {})
        name = parsed.get('name')
        args = parsed.get('args', [])
        
        if name:
            # Evaluate the symbolic call
            eval_result = await julia_client.eval(name, args)
            return {
                "status": "success",
                "parsed": parsed,
                "result": eval_result
            }
        else:
            return {
                "status": "success",
                "parsed": parsed,
                "result": None
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Symbolic evaluation failed: {str(e)}")


@app.get("/matrix/structures")
async def get_matrix_structures():
    """Get discovered matrix structures"""
    if al_uls_system is None:
        raise HTTPException(status_code=400, detail="System not initialized.")
    
    try:
        matrix_info = al_uls_system.matrix_optimizer.matrix_info
        compression_summary = al_uls_system.matrix_optimizer.get_compression_summary()
        symbolic_forms = al_uls_system.matrix_optimizer.export_symbolic_forms()
        
        return {
            "status": "success",
            "matrix_structures": matrix_info,
            "compression_summary": compression_summary,
            "symbolic_forms": symbolic_forms
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get matrix structures: {str(e)}")


@app.get("/system/snapshot")
async def get_system_snapshot():
    """Get complete system snapshot"""
    if al_uls_system is None:
        raise HTTPException(status_code=400, detail="System not initialized.")
    
    try:
        snapshot = al_uls_system.get_system_snapshot()
        return {
            "status": "success",
            "snapshot": snapshot
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get snapshot: {str(e)}")


@app.get("/system/symbolic_knowledge")
async def get_symbolic_knowledge():
    """Export all symbolic knowledge"""
    if al_uls_system is None:
        raise HTTPException(status_code=400, detail="System not initialized.")
    
    try:
        knowledge = al_uls_system.export_symbolic_knowledge()
        return {
            "status": "success",
            "knowledge": knowledge
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export knowledge: {str(e)}")


@app.get("/entropy/stats")
async def get_entropy_stats():
    """Get entropy statistics"""
    if al_uls_system is None:
        raise HTTPException(status_code=400, detail="System not initialized.")
    
    try:
        stats = al_uls_system.optimizer.get_entropy_statistics()
        chaos_signal = al_uls_system.optimizer.get_chaos_signal()
        should_explore = al_uls_system.optimizer.should_explore()
        
        return {
            "status": "success",
            "entropy_stats": stats,
            "chaos_signal": chaos_signal,
            "exploration_mode": should_explore
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get entropy stats: {str(e)}")


@app.post("/system/save")
async def save_checkpoint(path: str = "/app/data/checkpoint.pt"):
    """Save system checkpoint"""
    if al_uls_system is None:
        raise HTTPException(status_code=400, detail="System not initialized.")
    
    try:
        al_uls_system.save_checkpoint(path)
        return {
            "status": "success",
            "path": path,
            "step": al_uls_system.current_step
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save checkpoint: {str(e)}")


@app.post("/system/load")
async def load_checkpoint(path: str = "/app/data/checkpoint.pt"):
    """Load system checkpoint"""
    global al_uls_system
    
    try:
        if al_uls_system is None:
            al_uls_system = ALULSEvolutionSystem()
        
        al_uls_system.load_checkpoint(path)
        
        return {
            "status": "success",
            "path": path,
            "step": al_uls_system.current_step,
            "epoch": al_uls_system.current_epoch
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load checkpoint: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    julia_health = await julia_client.health() if julia_client else {"ok": False}
    
    return {
        "status": "healthy",
        "system_initialized": al_uls_system is not None,
        "julia_backend": julia_health.get('ok', False),
        "components": {
            "symbolic_memory": al_uls_system is not None,
            "neural_hybrid": al_uls_system is not None,
            "entropy_optimizer": al_uls_system is not None,
            "chaos_rag": al_uls_system is not None and al_uls_system.chaos_rag is not None,
            "matrix_optimizer": al_uls_system is not None
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
