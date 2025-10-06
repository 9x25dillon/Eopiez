"""
AL-ULS Evolution Orchestrator

Unified orchestration layer that brings together all components:
- Self-Evolving Symbolic Memory
- Hybrid Neural-Symbolic Reasoning
- Entropy-Guided Optimization
- Chaos RAG
- Matrix Symbolic Optimization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import asyncio
import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.symbolic_evolution import SelfEvolvingSymbolicMemory
from core.neural_symbolic_hybrid import NeuralSymbolicHybrid, SymbolicGradientModifier
from core.entropy_guided_optimizer import EntropyGuidedOptimizer
from core.chaos_rag import ChaosRAG
from core.matrix_symbolic_optimizer import MatrixSymbolicOptimizer


@dataclass
class TrainingState:
    """Complete state of the AL-ULS system during training"""
    epoch: int
    step: int
    loss: float
    learning_rate: float
    symbolic_constraints: Dict[str, Any]
    entropy_metrics: Dict[str, float]
    chaos_metrics: Dict[str, float]
    structure_metrics: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class ALULSConfig:
    """Configuration for AL-ULS Evolution system"""
    # Model architecture
    input_dim: int = 64
    hidden_dims: List[int] = field(default_factory=lambda: [128, 256, 128])
    output_dim: int = 10
    
    # Symbolic evolution
    evolution_rate: float = 0.1
    max_constraints: int = 1000
    entropy_target: float = 3.5
    
    # Optimization
    base_lr: float = 1e-3
    entropy_tolerance: float = 0.5
    
    # Chaos RAG
    chaos_strength: float = 0.5
    symbolic_weight: float = 0.3
    
    # Matrix optimization
    use_polynomial: bool = True
    polynomial_degree: int = 2
    structure_weight: float = 0.1
    
    # General
    use_symbolic_reasoning: bool = True
    use_chaos_rag: bool = True
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class ALULSEvolutionSystem:
    """
    Complete AL-ULS Evolution System.
    
    Integrates all components into a unified adaptive learning system
    with symbolic constraints.
    """
    
    def __init__(self, config: Optional[ALULSConfig] = None):
        self.config = config or ALULSConfig()
        self.device = torch.device(self.config.device)
        
        # Core components
        self._initialize_components()
        
        # Training state
        self.training_states: List[TrainingState] = []
        self.current_epoch = 0
        self.current_step = 0
        
        # Statistics
        self.total_training_time = 0.0
        self.best_loss = float('inf')
        self.best_state = None
        
    def _initialize_components(self):
        """Initialize all AL-ULS components"""
        
        # 1. Self-Evolving Symbolic Memory
        self.symbolic_memory = SelfEvolvingSymbolicMemory(
            evolution_rate=self.config.evolution_rate,
            max_constraints=self.config.max_constraints,
            entropy_target=self.config.entropy_target
        )
        
        # 2. Neural-Symbolic Hybrid Network
        self.model = NeuralSymbolicHybrid(
            input_dim=self.config.input_dim,
            hidden_dims=self.config.hidden_dims,
            output_dim=self.config.output_dim,
            use_symbolic=self.config.use_symbolic_reasoning
        ).to(self.device)
        
        # 3. Entropy-Guided Optimizer
        self.optimizer = EntropyGuidedOptimizer(
            parameters=self.model.parameters(),
            base_lr=self.config.base_lr,
            entropy_target=self.config.entropy_target,
            entropy_tolerance=self.config.entropy_tolerance
        )
        
        # Register hooks for activation tracking
        self.optimizer.register_activation_hooks(self.model)
        
        # 4. Symbolic Gradient Modifier
        self.gradient_modifier = SymbolicGradientModifier(self.symbolic_memory)
        
        # 5. Chaos RAG (for knowledge augmentation)
        if self.config.use_chaos_rag:
            self.chaos_rag = ChaosRAG(
                embedding_dim=self.config.hidden_dims[-1],
                chaos_strength=self.config.chaos_strength,
                symbolic_weight=self.config.symbolic_weight,
                entropy_target=self.config.entropy_target
            )
        else:
            self.chaos_rag = None
        
        # 6. Matrix Symbolic Optimizer
        self.matrix_optimizer = MatrixSymbolicOptimizer(
            model=self.model,
            use_polynomial=self.config.use_polynomial,
            polynomial_degree=self.config.polynomial_degree
        )
        
        # Initial analysis
        self.matrix_optimizer.analyze_model()
        
    def training_step(
        self,
        batch_data: torch.Tensor,
        batch_targets: torch.Tensor,
        loss_fn: Optional[nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Perform one training step with full AL-ULS evolution.
        
        Args:
            batch_data: Input batch [B, input_dim]
            batch_targets: Target batch [B, output_dim] or [B] for classification
            loss_fn: Loss function (defaults to CrossEntropyLoss)
        
        Returns:
            Dictionary with step metrics
        """
        start_time = time.time()
        self.current_step += 1
        
        # Move to device
        batch_data = batch_data.to(self.device)
        batch_targets = batch_targets.to(self.device)
        
        # Default loss function
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss(reduction='none')
        
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Get current symbolic constraints
        symbolic_state = self.symbolic_memory.get_constraint_state()
        symbolic_constraints = {
            'bounded': True,
            'lower_bound': -10.0,
            'upper_bound': 10.0,
            'sparse': self.current_step % 10 == 0  # Encourage sparsity periodically
        }
        
        # Forward pass with symbolic reasoning
        outputs, reasoning_trace = self.model(
            batch_data,
            symbolic_constraints=symbolic_constraints,
            return_trace=True
        )
        
        # Compute loss
        if batch_targets.dim() == 1 and outputs.dim() == 2:
            # Classification
            losses = loss_fn(outputs, batch_targets)
        else:
            # Regression or other
            losses = ((outputs - batch_targets) ** 2).mean(dim=-1)
        
        primary_loss = losses.mean()
        
        # Add structural regularization
        structural_info = self.matrix_optimizer.optimize_with_structure(
            structure_weight=self.config.structure_weight
        )
        structural_loss = structural_info['total_loss']
        
        # Total loss
        total_loss = primary_loss + structural_loss
        
        # Backward pass
        total_loss.backward()
        
        # Modify gradients based on symbolic constraints
        symbolic_state_dict = self.symbolic_memory.export_state()
        grad_modifications = self.gradient_modifier.modify_gradients(
            self.model,
            symbolic_state_dict
        )
        
        # Compute gradient statistics
        grad_norms = self._compute_gradient_norms()
        
        # Entropy-guided optimization step
        step_stats = self.optimizer.step(
            predictions=outputs,
            losses=losses,
            compute_metrics_flag=True
        )
        
        # Evolve symbolic memory based on training feedback
        evolution_feedback = {
            'loss_delta': (self.best_loss - float(primary_loss.item())) if self.best_loss != float('inf') else 0.0,
            'entropy': step_stats.get('total_entropy', 0.0),
            'gradient_norms': grad_norms,
            'performance_metrics': {
                'loss': float(primary_loss.item()),
                'structural_loss': float(structural_loss) if isinstance(structural_loss, torch.Tensor) else structural_loss
            }
        }
        
        evolution_stats = self.symbolic_memory.evolve_constraints(evolution_feedback)
        
        # Update best loss
        if primary_loss.item() < self.best_loss:
            self.best_loss = primary_loss.item()
            self.best_state = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.optimizer.state_dict(),
                'step': self.current_step,
                'epoch': self.current_epoch
            }
        
        # Chaos RAG update (if enabled)
        chaos_metrics = {}
        if self.chaos_rag:
            # Add training examples to knowledge base
            for i in range(min(5, len(batch_data))):  # Add a few samples
                self.chaos_rag.add_to_knowledge_base({
                    'embedding': batch_data[i].detach().cpu().numpy(),
                    'text': f"training_sample_{self.current_step}_{i}",
                    'metadata': {'step': self.current_step, 'loss': losses[i].item()}
                })
            
            chaos_metrics = self.chaos_rag.get_system_state()
        
        # Record training state
        training_state = TrainingState(
            epoch=self.current_epoch,
            step=self.current_step,
            loss=float(primary_loss.item()),
            learning_rate=step_stats['learning_rate'],
            symbolic_constraints=symbolic_state_dict['statistics'],
            entropy_metrics=step_stats,
            chaos_metrics=chaos_metrics,
            structure_metrics=structural_info
        )
        self.training_states.append(training_state)
        
        # Compute step time
        step_time = time.time() - start_time
        self.total_training_time += step_time
        
        # Compile metrics
        metrics = {
            'loss': float(primary_loss.item()),
            'structural_loss': float(structural_loss) if isinstance(structural_loss, torch.Tensor) else structural_loss,
            'total_loss': float(total_loss.item()),
            'learning_rate': step_stats['learning_rate'],
            'entropy': step_stats.get('total_entropy', 0.0),
            'gradient_norm': grad_norms.get('total', 0.0),
            'num_constraints': evolution_stats['num_constraints'],
            'constraint_evolution_rate': evolution_stats['avg_improvement'],
            'step_time': step_time,
            'reasoning_confidence': reasoning_trace.confidence if reasoning_trace else 0.0
        }
        
        return metrics
    
    def _compute_gradient_norms(self) -> Dict[str, float]:
        """Compute gradient norm statistics"""
        total_norm = 0.0
        max_norm = 0.0
        layer_norms = []
        
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
                max_norm = max(max_norm, param_norm.item())
                layer_norms.append(param_norm.item())
        
        total_norm = total_norm ** 0.5
        
        return {
            'total': total_norm,
            'max': max_norm,
            'mean': np.mean(layer_norms) if layer_norms else 0.0,
            'variance': np.var(layer_norms) if layer_norms else 0.0
        }
    
    def evaluate(
        self,
        eval_data: torch.Tensor,
        eval_targets: torch.Tensor,
        loss_fn: Optional[nn.Module] = None
    ) -> Dict[str, Any]:
        """
        Evaluate the model without training.
        """
        self.model.eval()
        
        with torch.no_grad():
            eval_data = eval_data.to(self.device)
            eval_targets = eval_targets.to(self.device)
            
            if loss_fn is None:
                loss_fn = nn.CrossEntropyLoss()
            
            outputs, _ = self.model(eval_data, return_trace=False)
            
            if eval_targets.dim() == 1 and outputs.dim() == 2:
                loss = loss_fn(outputs, eval_targets)
                # Classification accuracy
                preds = outputs.argmax(dim=-1)
                accuracy = (preds == eval_targets).float().mean()
                metrics = {
                    'eval_loss': float(loss.item()),
                    'eval_accuracy': float(accuracy.item())
                }
            else:
                loss = ((outputs - eval_targets) ** 2).mean()
                metrics = {
                    'eval_loss': float(loss.item()),
                    'eval_mse': float(loss.item())
                }
        
        self.model.train()
        return metrics
    
    async def chaos_augmented_query(
        self,
        query: np.ndarray,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Perform chaos-augmented retrieval for knowledge augmentation.
        """
        if not self.chaos_rag:
            return {'error': 'Chaos RAG not enabled'}
        
        context = await self.chaos_rag.async_retrieval(
            query=query,
            symbolic_constraints={
                'diversity': {'type': 'diversity', 'count': top_k},
                'temporal': {'type': 'temporal', 'preference': 'recent'}
            },
            top_k=top_k
        )
        
        return {
            'retrieved_items': context.retrieved_items,
            'relevance_scores': context.relevance_scores,
            'diversity_score': context.diversity_score,
            'chaos_state': {
                'lyapunov': context.chaos_state.lyapunov_exponent,
                'entropy': context.chaos_state.entropy
            }
        }
    
    def get_system_snapshot(self) -> Dict[str, Any]:
        """Get complete snapshot of the AL-ULS system"""
        return {
            'config': {
                'input_dim': self.config.input_dim,
                'hidden_dims': self.config.hidden_dims,
                'output_dim': self.config.output_dim,
                'entropy_target': self.config.entropy_target
            },
            'training_progress': {
                'epoch': self.current_epoch,
                'step': self.current_step,
                'best_loss': self.best_loss,
                'total_time': self.total_training_time
            },
            'symbolic_memory': self.symbolic_memory.export_state(),
            'optimizer_stats': self.optimizer.get_entropy_statistics(),
            'matrix_structures': self.matrix_optimizer.matrix_info,
            'compression_summary': self.matrix_optimizer.get_compression_summary(),
            'chaos_rag': self.chaos_rag.get_system_state() if self.chaos_rag else None,
            'model_state': {
                'symbolic_reasoning': self.model.get_symbolic_state() if hasattr(self.model, 'get_symbolic_state') else {}
            }
        }
    
    def export_symbolic_knowledge(self) -> Dict[str, Any]:
        """Export all symbolic knowledge discovered by the system"""
        return {
            'constraints': self.symbolic_memory.export_state()['constraints'],
            'matrix_symbolic_forms': self.matrix_optimizer.export_symbolic_forms(),
            'top_constraints': [
                {'id': cid, 'expression': c.expression, 'strength': c.strength, 'confidence': c.confidence}
                for cid, c in self.symbolic_memory.get_top_constraints(n=20)
            ],
            'evolution_metrics': self.symbolic_memory.export_state()['metrics']
        }
    
    def save_checkpoint(self, path: str):
        """Save complete system checkpoint"""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.optimizer.state_dict(),
            'symbolic_memory': self.symbolic_memory.export_state(),
            'config': self.config.__dict__,
            'training_state': {
                'epoch': self.current_epoch,
                'step': self.current_step,
                'best_loss': self.best_loss
            },
            'matrix_info': self.matrix_optimizer.matrix_info
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load system checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.current_epoch = checkpoint['training_state']['epoch']
        self.current_step = checkpoint['training_state']['step']
        self.best_loss = checkpoint['training_state']['best_loss']
