"""
Self-Evolving Symbolic Memory System

This module implements a symbolic memory system that evolves its own constraints
and optimization strategies through adaptive learning.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import time
import asyncio


@dataclass
class SymbolicConstraint:
    """Represents a learned symbolic constraint"""
    expression: str
    variables: List[str]
    strength: float  # 0.0 to 1.0
    confidence: float  # How confident we are in this constraint
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    
    def evolve(self, feedback: Dict[str, Any]) -> float:
        """
        Evolve the constraint based on feedback.
        Returns: improvement score
        """
        old_strength = self.strength
        
        # Update strength based on feedback
        if feedback.get('loss_improved'):
            self.strength = min(1.0, self.strength * 1.1)
            self.confidence = min(1.0, self.confidence * 1.05)
        else:
            self.strength = max(0.1, self.strength * 0.95)
            self.confidence = max(0.1, self.confidence * 0.98)
        
        # Record evolution
        self.evolution_history.append({
            'timestamp': time.time(),
            'old_strength': old_strength,
            'new_strength': self.strength,
            'feedback': feedback
        })
        self.last_updated = time.time()
        
        return abs(self.strength - old_strength)


@dataclass
class SymbolicMemoryState:
    """State of the symbolic memory at a point in time"""
    constraints: Dict[str, SymbolicConstraint]
    activations: Dict[str, float]  # Current activation levels
    entropy: float
    coherence: float
    timestamp: float = field(default_factory=time.time)


class SelfEvolvingSymbolicMemory:
    """
    A symbolic memory system that evolves its own constraints and representations
    through adaptive learning and symbolic reasoning.
    """
    
    def __init__(
        self,
        evolution_rate: float = 0.1,
        pruning_threshold: float = 0.05,
        max_constraints: int = 1000,
        entropy_target: float = 3.5
    ):
        self.evolution_rate = evolution_rate
        self.pruning_threshold = pruning_threshold
        self.max_constraints = max_constraints
        self.entropy_target = entropy_target
        
        # Core memory structures
        self.constraints: Dict[str, SymbolicConstraint] = {}
        self.activation_history: List[Dict[str, float]] = []
        self.evolution_metrics: Dict[str, List[float]] = defaultdict(list)
        
        # Symbolic variables for constraint generation
        self.symbolic_vars = ['s', 'τ', 'μ', 'σ', 'λ', 'ω', 'η']  # state, time, memory, spatial, learning, entropy, efficiency
        
        # Statistics
        self.total_evolutions = 0
        self.successful_evolutions = 0
        self.constraints_created = 0
        self.constraints_pruned = 0
        
    def create_constraint(
        self,
        expression: str,
        variables: List[str],
        initial_strength: float = 0.5
    ) -> SymbolicConstraint:
        """Create a new symbolic constraint"""
        constraint = SymbolicConstraint(
            expression=expression,
            variables=variables,
            strength=initial_strength,
            confidence=0.5  # Start uncertain
        )
        
        constraint_id = f"C_{len(self.constraints)}_{int(time.time()*1000)}"
        self.constraints[constraint_id] = constraint
        self.constraints_created += 1
        
        return constraint
    
    def evolve_constraints(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evolve all constraints based on training feedback.
        
        Args:
            feedback: Dictionary containing:
                - loss_delta: Change in loss
                - entropy: Current entropy
                - gradient_norms: Gradient statistics
                - performance_metrics: Any performance metrics
        
        Returns:
            Dictionary with evolution statistics
        """
        self.total_evolutions += 1
        improvements = []
        
        # Determine if this was a good update
        loss_improved = feedback.get('loss_delta', 0) < 0
        entropy_good = abs(feedback.get('entropy', self.entropy_target) - self.entropy_target) < 0.5
        
        evolution_feedback = {
            'loss_improved': loss_improved,
            'entropy_good': entropy_good,
            'gradient_stable': feedback.get('gradient_norms', {}).get('max', 0) < 10.0
        }
        
        # Evolve each constraint
        for constraint_id, constraint in list(self.constraints.items()):
            improvement = constraint.evolve(evolution_feedback)
            improvements.append(improvement)
            
            # Prune weak constraints
            if constraint.strength < self.pruning_threshold and constraint.confidence < 0.2:
                del self.constraints[constraint_id]
                self.constraints_pruned += 1
        
        if loss_improved:
            self.successful_evolutions += 1
        
        # Create new constraints if we're learning well
        if loss_improved and len(self.constraints) < self.max_constraints:
            self._generate_new_constraints(feedback)
        
        # Record metrics
        avg_improvement = np.mean(improvements) if improvements else 0.0
        self.evolution_metrics['avg_improvement'].append(avg_improvement)
        self.evolution_metrics['num_constraints'].append(len(self.constraints))
        self.evolution_metrics['success_rate'].append(self.successful_evolutions / max(1, self.total_evolutions))
        
        return {
            'num_constraints': len(self.constraints),
            'avg_improvement': avg_improvement,
            'constraints_created': self.constraints_created,
            'constraints_pruned': self.constraints_pruned,
            'success_rate': self.successful_evolutions / max(1, self.total_evolutions)
        }
    
    def _generate_new_constraints(self, feedback: Dict[str, Any]):
        """Generate new constraints based on learning patterns"""
        
        # Pattern 1: Gradient-based constraints
        if 'gradient_norms' in feedback:
            grad_norms = feedback['gradient_norms']
            if grad_norms.get('variance', 0) > 1.0:
                # High variance suggests need for stabilization
                self.create_constraint(
                    expression=f"σ(∇L) < {grad_norms.get('variance', 1.0):.2f}",
                    variables=['σ', 'λ'],
                    initial_strength=0.6
                )
        
        # Pattern 2: Entropy-based constraints
        if 'entropy' in feedback:
            ent = feedback['entropy']
            if abs(ent - self.entropy_target) > 1.0:
                # Entropy too far from target
                self.create_constraint(
                    expression=f"H(s) → {self.entropy_target:.2f}",
                    variables=['s', 'ω'],
                    initial_strength=0.7
                )
        
        # Pattern 3: Temporal coherence constraints
        if len(self.activation_history) > 5:
            # Check for temporal instability
            recent_activations = self.activation_history[-5:]
            variances = []
            for key in recent_activations[0].keys():
                vals = [act.get(key, 0) for act in recent_activations]
                variances.append(np.var(vals))
            
            if np.mean(variances) > 0.5:
                self.create_constraint(
                    expression="smooth(s(t), τ)",
                    variables=['s', 'τ'],
                    initial_strength=0.5
                )
    
    def get_constraint_state(self) -> SymbolicMemoryState:
        """Get current state of symbolic memory"""
        
        # Compute current activations
        activations = {
            constraint_id: constraint.strength * constraint.confidence
            for constraint_id, constraint in self.constraints.items()
        }
        
        # Compute entropy of constraint distribution
        activation_values = np.array(list(activations.values())) if activations else np.array([0.0])
        activation_values = activation_values / (np.sum(activation_values) + 1e-10)
        entropy = -np.sum(activation_values * np.log(activation_values + 1e-10))
        
        # Compute coherence (how aligned are the constraints)
        coherence = self._compute_coherence()
        
        return SymbolicMemoryState(
            constraints=self.constraints.copy(),
            activations=activations,
            entropy=float(entropy),
            coherence=coherence
        )
    
    def _compute_coherence(self) -> float:
        """Compute how coherent/aligned the constraints are"""
        if len(self.constraints) < 2:
            return 1.0
        
        # Constraints that share variables are more coherent
        var_constraints = defaultdict(list)
        for cid, constraint in self.constraints.items():
            for var in constraint.variables:
                var_constraints[var].append(cid)
        
        # Compute overlap score
        total_pairs = len(self.constraints) * (len(self.constraints) - 1) / 2
        overlapping_pairs = 0
        
        for var, constraint_ids in var_constraints.items():
            overlapping_pairs += len(constraint_ids) * (len(constraint_ids) - 1) / 2
        
        coherence = overlapping_pairs / (total_pairs + 1e-10)
        return min(1.0, coherence)
    
    def apply_constraints(self, parameters: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply learned symbolic constraints to parameters.
        This projects parameters onto the constraint manifold.
        """
        constrained_params = parameters.copy()
        
        for constraint_id, constraint in self.constraints.items():
            if constraint.strength < 0.2:
                continue  # Skip weak constraints
            
            # Simple constraint application: regularize toward constraint
            for param_name, param_value in constrained_params.items():
                if any(var in param_name.lower() for var in constraint.variables):
                    # Apply soft constraint via regularization
                    scale = 1.0 - (constraint.strength * self.evolution_rate)
                    constrained_params[param_name] = param_value * scale
        
        return constrained_params
    
    def export_state(self) -> Dict[str, Any]:
        """Export full state for persistence"""
        return {
            'constraints': {
                cid: {
                    'expression': c.expression,
                    'variables': c.variables,
                    'strength': c.strength,
                    'confidence': c.confidence,
                    'created_at': c.created_at,
                    'last_updated': c.last_updated,
                    'evolution_count': len(c.evolution_history)
                }
                for cid, c in self.constraints.items()
            },
            'statistics': {
                'total_evolutions': self.total_evolutions,
                'successful_evolutions': self.successful_evolutions,
                'constraints_created': self.constraints_created,
                'constraints_pruned': self.constraints_pruned,
                'current_constraints': len(self.constraints)
            },
            'metrics': {
                key: values[-100:] if len(values) > 100 else values
                for key, values in self.evolution_metrics.items()
            }
        }
    
    def get_top_constraints(self, n: int = 10) -> List[Tuple[str, SymbolicConstraint]]:
        """Get the top N most important constraints"""
        scored_constraints = [
            (cid, c, c.strength * c.confidence)
            for cid, c in self.constraints.items()
        ]
        scored_constraints.sort(key=lambda x: x[2], reverse=True)
        return [(cid, c) for cid, c, _ in scored_constraints[:n]]
