"""
Hybrid Neural-Symbolic Reasoning Engine

Combines neural network learning with symbolic constraint reasoning
for emergent intelligent behavior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import asyncio


@dataclass
class ReasoningTrace:
    """Trace of a reasoning step combining neural and symbolic"""
    neural_output: torch.Tensor
    symbolic_constraints: List[str]
    constraint_violations: List[Tuple[str, float]]
    corrected_output: torch.Tensor
    confidence: float
    reasoning_path: List[str]


class SymbolicReasoningLayer(nn.Module):
    """
    Neural layer that incorporates symbolic reasoning constraints.
    """
    
    def __init__(self, in_features: int, out_features: int, num_symbolic_paths: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_symbolic_paths = num_symbolic_paths
        
        # Multiple reasoning paths (symbolic decomposition)
        self.reasoning_paths = nn.ModuleList([
            nn.Linear(in_features, out_features)
            for _ in range(num_symbolic_paths)
        ])
        
        # Path selector (learns which symbolic path to use)
        self.path_selector = nn.Linear(in_features, num_symbolic_paths)
        
        # Constraint validator (checks symbolic constraints)
        self.constraint_validator = nn.Linear(out_features, num_symbolic_paths)
        
        # Initialize with symbolic structure
        self._initialize_symbolic_structure()
    
    def _initialize_symbolic_structure(self):
        """Initialize weights with symbolic structure"""
        with torch.no_grad():
            for i, path in enumerate(self.reasoning_paths):
                # Each path starts with a different symbolic bias
                # Path 0: identity-like (s)
                # Path 1: temporal (τ)  
                # Path 2: memory (μ)
                if i == 0:
                    nn.init.eye_(path.weight[:min(self.in_features, self.out_features), :min(self.in_features, self.out_features)])
                elif i == 1:
                    nn.init.orthogonal_(path.weight)
                else:
                    nn.init.sparse_(path.weight, sparsity=0.5)
    
    def forward(self, x: torch.Tensor, symbolic_constraints: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with symbolic reasoning.
        
        Returns:
            output: Constrained output
            reasoning_info: Dictionary with reasoning trace
        """
        batch_size = x.size(0)
        
        # Compute all symbolic paths
        path_outputs = torch.stack([path(x) for path in self.reasoning_paths], dim=1)  # [B, num_paths, out_features]
        
        # Select paths based on input
        path_weights = F.softmax(self.path_selector(x), dim=-1)  # [B, num_paths]
        path_weights = path_weights.unsqueeze(-1)  # [B, num_paths, 1]
        
        # Weighted combination of paths
        output = (path_outputs * path_weights).sum(dim=1)  # [B, out_features]
        
        # Apply symbolic constraints if provided
        if symbolic_constraints:
            output = self._apply_constraints(output, symbolic_constraints)
        
        # Validate against constraints
        constraint_scores = torch.sigmoid(self.constraint_validator(output))  # [B, num_paths]
        
        reasoning_info = {
            'path_weights': path_weights.squeeze(-1).detach().cpu().numpy(),
            'constraint_scores': constraint_scores.detach().cpu().numpy(),
            'path_outputs': path_outputs.detach().cpu().numpy()
        }
        
        return output, reasoning_info
    
    def _apply_constraints(self, output: torch.Tensor, constraints: Dict[str, Any]) -> torch.Tensor:
        """Apply symbolic constraints to output"""
        constrained_output = output.clone()
        
        # Constraint 1: Bounded output
        if constraints.get('bounded', False):
            lower = constraints.get('lower_bound', -10.0)
            upper = constraints.get('upper_bound', 10.0)
            constrained_output = torch.clamp(constrained_output, lower, upper)
        
        # Constraint 2: Sparsity
        if constraints.get('sparse', False):
            threshold = constraints.get('sparsity_threshold', 0.1)
            mask = (constrained_output.abs() > threshold).float()
            constrained_output = constrained_output * mask
        
        # Constraint 3: Orthogonality (for multi-dimensional outputs)
        if constraints.get('orthogonal', False) and constrained_output.dim() > 1:
            # Gram-Schmidt-like normalization
            constrained_output = F.normalize(constrained_output, dim=-1)
        
        return constrained_output


class HybridReasoningBlock(nn.Module):
    """
    Complete hybrid neural-symbolic reasoning block.
    Combines CompoundNode from ML2 with symbolic reasoning.
    """
    
    def __init__(self, in_features: int, out_features: int, use_symbolic: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_symbolic = use_symbolic
        
        if use_symbolic:
            # Symbolic reasoning layer
            self.symbolic_layer = SymbolicReasoningLayer(in_features, out_features, num_symbolic_paths=4)
        else:
            # Standard neural layer
            self.neural_layer = nn.Linear(in_features, out_features)
        
        # Residual connection
        if in_features == out_features:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Linear(in_features, out_features, bias=False)
        
        # Layer normalization for stability
        self.norm = nn.LayerNorm(out_features)
        
    def forward(self, x: torch.Tensor, symbolic_constraints: Optional[Dict[str, Any]] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward with optional symbolic reasoning"""
        
        if self.use_symbolic and symbolic_constraints:
            symbolic_out, reasoning_info = self.symbolic_layer(x, symbolic_constraints)
            output = symbolic_out + self.residual(x)
        else:
            if self.use_symbolic:
                symbolic_out, reasoning_info = self.symbolic_layer(x, None)
                output = symbolic_out + self.residual(x)
            else:
                output = self.neural_layer(x) + self.residual(x)
                reasoning_info = {}
        
        output = self.norm(output)
        
        return output, reasoning_info


class NeuralSymbolicHybrid(nn.Module):
    """
    Complete neural-symbolic hybrid network that combines:
    - Neural network learning (gradient-based)
    - Symbolic constraint reasoning
    - Adaptive path selection
    - Emergent behavior through interaction
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        use_symbolic: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.use_symbolic = use_symbolic
        
        # Build hybrid layers
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.layers.append(HybridReasoningBlock(prev_dim, hidden_dim, use_symbolic=use_symbolic))
            self.layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (always symbolic if enabled)
        if use_symbolic:
            self.output_layer = SymbolicReasoningLayer(prev_dim, output_dim, num_symbolic_paths=3)
        else:
            self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # Constraint tracking
        self.constraint_violations_history = []
        self.reasoning_traces = []
        
    def forward(
        self,
        x: torch.Tensor,
        symbolic_constraints: Optional[Dict[str, Any]] = None,
        return_trace: bool = False
    ) -> Tuple[torch.Tensor, Optional[ReasoningTrace]]:
        """
        Forward pass with optional symbolic reasoning trace.
        
        Args:
            x: Input tensor
            symbolic_constraints: Dictionary of symbolic constraints to apply
            return_trace: Whether to return detailed reasoning trace
        
        Returns:
            output: Network output
            trace: Optional reasoning trace
        """
        reasoning_infos = []
        current = x
        
        # Pass through hybrid layers
        for i, layer in enumerate(self.layers):
            if isinstance(layer, HybridReasoningBlock):
                current, info = layer(current, symbolic_constraints)
                reasoning_infos.append(info)
            else:
                # Dropout layer
                current = layer(current)
        
        # Output layer
        if self.use_symbolic:
            output, output_info = self.output_layer(current, symbolic_constraints)
            reasoning_infos.append(output_info)
        else:
            output = self.output_layer(current)
            output_info = {}
        
        # Create reasoning trace if requested
        trace = None
        if return_trace and self.use_symbolic:
            trace = self._create_reasoning_trace(x, output, reasoning_infos, symbolic_constraints)
            self.reasoning_traces.append(trace)
        
        return output, trace
    
    def _create_reasoning_trace(
        self,
        input_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
        reasoning_infos: List[Dict[str, Any]],
        constraints: Optional[Dict[str, Any]]
    ) -> ReasoningTrace:
        """Create a detailed reasoning trace"""
        
        # Extract constraint violations
        violations = []
        if constraints:
            for key, value in constraints.items():
                # Check each constraint
                if key == 'bounded':
                    lower = constraints.get('lower_bound', -10.0)
                    upper = constraints.get('upper_bound', 10.0)
                    out_of_bounds = ((output_tensor < lower) | (output_tensor > upper)).float().mean().item()
                    if out_of_bounds > 0:
                        violations.append((f'bounded[{lower},{upper}]', out_of_bounds))
        
        # Determine confidence based on constraint scores
        all_scores = []
        for info in reasoning_infos:
            if 'constraint_scores' in info:
                all_scores.extend(info['constraint_scores'].flatten().tolist())
        
        confidence = np.mean(all_scores) if all_scores else 0.5
        
        # Build reasoning path
        reasoning_path = []
        for i, info in enumerate(reasoning_infos):
            if 'path_weights' in info:
                weights = info['path_weights'][0]  # First batch item
                dominant_path = np.argmax(weights)
                reasoning_path.append(f"Layer{i}→Path{dominant_path}({weights[dominant_path]:.2f})")
        
        return ReasoningTrace(
            neural_output=output_tensor.detach(),
            symbolic_constraints=list(constraints.keys()) if constraints else [],
            constraint_violations=violations,
            corrected_output=output_tensor.detach(),  # After constraint application
            confidence=float(confidence),
            reasoning_path=reasoning_path
        )
    
    def get_symbolic_state(self) -> Dict[str, Any]:
        """Get current symbolic reasoning state"""
        
        # Extract path preferences from each layer
        layer_states = []
        for layer in self.layers:
            if isinstance(layer, HybridReasoningBlock) and hasattr(layer, 'symbolic_layer'):
                # Get current path selector biases
                with torch.no_grad():
                    selector_weights = layer.symbolic_layer.path_selector.weight.detach().cpu().numpy()
                    layer_states.append({
                        'path_selector_norm': float(np.linalg.norm(selector_weights)),
                        'path_selector_rank': int(np.linalg.matrix_rank(selector_weights))
                    })
        
        return {
            'num_layers': len(layer_states),
            'layer_states': layer_states,
            'recent_violations': self.constraint_violations_history[-10:] if self.constraint_violations_history else [],
            'num_traces': len(self.reasoning_traces)
        }
    
    def apply_symbolic_evolution(self, evolved_constraints: Dict[str, float]):
        """
        Apply evolved symbolic constraints to the network structure.
        This modifies the network based on learned symbolic patterns.
        """
        with torch.no_grad():
            for layer in self.layers:
                if isinstance(layer, HybridReasoningBlock) and hasattr(layer, 'symbolic_layer'):
                    # Adjust path selector based on evolved constraints
                    for constraint_name, strength in evolved_constraints.items():
                        if 'temporal' in constraint_name.lower():
                            # Boost temporal path (path 1)
                            layer.symbolic_layer.path_selector.weight[:, 1] *= (1.0 + strength * 0.1)
                        elif 'memory' in constraint_name.lower():
                            # Boost memory path (path 2)
                            layer.symbolic_layer.path_selector.weight[:, 2] *= (1.0 + strength * 0.1)
                        elif 'sparse' in constraint_name.lower():
                            # Encourage sparsity in all paths
                            for path in layer.symbolic_layer.reasoning_paths:
                                path.weight *= (1.0 - strength * 0.05)


class SymbolicGradientModifier:
    """
    Modifies gradients based on symbolic constraints.
    Works with GradNormalizer from ML2.
    """
    
    def __init__(self, symbolic_memory):
        self.symbolic_memory = symbolic_memory
        self.modification_history = []
    
    def modify_gradients(
        self,
        model: nn.Module,
        symbolic_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Modify gradients based on current symbolic constraints.
        
        Returns statistics about modifications.
        """
        modifications = {
            'scaled_layers': 0,
            'zeroed_gradients': 0,
            'total_modifications': 0
        }
        
        # Get active constraints from symbolic memory
        constraints = symbolic_state.get('constraints', {})
        
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            
            # Apply constraint-based modifications
            for constraint_id, constraint_info in constraints.items():
                if not isinstance(constraint_info, dict):
                    continue
                    
                strength = constraint_info.get('strength', 0.0)
                variables = constraint_info.get('variables', [])
                
                # Check if this parameter relates to constraint variables
                param_lower = name.lower()
                if any(var.lower() in param_lower for var in variables):
                    # Scale gradient based on constraint strength
                    scale = 1.0 - (strength * 0.2)  # Max 20% reduction
                    param.grad *= scale
                    modifications['scaled_layers'] += 1
                    modifications['total_modifications'] += 1
        
        self.modification_history.append(modifications)
        return modifications
