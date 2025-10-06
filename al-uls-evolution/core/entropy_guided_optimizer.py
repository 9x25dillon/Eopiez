"""
Entropy-Guided Learning Optimizer

Uses information theory and entropy measures to guide the learning process,
adapting learning rates, batch selection, and optimization strategies based
on entropy signals.
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from collections import deque
from dataclasses import dataclass, field
import math


@dataclass
class EntropyMetrics:
    """Metrics related to entropy in the learning process"""
    parameter_entropy: float
    gradient_entropy: float
    activation_entropy: float
    loss_entropy: float  # Entropy across batch
    prediction_entropy: float  # Entropy of model predictions
    timestamp: float = field(default_factory=lambda: torch.cuda.Event().elapsed_time(torch.cuda.Event()) if torch.cuda.is_available() else 0.0)
    
    def total_entropy(self) -> float:
        """Weighted combination of all entropy measures"""
        return (
            0.3 * self.parameter_entropy +
            0.3 * self.gradient_entropy +
            0.2 * self.activation_entropy +
            0.1 * self.loss_entropy +
            0.1 * self.prediction_entropy
        )
    
    def entropy_imbalance(self) -> float:
        """Measure how imbalanced the entropy distribution is"""
        entropies = [
            self.parameter_entropy,
            self.gradient_entropy,
            self.activation_entropy,
            self.loss_entropy,
            self.prediction_entropy
        ]
        mean_ent = np.mean(entropies)
        return float(np.std(entropies) / (mean_ent + 1e-10))


class EntropyGuidedOptimizer:
    """
    Optimizer that uses entropy signals to guide learning.
    
    Key concepts:
    - High gradient entropy → reduce learning rate (unstable)
    - Low gradient entropy → increase learning rate (plateaued)
    - High prediction entropy → model uncertain, explore more
    - Low prediction entropy → model confident, exploit more
    """
    
    def __init__(
        self,
        parameters,
        base_lr: float = 1e-3,
        entropy_target: float = 3.5,
        entropy_tolerance: float = 0.5,
        adaptation_rate: float = 0.1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        optimizer_type: str = 'adam'
    ):
        self.parameters = list(parameters)
        self.base_lr = base_lr
        self.current_lr = base_lr
        self.entropy_target = entropy_target
        self.entropy_tolerance = entropy_tolerance
        self.adaptation_rate = adaptation_rate
        
        # Create base optimizer
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(self.parameters, lr=base_lr, weight_decay=weight_decay)
        elif optimizer_type == 'sgd':
            self.optimizer = optim.SGD(self.parameters, lr=base_lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(self.parameters, lr=base_lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        # Entropy tracking
        self.entropy_history: deque = deque(maxlen=100)
        self.lr_history: deque = deque(maxlen=100)
        
        # Statistics
        self.total_steps = 0
        self.lr_increases = 0
        self.lr_decreases = 0
        
        # Activation hooks
        self.activation_entropies = {}
        self.hooks = []
    
    def compute_entropy(self, tensor: torch.Tensor) -> float:
        """
        Compute Shannon entropy of a tensor.
        Discretizes continuous values into bins.
        """
        if tensor.numel() == 0:
            return 0.0
        
        # Flatten tensor
        flat = tensor.detach().cpu().numpy().flatten()
        
        # Create histogram (discretize)
        hist, _ = np.histogram(flat, bins=50, density=True)
        
        # Remove zero bins
        hist = hist[hist > 0]
        
        # Normalize
        hist = hist / hist.sum()
        
        # Compute entropy
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        
        return float(entropy)
    
    def compute_gradient_entropy(self) -> float:
        """Compute entropy of gradients across all parameters"""
        all_grads = []
        
        for param in self.parameters:
            if param.grad is not None:
                all_grads.append(param.grad.detach().cpu().numpy().flatten())
        
        if not all_grads:
            return 0.0
        
        all_grads = np.concatenate(all_grads)
        
        # Compute histogram-based entropy
        hist, _ = np.histogram(all_grads, bins=50, density=True)
        hist = hist[hist > 0]
        hist = hist / hist.sum()
        
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        return float(entropy)
    
    def compute_parameter_entropy(self) -> float:
        """Compute entropy of parameter values"""
        all_params = []
        
        for param in self.parameters:
            all_params.append(param.detach().cpu().numpy().flatten())
        
        if not all_params:
            return 0.0
        
        all_params = np.concatenate(all_params)
        
        hist, _ = np.histogram(all_params, bins=50, density=True)
        hist = hist[hist > 0]
        hist = hist / hist.sum()
        
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        return float(entropy)
    
    def compute_prediction_entropy(self, predictions: torch.Tensor) -> float:
        """
        Compute entropy of model predictions.
        For classification: use softmax outputs.
        For regression: use distribution of predictions.
        """
        if predictions.dim() > 1 and predictions.size(-1) > 1:
            # Multi-class: compute entropy of softmax
            probs = torch.softmax(predictions, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean()
            return float(entropy.item())
        else:
            # Regression: compute entropy of prediction distribution
            return self.compute_entropy(predictions)
    
    def compute_loss_entropy(self, losses: torch.Tensor) -> float:
        """Compute entropy of loss distribution across batch"""
        return self.compute_entropy(losses)
    
    def register_activation_hooks(self, model: torch.nn.Module):
        """Register hooks to capture activation entropies"""
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activation_entropies[name] = self.compute_entropy(output)
            return hook
        
        # Clear existing hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
        # Register new hooks
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.LSTM)):
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
    
    def compute_metrics(
        self,
        predictions: Optional[torch.Tensor] = None,
        losses: Optional[torch.Tensor] = None
    ) -> EntropyMetrics:
        """Compute all entropy metrics"""
        
        param_entropy = self.compute_parameter_entropy()
        grad_entropy = self.compute_gradient_entropy()
        
        # Activation entropy (average across layers)
        if self.activation_entropies:
            activation_entropy = np.mean(list(self.activation_entropies.values()))
        else:
            activation_entropy = 0.0
        
        # Prediction and loss entropy
        if predictions is not None:
            pred_entropy = self.compute_prediction_entropy(predictions)
        else:
            pred_entropy = 0.0
        
        if losses is not None:
            loss_entropy = self.compute_loss_entropy(losses)
        else:
            loss_entropy = 0.0
        
        return EntropyMetrics(
            parameter_entropy=param_entropy,
            gradient_entropy=grad_entropy,
            activation_entropy=activation_entropy,
            loss_entropy=loss_entropy,
            prediction_entropy=pred_entropy
        )
    
    def adapt_learning_rate(self, metrics: EntropyMetrics) -> float:
        """
        Adapt learning rate based on entropy metrics.
        
        Strategy:
        - If total entropy too high → reduce LR (system too chaotic)
        - If total entropy too low → increase LR (system too ordered/stuck)
        - If gradient entropy high → reduce LR (unstable gradients)
        - If prediction entropy high → increase exploration
        """
        total_entropy = metrics.total_entropy()
        
        # Compute entropy deviation from target
        entropy_deviation = total_entropy - self.entropy_target
        
        # Adaptation signal
        if abs(entropy_deviation) < self.entropy_tolerance:
            # In good range, no change
            lr_multiplier = 1.0
        elif entropy_deviation > 0:
            # Too much entropy, reduce LR
            excess = (entropy_deviation / self.entropy_target)
            lr_multiplier = 1.0 / (1.0 + self.adaptation_rate * excess)
            self.lr_decreases += 1
        else:
            # Too little entropy, increase LR
            deficit = abs(entropy_deviation / self.entropy_target)
            lr_multiplier = 1.0 + self.adaptation_rate * deficit
            self.lr_increases += 1
        
        # Additional adjustment based on gradient entropy
        if metrics.gradient_entropy > 4.0:
            # Very high gradient entropy, be more conservative
            lr_multiplier *= 0.9
        elif metrics.gradient_entropy < 2.0:
            # Very low gradient entropy, can be more aggressive
            lr_multiplier *= 1.1
        
        # Clip multiplier
        lr_multiplier = np.clip(lr_multiplier, 0.5, 2.0)
        
        # Update learning rate
        new_lr = self.current_lr * lr_multiplier
        new_lr = np.clip(new_lr, self.base_lr * 0.01, self.base_lr * 10.0)
        
        self.current_lr = new_lr
        
        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        
        return new_lr
    
    def step(
        self,
        predictions: Optional[torch.Tensor] = None,
        losses: Optional[torch.Tensor] = None,
        compute_metrics_flag: bool = True
    ) -> Dict[str, Any]:
        """
        Perform optimization step with entropy guidance.
        
        Returns:
            Dictionary with step statistics
        """
        # Compute metrics
        if compute_metrics_flag:
            metrics = self.compute_metrics(predictions, losses)
            self.entropy_history.append(metrics)
            
            # Adapt learning rate
            new_lr = self.adapt_learning_rate(metrics)
            self.lr_history.append(new_lr)
        else:
            metrics = None
            new_lr = self.current_lr
        
        # Perform optimization step
        self.optimizer.step()
        self.total_steps += 1
        
        # Return statistics
        stats = {
            'learning_rate': new_lr,
            'total_steps': self.total_steps,
            'lr_increases': self.lr_increases,
            'lr_decreases': self.lr_decreases
        }
        
        if metrics:
            stats.update({
                'total_entropy': metrics.total_entropy(),
                'gradient_entropy': metrics.gradient_entropy,
                'parameter_entropy': metrics.parameter_entropy,
                'activation_entropy': metrics.activation_entropy,
                'entropy_imbalance': metrics.entropy_imbalance()
            })
        
        return stats
    
    def zero_grad(self):
        """Zero gradients"""
        self.optimizer.zero_grad()
    
    def get_entropy_statistics(self) -> Dict[str, Any]:
        """Get statistics about entropy over training"""
        if not self.entropy_history:
            return {}
        
        total_entropies = [m.total_entropy() for m in self.entropy_history]
        gradient_entropies = [m.gradient_entropy for m in self.entropy_history]
        
        return {
            'mean_total_entropy': np.mean(total_entropies),
            'std_total_entropy': np.std(total_entropies),
            'mean_gradient_entropy': np.mean(gradient_entropies),
            'std_gradient_entropy': np.std(gradient_entropies),
            'current_lr': self.current_lr,
            'lr_adaptation_ratio': self.lr_increases / max(1, self.lr_decreases),
            'avg_lr': np.mean(list(self.lr_history)) if self.lr_history else self.base_lr
        }
    
    def should_explore(self, threshold: float = 0.7) -> bool:
        """
        Determine if the optimizer should explore (vs exploit).
        Based on recent prediction entropy.
        """
        if len(self.entropy_history) < 5:
            return True  # Explore early in training
        
        recent = list(self.entropy_history)[-5:]
        avg_pred_entropy = np.mean([m.prediction_entropy for m in recent])
        
        # High prediction entropy → explore
        # Low prediction entropy → exploit
        return avg_pred_entropy > threshold
    
    def get_chaos_signal(self) -> float:
        """
        Get a "chaos signal" indicating system stability.
        0.0 = very stable, 1.0 = very chaotic
        """
        if not self.entropy_history:
            return 0.5
        
        recent = list(self.entropy_history)[-10:]
        
        # Measure volatility of entropy
        total_entropies = [m.total_entropy() for m in recent]
        entropy_volatility = np.std(total_entropies) / (np.mean(total_entropies) + 1e-10)
        
        # Measure imbalance
        avg_imbalance = np.mean([m.entropy_imbalance() for m in recent])
        
        # Combine signals
        chaos = 0.6 * entropy_volatility + 0.4 * avg_imbalance
        
        return float(np.clip(chaos, 0.0, 1.0))
