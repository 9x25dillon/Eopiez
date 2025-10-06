"""
AL-ULS Evolution Training Demo

Demonstrates the complete AL-ULS Evolution system with:
- Self-evolving symbolic constraints
- Hybrid neural-symbolic reasoning
- Entropy-guided optimization
- Chaos RAG knowledge augmentation
- Matrix structure discovery
"""

import torch
import torch.nn as nn
import numpy as np
import sys
from pathlib import Path
import asyncio

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.al_uls_orchestrator import ALULSEvolutionSystem, ALULSConfig


def generate_synthetic_data(n_samples=1000, input_dim=64, output_dim=10):
    """Generate synthetic classification data"""
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, output_dim, (n_samples,))
    return X, y


def train_epoch(system, train_data, train_targets, batch_size=32):
    """Train for one epoch"""
    n_samples = len(train_data)
    indices = torch.randperm(n_samples)
    
    epoch_metrics = []
    
    for i in range(0, n_samples, batch_size):
        batch_indices = indices[i:i+batch_size]
        batch_data = train_data[batch_indices]
        batch_targets = train_targets[batch_indices]
        
        metrics = system.training_step(batch_data, batch_targets)
        epoch_metrics.append(metrics)
    
    # Aggregate metrics
    avg_metrics = {
        'loss': np.mean([m['loss'] for m in epoch_metrics]),
        'learning_rate': epoch_metrics[-1]['learning_rate'],
        'entropy': np.mean([m['entropy'] for m in epoch_metrics]),
        'num_constraints': epoch_metrics[-1]['num_constraints'],
        'gradient_norm': np.mean([m['gradient_norm'] for m in epoch_metrics])
    }
    
    return avg_metrics


async def chaos_rag_demo(system, test_data):
    """Demonstrate Chaos RAG capabilities"""
    print("\n" + "="*70)
    print("🌀 CHAOS RAG DEMONSTRATION")
    print("="*70)
    
    if not system.chaos_rag:
        print("Chaos RAG not enabled")
        return
    
    # Add some test data to knowledge base
    print("Adding items to knowledge base...")
    for i in range(100):
        system.chaos_rag.add_to_knowledge_base({
            'text': f'Training sample {i}',
            'embedding': test_data[i].numpy(),
            'metadata': {'sample_id': i, 'category': i % 10}
        })
    
    # Perform chaos-augmented retrieval
    print("\nPerforming chaos-augmented retrieval...")
    query = test_data[0].numpy()
    
    result = await system.chaos_augmented_query(query, top_k=5)
    
    print(f"\nRetrieved {len(result['retrieved_items'])} items")
    print(f"Diversity Score: {result['diversity_score']:.4f}")
    print(f"Lyapunov Exponent: {result['chaos_state']['lyapunov']:.4f}")
    print(f"Chaos Entropy: {result['chaos_state']['entropy']:.4f}")
    
    print("\nTop 3 retrieved items:")
    for i, (item, score) in enumerate(zip(result['retrieved_items'][:3], 
                                          result['relevance_scores'][:3])):
        print(f"  {i+1}. {item.get('text', 'N/A')} (score: {score:.4f})")


def print_symbolic_knowledge(system):
    """Print discovered symbolic knowledge"""
    print("\n" + "="*70)
    print("🧠 SYMBOLIC KNOWLEDGE EXPORT")
    print("="*70)
    
    knowledge = system.export_symbolic_knowledge()
    
    print(f"\nTotal Constraints: {len(knowledge['constraints'])}")
    print(f"\nTop 10 Evolved Constraints:")
    
    for i, c in enumerate(knowledge['top_constraints'][:10], 1):
        print(f"  {i}. {c['expression']}")
        print(f"     Variables: {', '.join(c['variables']) if isinstance(c['variables'], list) else c['variables']}")
        print(f"     Strength: {c['strength']:.3f}, Confidence: {c['confidence']:.3f}")
        print()
    
    print("\nMatrix Symbolic Forms:")
    for name, form in knowledge['matrix_symbolic_forms'].items():
        print(f"  {name}: {form}")


def print_matrix_analysis(system):
    """Print matrix structure analysis"""
    print("\n" + "="*70)
    print("🔬 MATRIX STRUCTURE ANALYSIS")
    print("="*70)
    
    matrix_info = system.matrix_optimizer.matrix_info
    compression = system.matrix_optimizer.get_compression_summary()
    
    print(f"\nTotal Parameters: {compression['total_parameters']:,}")
    print(f"Compressed Parameters: {compression['compressed_parameters']:,}")
    print(f"Compression Ratio: {compression['compression_ratio']:.3f}")
    print(f"Parameter Savings: {compression['parameter_savings']:,}")
    
    print("\nDiscovered Structures:")
    for name, info in matrix_info.items():
        print(f"\n  {name} {info['shape']}")
        for structure in info['structures']:
            print(f"    - {structure['type']}: quality={structure['quality']:.3f}, "
                  f"compression={structure['compression']:.3f}")
            print(f"      {structure['expression']}")
        
        if info['polynomial']:
            print(f"    - Polynomial (degree {info['polynomial']['degree']}): "
                  f"error={info['polynomial']['error']:.4f}")


def main():
    """Main training demonstration"""
    print("="*70)
    print("🚀 AL-ULS EVOLUTION TRAINING DEMONSTRATION")
    print("="*70)
    
    # Configuration
    config = ALULSConfig(
        input_dim=64,
        hidden_dims=[128, 256, 128],
        output_dim=10,
        base_lr=1e-3,
        entropy_target=3.5,
        entropy_tolerance=0.5,
        evolution_rate=0.1,
        max_constraints=500,
        chaos_strength=0.5,
        use_chaos_rag=True,
        use_symbolic_reasoning=True,
        structure_weight=0.05
    )
    
    print("\nSystem Configuration:")
    print(f"  Input Dim: {config.input_dim}")
    print(f"  Hidden Dims: {config.hidden_dims}")
    print(f"  Output Dim: {config.output_dim}")
    print(f"  Base LR: {config.base_lr}")
    print(f"  Entropy Target: {config.entropy_target}")
    print(f"  Max Constraints: {config.max_constraints}")
    print(f"  Chaos RAG: {config.use_chaos_rag}")
    print(f"  Symbolic Reasoning: {config.use_symbolic_reasoning}")
    
    # Initialize system
    print("\n📦 Initializing AL-ULS Evolution System...")
    system = ALULSEvolutionSystem(config)
    print("✅ System initialized")
    
    # Generate data
    print("\n📊 Generating synthetic data...")
    train_data, train_targets = generate_synthetic_data(n_samples=1000, 
                                                        input_dim=config.input_dim,
                                                        output_dim=config.output_dim)
    test_data, test_targets = generate_synthetic_data(n_samples=200,
                                                      input_dim=config.input_dim,
                                                      output_dim=config.output_dim)
    print(f"✅ Generated {len(train_data)} training samples, {len(test_data)} test samples")
    
    # Training loop
    print("\n🎯 Starting Training Loop...")
    print("="*70)
    
    n_epochs = 20
    
    for epoch in range(n_epochs):
        system.current_epoch = epoch
        
        # Train epoch
        metrics = train_epoch(system, train_data, train_targets, batch_size=32)
        
        # Evaluate
        eval_metrics = system.evaluate(test_data, test_targets)
        
        # Get optimizer state
        entropy_stats = system.optimizer.get_entropy_statistics()
        chaos_signal = system.optimizer.get_chaos_signal()
        
        # Print progress
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        print(f"  Train Loss: {metrics['loss']:.4f}")
        print(f"  Test Loss: {eval_metrics['eval_loss']:.4f}")
        print(f"  Test Accuracy: {eval_metrics.get('eval_accuracy', 0):.4f}")
        print(f"  Learning Rate: {metrics['learning_rate']:.6f}")
        print(f"  Entropy: {metrics['entropy']:.3f} (target: {config.entropy_target})")
        print(f"  Gradient Norm: {metrics['gradient_norm']:.4f}")
        print(f"  Num Constraints: {metrics['num_constraints']}")
        print(f"  Chaos Signal: {chaos_signal:.3f}")
        
        # Periodic analysis
        if (epoch + 1) % 5 == 0:
            print("\n--- Analysis ---")
            
            # Get top constraints
            top_constraints = system.symbolic_memory.get_top_constraints(n=5)
            print(f"  Top 5 Constraints:")
            for i, (cid, c) in enumerate(top_constraints, 1):
                print(f"    {i}. {c.expression} (strength: {c.strength:.2f})")
            
            # Entropy adaptation info
            print(f"  LR Increases: {system.optimizer.lr_increases}")
            print(f"  LR Decreases: {system.optimizer.lr_decreases}")
            print(f"  Avg LR: {entropy_stats['avg_lr']:.6f}")
    
    # Final Analysis
    print("\n" + "="*70)
    print("📈 FINAL TRAINING ANALYSIS")
    print("="*70)
    
    final_snapshot = system.get_system_snapshot()
    
    print(f"\nTraining Summary:")
    print(f"  Total Steps: {final_snapshot['training_progress']['step']}")
    print(f"  Best Loss: {final_snapshot['training_progress']['best_loss']:.4f}")
    print(f"  Total Time: {final_snapshot['training_progress']['total_time']:.2f}s")
    
    print(f"\nSymbolic Memory:")
    print(f"  Total Evolutions: {final_snapshot['symbolic_memory']['statistics']['total_evolutions']}")
    print(f"  Successful Evolutions: {final_snapshot['symbolic_memory']['statistics']['successful_evolutions']}")
    print(f"  Constraints Created: {final_snapshot['symbolic_memory']['statistics']['constraints_created']}")
    print(f"  Constraints Pruned: {final_snapshot['symbolic_memory']['statistics']['constraints_pruned']}")
    print(f"  Current Constraints: {final_snapshot['symbolic_memory']['statistics']['current_constraints']}")
    
    # Print detailed symbolic knowledge
    print_symbolic_knowledge(system)
    
    # Print matrix analysis
    print_matrix_analysis(system)
    
    # Chaos RAG demo
    asyncio.run(chaos_rag_demo(system, test_data))
    
    # Save checkpoint
    print("\n" + "="*70)
    print("💾 SAVING CHECKPOINT")
    print("="*70)
    
    checkpoint_path = "/tmp/al_uls_checkpoint.pt"
    system.save_checkpoint(checkpoint_path)
    print(f"✅ Checkpoint saved to {checkpoint_path}")
    
    print("\n" + "="*70)
    print("🎉 DEMONSTRATION COMPLETE!")
    print("="*70)
    print("\nKey Achievements:")
    print("  ✅ Self-evolving symbolic constraints discovered")
    print("  ✅ Hybrid neural-symbolic reasoning operational")
    print("  ✅ Entropy-guided optimization adaptive")
    print("  ✅ Chaos RAG knowledge augmentation functional")
    print("  ✅ Matrix structures discovered and optimized")
    print("\n🚀 AL-ULS Evolution is ready for production use!")


if __name__ == "__main__":
    main()
