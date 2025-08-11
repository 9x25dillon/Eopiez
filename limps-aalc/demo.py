#!/usr/bin/env python3
"""
LIMPS-AALC Demo Script

This script demonstrates the integrated functionality of the LIMPS-AALC monorepo
without requiring full Docker deployment.
"""

import sys
import json
import asyncio
from pathlib import Path

# Add service paths
limps_root = Path(__file__).parent
sys.path.insert(0, str(limps_root / "services" / "admin-api"))
sys.path.insert(0, str(limps_root / "services" / "choppy-backend"))

def demo_admin_api():
    """Demonstrate admin API functionality"""
    print("🔧 Admin API Demo")
    print("-" * 30)
    
    # Test data selection
    from ds_adapter import simple_prefix_sql
    query = "machine learning"
    sql = simple_prefix_sql(query, k=20)
    print(f"Data Selection Query: '{query}'")
    print(f"Generated SQL: {sql}")
    print()
    
    # Test coaching system
    from coach import coach_update
    
    print("Training Coach Demo:")
    initial_state = {"lr": 1e-3, "top_k": 50, "entropy_floor": 3.0}
    print(f"Initial State: {initial_state}")
    
    # Scenario 1: Loss not improving
    metrics1 = {"dev_loss_delta": 0.05}
    entropy1 = {"avg_token_entropy": 4.2}
    state1 = coach_update(metrics1, entropy1, initial_state.copy())
    print(f"After poor loss (delta=0.05): {state1}")
    
    # Scenario 2: Low entropy
    metrics2 = {"dev_loss_delta": -0.02}
    entropy2 = {"avg_token_entropy": 2.1}
    state2 = coach_update(metrics2, entropy2, initial_state.copy())
    print(f"After low entropy (2.1): {state2}")
    print()

def demo_choppy_backend():
    """Demonstrate choppy backend functionality"""
    print("📄 Choppy Backend Demo")
    print("-" * 30)
    
    # Test text chunking
    from chunker import chunk_text
    
    sample_text = """
    The LIMPS-AALC system represents a comprehensive integration of machine learning,
    symbolic computation, and adaptive learning techniques. It combines gradient normalization
    with skip-preserve architectures and compound nodes to create robust neural networks.
    The system includes entropy-based coaching for dynamic learning rate adjustment and
    sophisticated data selection algorithms for improved training efficiency.
    """
    
    chunks = chunk_text(sample_text.strip(), max_tokens=20, overlap=0.2)
    print(f"Original text length: {len(sample_text.split())} words")
    print(f"Number of chunks: {len(chunks)}")
    print(f"First chunk: {chunks[0][:100]}...")
    print()
    
    # Test entropy analysis
    from entropy_lamps import entropy_report
    
    ent_report = entropy_report(chunks)
    print("Entropy Analysis:")
    print(f"Average entropy: {ent_report['avg']:.3f}")
    print(f"Min entropy: {ent_report['min']:.3f}")
    print(f"Max entropy: {ent_report['max']:.3f}")
    print()
    
    # Test SQL generation
    from adapters.chaos_sql import generate_sql
    
    queries = ["neural networks", "machine learning", "symbolic computation"]
    print("Chaos SQL Generation:")
    for q in queries:
        sql = generate_sql(q, top_k=15)
        print(f"  '{q}' -> {sql}")
    print()

def demo_integration_flow():
    """Demonstrate end-to-end integration flow"""
    print("🔄 Integration Flow Demo")
    print("-" * 30)
    
    # Simulate a complete workflow
    from chunker import chunk_text
    from entropy_lamps import entropy_report
    from adapters.chaos_sql import generate_sql
    from ds_adapter import simple_prefix_sql
    from coach import coach_update
    
    # Step 1: Ingest content
    content = "Advanced machine learning with symbolic reasoning and adaptive optimization"
    chunks = chunk_text(content, max_tokens=5, overlap=0.1)
    print(f"1. Content chunked into {len(chunks)} segments")
    
    # Step 2: Analyze entropy
    entropy = entropy_report(chunks)
    print(f"2. Entropy analysis: avg={entropy['avg']:.2f}")
    
    # Step 3: Generate queries from chunks
    queries = [chunk.split()[0] for chunk in chunks[:3]]  # First word of each chunk
    print(f"3. Generated queries: {queries}")
    
    # Step 4: Create data selection SQL
    sql_queries = [simple_prefix_sql(q, k=10) for q in queries]
    print(f"4. Data selection queries created: {len(sql_queries)} SQL statements")
    
    # Step 5: Simulate training feedback
    training_state = {"lr": 1e-3, "top_k": 50, "entropy_floor": 3.0}
    metrics = {"dev_loss_delta": 0.01}
    entropy_feedback = {"avg_token_entropy": entropy['avg']}
    
    updated_state = coach_update(metrics, entropy_feedback, training_state)
    print(f"5. Training state updated: lr={updated_state['lr']:.1e}, top_k={updated_state['top_k']}")
    print()

def demo_ml_architecture():
    """Demonstrate ML architecture concepts (conceptual)"""
    print("🧠 ML Architecture Demo (Conceptual)")
    print("-" * 40)
    
    print("CompoundNode Architecture:")
    print("  - Linear branch: Direct transformation")
    print("  - ReLU branch: Non-linear activation")
    print("  - Sigmoid branch: Bounded activation")
    print("  - Softmax mixing: Adaptive branch weighting")
    print()
    
    print("SkipPreserveBlock:")
    print("  - New path: CompoundNode transformation")
    print("  - Skip path: Identity connection (initially zero)")
    print("  - Output: sum of both paths")
    print()
    
    print("GradNormalizer:")
    print("  - Layer-wise gradient analysis")
    print("  - Max/L2 norm computation")
    print("  - Adaptive scaling (0.1x to 10x)")
    print("  - Prevents gradient explosion/vanishing")
    print()

def main():
    """Run the complete demo"""
    print("🚀 LIMPS-AALC Monorepo Demo")
    print("=" * 50)
    print()
    
    demo_admin_api()
    demo_choppy_backend()
    demo_integration_flow()
    demo_ml_architecture()
    
    print("✅ Demo Complete!")
    print()
    print("🐳 To run the full system:")
    print("  cd deploy")
    print("  docker compose up --build")
    print()
    print("🔗 Service endpoints:")
    print("  Admin API:      http://localhost:8080")
    print("  Choppy Backend: http://localhost:8090")
    print("  Julia Ref:      http://localhost:8008")
    print()
    print("📊 API Examples:")
    print("  curl -X POST http://localhost:8080/pq/lease")
    print("  curl -X POST http://localhost:8080/ds/select -H 'Content-Type: application/json' -d '{\"query\":\"test\",\"top_k\":10}'")
    print("  curl -X POST http://localhost:8090/chunks/query?q=example&top_k=5")

if __name__ == "__main__":
    main()