#!/usr/bin/env python3
"""
Full Pipeline Example

This example demonstrates the complete Eopiez pipeline:
Text → Motif Detection → Vectorization → Symbolic Memory Storage → Retrieval
"""

import requests
import json
from datetime import datetime
from typing import Dict, List

# API endpoint
API_BASE = "http://localhost:8000/api/v1"

def step_1_detect_motifs(text: str) -> Dict:
    """Step 1: Detect motifs in text"""
    print(f"\n📍 Step 1: Detecting motifs...")
    response = requests.post(
        f"{API_BASE}/motif/detect",
        json={"text": text}
    )
    response.raise_for_status()
    result = response.json()

    motifs = result.get("motifs", [])
    print(f"   ✓ Found {len(motifs)} motifs")
    for motif in motifs[:3]:  # Show top 3
        print(f"     - {motif.get('category')}: {motif.get('pattern')} ({motif.get('score', 0):.2f})")

    return result

def step_2_vectorize(motifs: List[Dict]) -> Dict:
    """Step 2: Convert motifs to vector representation"""
    print(f"\n📍 Step 2: Vectorizing motifs...")
    response = requests.post(
        f"{API_BASE}/vector/encode",
        json={
            "motifs": motifs,
            "embedding_dim": 64,
            "entropy_threshold": 0.5,
            "compression_ratio": 0.8
        }
    )
    response.raise_for_status()
    result = response.json()

    message_state = result.get("message_state", {})
    print(f"   ✓ Generated {len(result.get('vector', []))}D vector")
    print(f"     Entropy score: {message_state.get('entropy_score', 0):.3f}")
    print(f"     Information density: {message_state.get('information_density', 0):.3f}")

    return result

def step_3_store_memory(text: str, vector: List[float], context: Dict) -> Dict:
    """Step 3: Store in symbolic memory (LiMps)"""
    print(f"\n📍 Step 3: Storing in symbolic memory...")
    response = requests.post(
        f"{API_BASE}/limps/store",
        json={
            "vector": vector,
            "text": text,
            "context": context
        }
    )
    response.raise_for_status()
    result = response.json()

    memory_id = result.get("id", "unknown")
    print(f"   ✓ Stored with ID: {memory_id}")

    return result

def step_4_retrieve_similar(query: str, limit: int = 5) -> Dict:
    """Step 4: Retrieve similar memories"""
    print(f"\n📍 Step 4: Retrieving similar memories...")
    response = requests.get(
        f"{API_BASE}/limps/retrieve",
        params={
            "query": query,
            "limit": limit,
            "threshold": 0.0
        }
    )
    response.raise_for_status()
    result = response.json()

    memories = result.get("memories", [])
    print(f"   ✓ Found {len(memories)} similar memories")
    for i, memory in enumerate(memories[:3], 1):
        print(f"     {i}. Similarity: {memory.get('similarity', 0):.3f}")
        print(f"        Text: {memory.get('text', '')[:60]}...")

    return result

def run_pipeline_for_text(text: str, context: Dict = None):
    """Run the complete pipeline for a given text"""
    if context is None:
        context = {
            "timestamp": datetime.now().isoformat(),
            "source": "example_script"
        }

    print("\n" + "="*80)
    print("RUNNING EOPIEZ PIPELINE")
    print("="*80)
    print(f"\nInput text: {text}\n")

    try:
        # Step 1: Detect motifs
        motif_result = step_1_detect_motifs(text)
        motifs = motif_result.get("motifs", [])

        if not motifs:
            print("\n⚠️  No motifs detected, skipping vectorization")
            return

        # Step 2: Vectorize
        vector_result = step_2_vectorize(motifs)
        vector = vector_result.get("vector", [])

        # Step 3: Store
        memory_result = step_3_store_memory(text, vector, context)

        print("\n" + "="*80)
        print("✅ PIPELINE COMPLETE")
        print("="*80)
        print(f"\nMemory stored successfully!")
        print(f"  ID: {memory_result.get('id')}")
        print(f"  Vector dimension: {len(vector)}")
        print(f"  Motifs detected: {len(motifs)}")

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the Eopiez API is running:")
        print("  docker-compose up")
        print("  or")
        print("  make start")

def main():
    """Run full pipeline examples"""
    print("\n🚀 Eopiez Full Pipeline Demo\n")

    # Example texts to process
    texts = [
        "The soldier stood alone in the wasteland, memories of war fading like ghosts in the machinery.",
        "Snake's wisdom echoed through time: war has changed, but the human spirit endures.",
        "In the strands of connection between past and future, we find phantom pain from battles lost.",
        "The boss faced the ultimate question: what does it mean to be human in an age of technology?",
        "Isolation breeds reflection; in loneliness, we discover the meta-narrative of our existence."
    ]

    # Process each text through the pipeline
    for i, text in enumerate(texts, 1):
        print(f"\n{'#'*80}")
        print(f"# TEXT {i}/{len(texts)}")
        print(f"{'#'*80}")

        run_pipeline_for_text(
            text,
            context={
                "timestamp": datetime.now().isoformat(),
                "source": "full_pipeline_example",
                "batch": f"example_{i}"
            }
        )

    # Now demonstrate retrieval
    print("\n\n" + "="*80)
    print("TESTING RETRIEVAL")
    print("="*80)

    queries = [
        "memories of war",
        "human and technology",
        "loneliness and isolation"
    ]

    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        try:
            step_4_retrieve_similar(query, limit=3)
        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {e}")
            break

    print("\n\n✨ Demo complete!")
    print("\nNext steps:")
    print("  - Explore the API docs: http://localhost:8000/docs")
    print("  - Try the QVNM example: python examples/qvnm_example.py")
    print("  - Launch Jupyter notebook: make notebook")
    print()

if __name__ == "__main__":
    main()
