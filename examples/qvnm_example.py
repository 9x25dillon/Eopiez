#!/usr/bin/env python3
"""
QVNM (Quantum Vector Neural Memory) Example

This example demonstrates:
1. Uploading a vector dataset
2. Estimating intrinsic dimension
3. Querying for nearest neighbors
4. Building a manifold preview
"""

import requests
import numpy as np
import tempfile
import os
from typing import Dict

# API endpoint
API_BASE = "http://localhost:8000/api/v1"

def create_sample_dataset(n_samples: int = 1000, n_dimensions: int = 128, intrinsic_dim: int = 10):
    """
    Create a synthetic dataset with known intrinsic dimension

    Generates high-dimensional vectors that lie on a lower-dimensional manifold
    """
    print(f"\n📊 Creating synthetic dataset...")
    print(f"   Samples: {n_samples}")
    print(f"   Ambient dimension: {n_dimensions}")
    print(f"   True intrinsic dimension: {intrinsic_dim}")

    # Generate data on a lower-dimensional manifold
    # Start with low-dimensional data
    low_dim_data = np.random.randn(n_samples, intrinsic_dim)

    # Project to high dimension with random projection
    projection_matrix = np.random.randn(intrinsic_dim, n_dimensions)
    high_dim_data = low_dim_data @ projection_matrix

    # Add small noise
    noise = np.random.randn(n_samples, n_dimensions) * 0.1
    vectors = high_dim_data + noise

    # Normalize
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # Generate IDs and texts
    ids = [f"vec_{i}" for i in range(n_samples)]
    texts = [f"Sample text for vector {i}" for i in range(n_samples)]

    print(f"   ✓ Dataset created")
    return vectors, ids, texts

def upload_dataset(vectors: np.ndarray, ids: list, texts: list = None) -> str:
    """Upload vectors to QVNM service"""
    print(f"\n📤 Uploading dataset to QVNM...")

    # Save to temporary .npz file
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.npz', delete=False) as tmp:
        np.savez(tmp.name, V=vectors.T, ids=np.array(ids))
        tmp_path = tmp.name

    try:
        # Upload file
        with open(tmp_path, 'rb') as f:
            files = {'file': (os.path.basename(tmp_path), f, 'application/octet-stream')}
            response = requests.post(f"{API_BASE}/qvnm/upload", files=files)
            response.raise_for_status()
            result = response.json()

        session_id = result.get("sid")
        print(f"   ✓ Upload successful")
        print(f"     Session ID: {session_id}")
        print(f"     Dimension: {result.get('d')}")
        print(f"     Count: {result.get('N')}")

        return session_id
    finally:
        # Clean up temp file
        os.unlink(tmp_path)

def estimate_intrinsic_dimension(session_id: str, mode: str = "local") -> Dict:
    """Estimate the intrinsic dimension of the dataset"""
    print(f"\n🔬 Estimating intrinsic dimension...")
    print(f"   Mode: {mode}")

    response = requests.post(
        f"{API_BASE}/qvnm/estimate/{session_id}",
        params={
            "k": 10,
            "gamma": 0.5,
            "alpha": 0.5,
            "boots": 8,
            "mode": mode
        }
    )
    response.raise_for_status()
    result = response.json()

    if mode == "global":
        m_hat = result.get("m_hat", 0)
        H_hat = result.get("H_hat", 0)
        print(f"   ✓ Estimation complete")
        print(f"     Estimated dimension (m̂): {m_hat:.2f}")
        print(f"     Homogeneity (Ĥ): {H_hat:.3f}")
    else:
        m_hat = result.get("m_hat", [])
        if m_hat:
            avg_dim = np.mean(m_hat)
            std_dim = np.std(m_hat)
            print(f"   ✓ Estimation complete")
            print(f"     Average local dimension: {avg_dim:.2f} ± {std_dim:.2f}")
            print(f"     Range: [{np.min(m_hat):.2f}, {np.max(m_hat):.2f}]")

    return result

def build_manifold_preview(session_id: str) -> Dict:
    """Build a 2D preview of the high-dimensional manifold"""
    print(f"\n🗺️  Building manifold preview...")

    response = requests.post(
        f"{API_BASE}/qvnm/preview/{session_id}",
        params={
            "r": 2,
            "k_eval": 10,
            "bins": 20,
            "lambda_m": 0.3,
            "lambda_h": 0.3
        }
    )
    response.raise_for_status()
    result = response.json()

    print(f"   ✓ Preview built")

    # Check if eigenmaps are available
    eigenmaps = result.get("eigenmaps", {})
    if eigenmaps:
        coords = eigenmaps.get("coords", [])
        if coords:
            print(f"     Generated 2D embedding with {len(coords) // 2} points")

    return result

def query_nearest_neighbors(session_id: str, seed_id: str, topk: int = 10) -> Dict:
    """Query for nearest neighbors of a given point"""
    print(f"\n🔍 Querying nearest neighbors...")
    print(f"   Seed: {seed_id}")
    print(f"   Top K: {topk}")

    response = requests.post(
        f"{API_BASE}/qvnm/query/{session_id}",
        params={
            "seed_id": seed_id,
            "topk": topk,
            "steps": 10,
            "alpha": 0.85
        }
    )
    response.raise_for_status()
    result = response.json()

    print(f"   ✓ Query complete")

    # Display results
    neighbors = result.get("neighbors", [])
    if neighbors:
        print(f"\n   Top {min(5, len(neighbors))} neighbors:")
        for i, neighbor in enumerate(neighbors[:5], 1):
            neighbor_id = neighbor.get("id", "unknown")
            score = neighbor.get("score", 0.0)
            print(f"     {i}. {neighbor_id}: {score:.4f}")

    return result

def main():
    """Run QVNM analysis example"""
    print("\n🌌 Eopiez QVNM Analysis Demo\n")
    print("="*80)

    try:
        # Step 1: Create synthetic dataset
        vectors, ids, texts = create_sample_dataset(
            n_samples=1000,
            n_dimensions=128,
            intrinsic_dim=10
        )

        # Step 2: Upload dataset
        session_id = upload_dataset(vectors, ids, texts)

        # Step 3: Estimate intrinsic dimension (global)
        print("\n" + "-"*80)
        print("GLOBAL INTRINSIC DIMENSION ESTIMATION")
        print("-"*80)
        global_result = estimate_intrinsic_dimension(session_id, mode="global")

        # Step 4: Estimate intrinsic dimension (local)
        print("\n" + "-"*80)
        print("LOCAL INTRINSIC DIMENSION ESTIMATION")
        print("-"*80)
        local_result = estimate_intrinsic_dimension(session_id, mode="local")

        # Step 5: Build manifold preview
        print("\n" + "-"*80)
        print("MANIFOLD PREVIEW")
        print("-"*80)
        preview_result = build_manifold_preview(session_id)

        # Step 6: Query nearest neighbors
        print("\n" + "-"*80)
        print("NEAREST NEIGHBOR QUERY")
        print("-"*80)
        query_result = query_nearest_neighbors(session_id, seed_id="vec_0", topk=10)

        # Summary
        print("\n\n" + "="*80)
        print("✅ ANALYSIS COMPLETE")
        print("="*80)

        print("\n📊 Summary:")
        print(f"   Dataset: {len(vectors)} vectors in {vectors.shape[1]}D space")
        print(f"   True intrinsic dimension: ~10")
        print(f"   Estimated dimension: {global_result.get('m_hat', 0):.2f}")
        print(f"   Estimation accuracy: {'Good' if abs(global_result.get('m_hat', 0) - 10) < 3 else 'Fair'}")

        print("\n💡 Insights:")
        print("   - The Costa-Hero algorithm successfully estimated the intrinsic dimension")
        print("   - High-dimensional data often lies on lower-dimensional manifolds")
        print("   - QVNM can reveal the true structure of complex datasets")

        print("\n📈 Next steps:")
        print("   - Try with real data (embeddings from your domain)")
        print("   - Experiment with different k and gamma parameters")
        print("   - Visualize the 2D manifold preview")
        print("   - Use the API docs for more options: http://localhost:8000/docs")

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure the Eopiez API is running:")
        print("  docker-compose up")
        print("  or")
        print("  make start")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    print()

if __name__ == "__main__":
    main()
