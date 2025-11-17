#!/usr/bin/env python3
"""
Motif Detection Example

This example demonstrates how to use the Eopiez motif detection system
to identify symbolic patterns in text.
"""

import requests
import json
from typing import List, Dict

# API endpoint
API_BASE = "http://localhost:8000/api/v1"

def detect_motifs(text: str) -> Dict:
    """Detect motifs in text"""
    response = requests.post(
        f"{API_BASE}/motif/detect",
        json={"text": text}
    )
    response.raise_for_status()
    return response.json()

def print_motif_analysis(text: str, result: Dict):
    """Pretty print motif analysis results"""
    print("\n" + "="*80)
    print("MOTIF DETECTION ANALYSIS")
    print("="*80)
    print(f"\nText: {text}\n")

    print("-" * 80)
    print("DETECTED MOTIFS")
    print("-" * 80)

    motifs = result.get("motifs", [])
    if motifs:
        for i, motif in enumerate(motifs, 1):
            print(f"\n{i}. Category: {motif.get('category', 'Unknown')}")
            print(f"   Pattern: {motif.get('pattern', 'N/A')}")
            print(f"   Score: {motif.get('score', 0.0):.2f}")
            if 'context' in motif:
                print(f"   Context: {motif.get('context', '')}")
    else:
        print("No motifs detected")

    print("\n" + "-" * 80)
    print("DOCUMENT ANALYSIS")
    print("-" * 80)

    doc_analysis = result.get("document_analysis", {})
    if doc_analysis:
        print(json.dumps(doc_analysis, indent=2))
    else:
        print("No document analysis available")

    print("\n" + "="*80 + "\n")

def main():
    """Run motif detection examples"""
    # Example texts with various Kojima-esque themes
    examples = [
        "I felt isolated, like a phantom in the machinery of war. The snake's wisdom echoed in my memory.",
        "War has changed. Technology and humanity are intertwined, creating a new kind of soldier.",
        "The strands of connection between us persist, even as memories fade into the fog.",
        "A phantom pain lingers from what was lost, a reminder of battles fought in the name of peace.",
        "The boss stood alone on the battlefield, a meta-narrative unfolding in real-time."
    ]

    print("\n🎮 Eopiez Motif Detection Examples\n")

    for i, text in enumerate(examples, 1):
        print(f"\n{'='*80}")
        print(f"EXAMPLE {i}/{len(examples)}")
        try:
            result = detect_motifs(text)
            print_motif_analysis(text, result)
        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {e}")
            print("\nMake sure the Eopiez API is running:")
            print("  docker-compose up")
            print("  or")
            print("  make start")
            break

    print("\n✨ Done! Check out the other examples:")
    print("  python examples/full_pipeline_example.py")
    print("  python examples/qvnm_example.py")
    print()

if __name__ == "__main__":
    main()
