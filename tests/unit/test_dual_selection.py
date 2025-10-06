import os
import json
import pytest
from fastapi.testclient import TestClient

import api as service


@pytest.fixture(scope="module")
def client():
    # Point to a default Julia base; in CI this can be a mocked server.
    os.environ.setdefault("JULIA_BASE", "http://localhost:9000")
    return TestClient(service.app)


def test_dual_select_contexts_empty(client):
    r = client.post("/dual/select_contexts", json={"candidates": []})
    assert r.status_code == 400


def test_dual_select_contexts_schema(client, monkeypatch):
    # Mock out Julia calls
    def fake_motif(texts):
        return [{
            "document_analysis": {
                "detected_motifs": {
                    "isolation": {"confidence": 0.9}
                }
            },
            "motif_tokens": [{
                "name": "isolation",
                "properties": {"frequency": 1, "confidence": 0.9, "weight": 0.9},
                "weight": 0.9,
                "context": ["technology"]
            }]
        } for _ in texts]

    def fake_vec(tokens, embedding_dim=64, entropy_threshold=0.5, compression_ratio=0.8):
        return {
            "message_state": {
                "entropy_score": 1.23,
                "information_density": 0.01,
                "vector_representation": [0.0] * embedding_dim,
            }
        }

    monkeypatch.setattr(service, "_motif_detect_local", fake_motif)
    monkeypatch.setattr(service, "_vectorize_tokens_local", fake_vec)

    payload = {"candidates": [{"id": "c1", "text": "alone in the digital desert"}]}
    r = client.post("/dual/select_contexts", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "ranked" in data and len(data["ranked"]) == 1
    item = data["ranked"][0]
    assert item["id"] == "c1"
    assert item["score"] == pytest.approx(1.23)
    assert "motif_tokens" in item and item["motif_tokens"][0]["name"] == "isolation"
