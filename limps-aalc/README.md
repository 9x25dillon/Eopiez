# LIMPS‑AALC Monorepo

## Run (Docker Compose)
```bash
cd deploy
docker compose up --build
```

- Admin API: http://localhost:8080
  - POST /pq/lease
  - POST /ds/select {"query":"hello", "top_k":50}
  - POST /ml2/train_step {"result_uri":"s3://..."}
  - POST /rfv/publish {"name":"rfv1", ...}

- Choppy Backend: http://localhost:8090
  - POST /ingest (multipart file)
  - POST /chunks/query?q=term&top_k=50

- Julia Ref: http://localhost:8008
  - POST /simplify {"exprs":["x^2+2x+1"]}
  - WS /ws {"fn":"simplify","expr":"x^2"}

## Notes
- **Gradient Normalization** implemented layer‑wise in `ml2_core.GradNormalizer`.
- **Skip‑preserve** blocks for incrementally adding layers.
- **Compound nodes** with softmax‑mixed activations.
- **Choppy** provides ingestion→chunking→entropy; adapter generates Chaos RAG/SQL prefix queries.
- **AL‑ULS** clients ready (HTTP + WS) to call Julia Ref service.
- **MatrixProcessor** hook included; route tensors to GPU.

---

## What to modify next

1. Wire real LIMPS entropy in entropy_lamps.py (replace toy estimator).

2. Connect Postgres via SQLAlchemy in admin-api/app.py for real PQ leases + repo persistence.

3. Add BPTT normalizer if using RNN/Transformer time unrolling (extend ml2_core.py).

4. Expose RFV training endpoints that snapshot and register feature vectors.

5. Choppy ↔ Admin‑API: on ingest completion, call /ds/select with chunk‑derived queries.

6. Security/Auth: add API keys/JWT as needed.