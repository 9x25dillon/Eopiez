# Getting Started with Eopiez

Welcome to Eopiez! This guide will help you get up and running quickly.

## What is Eopiez?

Eopiez is a hybrid AI/ML symbolic computation platform that combines:
- **Symbolic Pattern Detection** - Identifies thematic motifs in text
- **Neural-Symbolic Reasoning** - Bridges deep learning and symbolic AI
- **Quantum-Inspired Memory** - Advanced vector analysis and retrieval
- **Adaptive Learning** - Self-evolving symbolic constraints

## Quick Start (5 minutes)

### Prerequisites

- Docker & Docker Compose
- 8GB RAM minimum
- 10GB disk space

### 1. Clone and Start

```bash
git clone https://github.com/9x25dillon/Eopiez.git
cd Eopiez
docker-compose up
```

That's it! The system will start:
- ✓ API Gateway on `http://localhost:8000`
- ✓ Julia symbolic computation backend
- ✓ All microservices
- ✓ Redis cache
- ✓ PostgreSQL database

### 2. Verify Installation

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "services": {
    "motif_detector": "ready",
    "vectorizer": "ready",
    "limps": "ready",
    "qvnm": "ready",
    "al_uls": "ready"
  }
}
```

### 3. Run Your First Example

**Detect Motifs in Text:**
```bash
curl -X POST http://localhost:8000/api/v1/motif/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "I felt isolated, like a phantom in the machinery of war"}'
```

**Expected Response:**
```json
{
  "motifs": [
    {"category": "isolation", "pattern": "isolated", "score": 0.95},
    {"category": "phantom_pain", "pattern": "phantom", "score": 0.89},
    {"category": "technology_humanity", "pattern": "machinery", "score": 0.82},
    {"category": "war_peace", "pattern": "war", "score": 0.91}
  ]
}
```

## Usage Examples

### Complete Pipeline: Text → Symbolic Memory

```python
import requests

# Step 1: Detect motifs
response = requests.post('http://localhost:8000/api/v1/motif/detect', json={
    'text': 'The phantom pain of memories lost in the fog of war'
})
motifs = response.json()['motifs']

# Step 2: Vectorize
response = requests.post('http://localhost:8000/api/v1/vector/encode', json={
    'motifs': motifs
})
vector = response.json()['vector']

# Step 3: Store in symbolic memory
response = requests.post('http://localhost:8000/api/v1/limps/store', json={
    'vector': vector,
    'text': 'The phantom pain of memories lost in the fog of war',
    'context': {'source': 'user_input', 'timestamp': '2025-11-17'}
})
memory_id = response.json()['id']

# Step 4: Retrieve similar memories
response = requests.get('http://localhost:8000/api/v1/limps/retrieve', params={
    'query': 'memories of war',
    'limit': 5
})
similar_memories = response.json()['memories']
```

### QVNM: Vector Analysis

```python
import numpy as np
import requests

# Upload vector dataset
vectors = np.random.randn(1000, 128)  # 1000 vectors, 128 dimensions
np.savez('vectors.npz', vectors=vectors)

with open('vectors.npz', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/api/v1/qvnm/upload', files=files)

dataset_id = response.json()['dataset_id']

# Estimate intrinsic dimension
response = requests.get(f'http://localhost:8000/api/v1/qvnm/estimate/{dataset_id}')
intrinsic_dim = response.json()['intrinsic_dimension']
print(f"Estimated intrinsic dimension: {intrinsic_dim}")

# Query nearest neighbors
query_vector = np.random.randn(128).tolist()
response = requests.post(f'http://localhost:8000/api/v1/qvnm/query/{dataset_id}', json={
    'vector': query_vector,
    'k': 10
})
neighbors = response.json()['neighbors']
```

### AL-ULS: Adaptive Learning

```python
import requests

# Train adaptive model
response = requests.post('http://localhost:8000/api/v1/al-uls/train', json={
    'training_data': [
        {'input': 'example 1', 'target': 'label A'},
        {'input': 'example 2', 'target': 'label B'},
        # ... more examples
    ],
    'constraints': {
        'entropy_threshold': 0.7,
        'symbolic_rules': ['rule1', 'rule2']
    }
})
model_id = response.json()['model_id']

# Run inference
response = requests.post(f'http://localhost:8000/api/v1/al-uls/infer/{model_id}', json={
    'input': 'new example'
})
prediction = response.json()['prediction']
```

## Architecture Overview

```
┌─────────────────────────────────────────┐
│          Your Application                │
└──────────────┬──────────────────────────┘
               │ HTTP/REST
┌──────────────▼──────────────────────────┐
│        API Gateway (FastAPI)             │
│      http://localhost:8000               │
└──┬───┬───┬───┬───┬───┬───┬───┬──────────┘
   │   │   │   │   │   │   │   │
   │   │   │   │   │   │   │   │
┌──▼─┐ │   │   │   │   │   │   │
│Motif│ │   │   │   │   │   │   │
└────┘ │   │   │   │   │   │   │
   ┌───▼─┐ │   │   │   │   │   │
   │Vec  │ │   │   │   │   │   │
   └─────┘ │   │   │   │   │   │
       ┌───▼─┐ │   │   │   │   │
       │LiMps│ │   │   │   │   │
       └─────┘ │   │   │   │   │
           ┌───▼─┐ │   │   │   │
           │QVNM │ │   │   │   │
           └─────┘ │   │   │   │
               ┌───▼───┐   │   │
               │AL-ULS │   │   │
               └───────┘   │   │
                   ┌───────▼─┐ │
                   │Context  │ │
                   └─────────┘ │
                       ┌───────▼────┐
                       │Neutronics  │
                       └────────────┘
```

## Directory Structure

```
eopiez/
├── services/              # Microservices (future reorganization)
├── src/                   # Current Julia source
├── api.py                 # Current main API (being unified)
├── examples/              # Usage examples
├── tests/                 # Test suite
├── docker-compose.yml     # System startup
├── ARCHITECTURE.md        # Architecture details
└── README.md              # Project overview
```

## API Documentation

Interactive API documentation available at:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## Development Setup

### Without Docker (Advanced)

**Requirements:**
- Julia 1.9+
- Python 3.9+
- Redis
- PostgreSQL

**Install Julia Dependencies:**
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

**Install Python Dependencies:**
```bash
pip install -r requirements.txt
```

**Start Services Manually:**

Terminal 1 - Redis:
```bash
redis-server
```

Terminal 2 - PostgreSQL:
```bash
# Follow PostgreSQL installation for your OS
```

Terminal 3 - Julia Backend:
```bash
julia --project=. src/qvnm_server.jl
```

Terminal 4 - Python API:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

## Testing

```bash
# Run all tests
docker-compose run --rm test

# Run Python tests only
pytest tests/python

# Run Julia tests only
julia --project=. -e 'using Pkg; Pkg.test()'

# Run specific test
pytest tests/python/test_motif_detection.py -v
```

## Common Use Cases

### 1. Text Analysis Pipeline

**Goal:** Analyze text for symbolic patterns and store in memory

**Steps:**
1. Send text to motif detector
2. Vectorize detected motifs
3. Store in LiMps symbolic memory
4. Query for similar texts

**See:** `examples/text_analysis_pipeline.py`

### 2. Vector Dataset Analysis

**Goal:** Estimate intrinsic dimension of high-dimensional data

**Steps:**
1. Upload vector dataset (.npz or .jsonl)
2. Request intrinsic dimension estimation
3. Query for nearest neighbors
4. Visualize manifold structure

**See:** `examples/qvnm_analysis.py`

### 3. Adaptive Learning

**Goal:** Train model with symbolic constraints

**Steps:**
1. Prepare training data
2. Define symbolic constraints
3. Train AL-ULS model
4. Run inference with evolved constraints

**See:** `examples/al_uls_training.py`

### 4. Context Selection

**Goal:** Select optimal contexts for LLM prompting

**Steps:**
1. Provide candidate context pairs
2. Request entropy-based ranking
3. Receive ranked contexts with scores

**See:** `examples/context_selection.py`

## Configuration

### Environment Variables

Create `.env` file:
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Julia Backend
JULIA_HOST=localhost
JULIA_PORT=8080

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=eopiez
POSTGRES_USER=eopiez
POSTGRES_PASSWORD=changeme

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# Logging
LOG_LEVEL=INFO

# Model Paths
MODEL_CACHE_DIR=/models
```

### Service Ports

| Service | Port | URL |
|---------|------|-----|
| API Gateway | 8000 | http://localhost:8000 |
| Julia QVNM | 8080 | http://localhost:8080 |
| AL-ULS API | 8001 | http://localhost:8001 |
| PostgreSQL | 5432 | postgresql://localhost:5432 |
| Redis | 6379 | redis://localhost:6379 |
| Prometheus | 9090 | http://localhost:9090 |
| Grafana | 3000 | http://localhost:3000 |

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>
```

### Julia Dependencies Not Installing

```bash
# Clear Julia packages and reinstall
rm -rf ~/.julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### Services Not Healthy

```bash
# Check logs
docker-compose logs -f api-gateway

# Restart specific service
docker-compose restart api-gateway

# Rebuild from scratch
docker-compose down -v
docker-compose build --no-cache
docker-compose up
```

### Out of Memory

```bash
# Increase Docker memory limit (Docker Desktop)
# Settings → Resources → Memory → 8GB+

# Or reduce batch sizes in .env
MAX_BATCH_SIZE=32
```

## Performance Tuning

### For Development
- Use smaller models
- Reduce batch sizes
- Enable caching
- Limit concurrent requests

### For Production
- Enable GPU support
- Increase worker count
- Configure load balancing
- Set up CDN for static assets

## Next Steps

1. **Read the Architecture:** See `ARCHITECTURE.md` for detailed system design
2. **Explore Examples:** Check `examples/` directory for use cases
3. **Read API Docs:** Visit http://localhost:8000/docs
4. **Join Community:** [GitHub Discussions](https://github.com/9x25dillon/Eopiez/discussions)
5. **Contribute:** See `CONTRIBUTING.md` for guidelines

## Resources

- **Documentation:** `/docs` directory
- **Examples:** `/examples` directory
- **API Reference:** http://localhost:8000/docs
- **Architecture:** `ARCHITECTURE.md`
- **Tests:** `/tests` directory

## Getting Help

- **Issues:** https://github.com/9x25dillon/Eopiez/issues
- **Discussions:** https://github.com/9x25dillon/Eopiez/discussions
- **Email:** [maintainer email]

## What's Next?

- [ ] Complete the unified API gateway
- [ ] Migrate to microservices architecture
- [ ] Add WebSocket support
- [ ] Implement monitoring dashboards
- [ ] Add more examples and tutorials
- [ ] Create video walkthroughs

Welcome to Eopiez! 🚀
