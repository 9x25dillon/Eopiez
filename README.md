# Eopiez

> A hybrid AI/ML symbolic computation platform combining neural networks, symbolic reasoning, and quantum-inspired algorithms for advanced pattern detection and memory processing.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Julia 1.9+](https://img.shields.io/badge/julia-1.9+-purple.svg)](https://julialang.org/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

## What is Eopiez?

Eopiez is a sophisticated platform that bridges symbolic AI and neural networks to detect, analyze, and store thematic patterns in text. It combines multiple cutting-edge technologies:

- **🎯 Motif Detection** - Identifies symbolic patterns across 9 thematic categories
- **🧠 Neural-Symbolic Reasoning** - Bridges deep learning and symbolic computation
- **💾 LiMps Symbolic Memory** - Graph-based memory with temporal awareness
- **🌌 QVNM** - Quantum-inspired vector analysis and manifold learning
- **🔄 AL-ULS Evolution** - Adaptive learning with self-evolving constraints
- **⚛️ Neutronics Surrogate** - Physics simulation via polynomial regression

## Quick Start

### Prerequisites

- Docker & Docker Compose (recommended)
- **OR** Python 3.9+ and Julia 1.9+ for local development

### Start with Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/9x25dillon/Eopiez.git
cd Eopiez

# Start all services
docker-compose up

# Or use Makefile
make start
```

That's it! Access the API at:
- **API Gateway:** http://localhost:8000
- **Interactive Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health

### Your First Request

Detect motifs in text:
```bash
curl -X POST http://localhost:8000/api/v1/motif/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "I felt isolated, like a phantom in the machinery of war"}'
```

Response:
```json
{
  "motifs": [
    {"category": "isolation", "pattern": "isolated", "score": 0.95},
    {"category": "phantom_pain", "pattern": "phantom", "score": 0.89},
    {"category": "technology_humanity", "pattern": "machinery", "score": 0.82},
    {"category": "war_peace", "pattern": "war", "score": 0.91}
  ],
  "document_analysis": {...}
}
```

## Features

### 🎯 Motif Detection

Identifies thematic patterns across **9 categories**:

| Category | Examples | Use Case |
|----------|----------|----------|
| Isolation & Loneliness | "isolated", "alone", "solitary" | Emotional analysis |
| Snake Symbolism | "serpent", "venom", "snake" | Metaphor detection |
| Memory & Identity | "remember", "forget", "amnesia" | Cognitive themes |
| Strands of Connection | "bond", "link", "thread" | Relationship analysis |
| Technology & Humanity | "machine", "cyborg", "digital" | Human-tech themes |
| Phantom Pain | "phantom", "ghost", "haunted" | Loss & trauma |
| War & Peace | "battle", "conflict", "war" | Violence themes |
| Boss Themes | "leader", "authority", "power" | Leadership motifs |
| Meta-Narrative | "story", "narrative", "author" | Self-reference |

**API Endpoint:** `POST /api/v1/motif/detect`

### 🧮 Message Vectorization

Converts detected motifs into **symbolic vector representations**:
- Symbolic computation via Symbolics.jl
- Entropy-based compression
- Information density scoring
- 64-512 dimensional embeddings

**API Endpoint:** `POST /api/v1/vector/encode`

### 💾 LiMps (Symbolic Memory)

**Li**nked **M**emory **P**attern**s** - A graph-based memory system:
- Temporal decay modeling
- Contextual relationship mapping
- Semantic similarity search
- Symbolic memory graph

**API Endpoints:**
- `POST /api/v1/limps/store` - Store memories
- `GET /api/v1/limps/retrieve` - Retrieve similar memories
- `POST /api/v1/limps/relate` - Create relationships

### 🌌 QVNM (Quantum Vector Neural Memory)

**Intrinsic dimension estimation** using Costa-Hero algorithm:
- Upload high-dimensional vector datasets
- Estimate true dimensionality
- Manifold-aware nearest neighbor search
- 2D preview generation

**API Endpoints:**
- `POST /api/v1/qvnm/upload` - Upload vectors (.npz, .jsonl)
- `POST /api/v1/qvnm/estimate/{session_id}` - Estimate dimension
- `POST /api/v1/qvnm/query/{session_id}` - Nearest neighbors
- `POST /api/v1/qvnm/preview/{session_id}` - 2D manifold preview

### 🔄 AL-ULS Evolution

**A**daptive **L**earning with **U**niversal **L**ogical **S**ymbolic constraints:
- Neural-symbolic hybrid architecture
- Entropy-guided optimization
- Chaos-augmented retrieval (Chaos RAG)
- Self-evolving symbolic constraints
- Matrix structure discovery

**API Endpoints:**
- `POST /api/v1/al-uls/train` - Train adaptive model
- `POST /api/v1/al-uls/infer/{model_id}` - Run inference
- `POST /api/v1/al-uls/evolve/{model_id}` - Evolve constraints

### 🎛️ Context Selection

Select optimal contexts using **motif-based entropy scoring**:
- Dual-context ranking
- Information density analysis
- Symbolic richness scoring
- Ideal for LLM prompt engineering

**API Endpoint:** `POST /api/v1/context/select`

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                              │
│  (HTTP Requests, Python SDK, CLI Tools)                          │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                   API GATEWAY (FastAPI)                          │
│  - Unified Routing  - Authentication  - Rate Limiting            │
└─────┬──────┬──────┬──────┬──────┬──────┬──────┬─────────────────┘
      │      │      │      │      │      │      │
┌─────▼──┐ ┌─▼────┐ ┌▼────┐ ┌▼───┐ ┌▼───┐ ┌▼──┐ ┌▼────────┐
│ Motif  │ │Vector│ │LiMps│ │QVNM│ │AL  │ │Ctx│ │Neutron  │
│Detector│ │ izer │ │Mem  │ │    │ │ULS │ │Sel│ │Surrogate│
└────────┘ └──────┘ └─────┘ └────┘ └────┘ └───┘ └─────────┘
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## Complete Pipeline Example

```python
import requests

API_BASE = "http://localhost:8000/api/v1"

# Step 1: Detect motifs
response = requests.post(f"{API_BASE}/motif/detect", json={
    'text': 'The phantom pain of memories lost in the fog of war'
})
motifs = response.json()['motifs']

# Step 2: Vectorize
response = requests.post(f"{API_BASE}/vector/encode", json={
    'motifs': motifs,
    'embedding_dim': 64
})
vector = response.json()['vector']

# Step 3: Store in symbolic memory
response = requests.post(f"{API_BASE}/limps/store", json={
    'vector': vector,
    'text': 'The phantom pain of memories lost in the fog of war',
    'context': {'source': 'user_input'}
})
memory_id = response.json()['id']

# Step 4: Retrieve similar memories
response = requests.get(f"{API_BASE}/limps/retrieve", params={
    'query': 'memories of war',
    'limit': 5
})
similar_memories = response.json()['memories']
```

More examples in [`examples/`](examples/) directory.

## Documentation

| Document | Description |
|----------|-------------|
| [GETTING_STARTED.md](GETTING_STARTED.md) | Quick start guide with examples |
| [ARCHITECTURE.md](ARCHITECTURE.md) | System architecture and design |
| [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) | Migration guide for new structure |
| [API Docs](http://localhost:8000/docs) | Interactive API documentation |

## Development

### Using Makefile

```bash
make help          # Show all available commands
make start         # Start all services
make stop          # Stop all services
make test          # Run tests
make logs          # View logs
make notebook      # Start Jupyter notebook
make monitoring    # Start Prometheus + Grafana
```

### Manual Installation (Local Development)

**Install Julia dependencies:**
```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

**Install Python dependencies:**
```bash
pip install -r requirements.txt
```

**Start services:**
```bash
# Terminal 1: Julia backend
julia --project=. src/qvnm_server.jl

# Terminal 2: Python API
uvicorn api_gateway:app --reload --port 8000
```

### Testing

```bash
# Run all tests
make test

# Run specific tests
pytest tests/python -v
julia --project=. -e 'using Pkg; Pkg.test()'

# With coverage
make test-coverage
```

## Technology Stack

### Backend
- **Julia 1.9+** - Symbolic computation, numerical processing
- **Python 3.9+** - API layer, orchestration, ML training

### Key Libraries
- **Julia:** Symbolics.jl, HTTP.jl, Graphs.jl, NearestNeighbors.jl
- **Python:** FastAPI, PyTorch, NumPy, SciPy, Transformers

### Infrastructure
- Docker & Docker Compose
- PostgreSQL (persistence)
- Redis (caching)
- Prometheus & Grafana (monitoring)

## Project Structure

```
eopiez/
├── api_gateway.py         # Unified API gateway (new!)
├── docker-compose.yml     # One-command startup
├── Makefile               # Development commands
├── ARCHITECTURE.md        # Architecture documentation
├── GETTING_STARTED.md     # Quick start guide
│
├── deployment/            # Deployment configs
│   ├── docker/            # Dockerfiles
│   └── monitoring/        # Monitoring configs
│
├── examples/              # Usage examples
│   ├── motif_detection_example.py
│   ├── full_pipeline_example.py
│   └── qvnm_example.py
│
├── src/                   # Julia source code
│   ├── qvnm_server.jl
│   ├── motif_detection/
│   ├── limps/
│   └── quantum_neural/
│
├── al-uls-evolution/      # AL-ULS service
│   └── api/
│
└── tests/                 # Test suite
    ├── integration/
    ├── python/
    └── julia/
```

## Use Cases

### 1. **Thematic Text Analysis**
Analyze literature, articles, or user content for symbolic patterns.

### 2. **LLM Context Selection**
Rank candidate contexts by information density for optimal prompting.

### 3. **High-Dimensional Data Analysis**
Estimate intrinsic dimensionality of embeddings or feature spaces.

### 4. **Adaptive Learning Systems**
Build models that evolve their constraints based on symbolic rules.

### 5. **Symbolic Memory Graphs**
Create knowledge graphs with temporal and contextual awareness.

## Performance

- **Motif Detection:** ~100 texts/sec
- **Vectorization:** ~200 ops/sec
- **QVNM Estimation:** ~10K vectors in <5s
- **LiMps Retrieval:** <100ms for 10K memories

## Roadmap

- [x] Unified API Gateway
- [x] Docker Compose setup
- [x] Comprehensive documentation
- [ ] WebSocket support for streaming
- [ ] GraphQL API alternative
- [ ] Enhanced monitoring dashboards
- [ ] Python/Julia SDK packages
- [ ] Web UI dashboard
- [ ] Multi-region deployment
- [ ] Advanced security (OAuth2, JWT)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use Eopiez in your research, please cite:

```bibtex
@software{eopiez2025,
  title={Eopiez: A Hybrid Symbolic Computation Platform},
  author={Your Name},
  year={2025},
  url={https://github.com/9x25dillon/Eopiez}
}
```

## Acknowledgments

- Inspired by Kojima-esque narrative themes
- Built with Julia and Python communities
- Powered by open-source ML/AI libraries

## Support

- **Documentation:** [GETTING_STARTED.md](GETTING_STARTED.md)
- **Issues:** https://github.com/9x25dillon/Eopiez/issues
- **Discussions:** https://github.com/9x25dillon/Eopiez/discussions

---

**Made with ❤️ and symbolic computation**

*"In the strands of connection between code and meaning, we find the phantom pain of memories encoded in silicon."*
