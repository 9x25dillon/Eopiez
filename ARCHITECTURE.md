# Eopiez Architecture

## Overview

Eopiez is a hybrid AI/ML symbolic computation platform that combines neural networks, symbolic reasoning, and quantum-inspired algorithms for advanced pattern detection and memory processing.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                              │
│  (HTTP Requests, WebSocket Connections, CLI Tools)              │
└────────────────────┬────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────────────┐
│                   API GATEWAY (FastAPI)                          │
│  - Request Routing                                               │
│  - Authentication & Rate Limiting                                │
│  - Response Formatting                                           │
└─────┬──────┬──────┬──────┬──────┬──────┬──────┬─────────────────┘
      │      │      │      │      │      │      │
      │      │      │      │      │      │      │
┌─────▼──┐ ┌─▼────┐ ┌▼────┐ ┌▼───┐ ┌▼───┐ ┌▼──┐ ┌▼────────┐
│ Motif  │ │Vector│ │LiMps│ │QVNM│ │AL  │ │Ctx│ │Neutron  │
│Detector│ │ izer │ │Mem  │ │    │ │ULS │ │Sel│ │Surrogate│
└────────┘ └──────┘ └─────┘ └────┘ └────┘ └───┘ └─────────┘
```

## Core Components

### 1. **Motif Detector**
**Purpose:** Identifies symbolic patterns in text (Kojima-esque themes)

**Location:** `/services/motif-detection`

**Input:** Text strings
**Output:** Detected motif tokens with scores

**Categories:**
- Isolation & Loneliness
- Snake Symbolism
- Memory & Identity
- Strands of Connection
- Technology & Humanity
- Phantom Pain
- War & Peace
- Boss Themes
- Meta-Narrative

### 2. **Message Vectorizer**
**Purpose:** Transforms motifs into symbolic vector representations

**Location:** `/services/message-vectorizer`

**Input:** Motif tokens
**Output:** Symbolic state vectors

**Pipeline:**
```
Text → Motif Detection → Symbolic Encoding → State Vector
```

### 3. **LiMps (Symbolic Memory)**
**Purpose:** Stores and relates symbolic memories with temporal/contextual awareness

**Location:** `/services/limps`

**Features:**
- Graph-based memory storage
- Temporal decay modeling
- Contextual retrieval
- Symbolic relationship mapping

### 4. **QVNM (Quantum Vector Neural Memory)**
**Purpose:** Intrinsic dimension estimation and manifold analysis

**Location:** `/services/qvnm`

**Features:**
- Costa-Hero ID estimation
- Nearest neighbor queries
- Manifold preview generation
- Vector upload (.npz, .jsonl)

### 5. **AL-ULS Evolution**
**Purpose:** Adaptive learning with universal symbolic constraints

**Location:** `/services/al-uls-evolution`

**Components:**
- Neural-Symbolic Hybrid
- Entropy Optimizer
- Chaos RAG
- Matrix Optimizer
- Symbolic Evolution

**Pipeline:**
```
Training Data → Neural-Symbolic → Entropy Opt →
Chaos RAG → Matrix Opt → Evolved Constraints
```

### 6. **Context Selector**
**Purpose:** Dual context selection with motif-based ranking

**Location:** `/services/context-selector`

**Input:** Candidate text pairs
**Output:** Ranked contexts with entropy scores

### 7. **Neutronics Surrogate**
**Purpose:** Polynomial regression for reactor physics simulations

**Location:** `/services/neutronics`

**Input:** Reactor parameters
**Output:** Flux predictions

## Data Flow Diagrams

### Primary Pipeline: Text → Symbolic Memory

```
┌─────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────┐
│  Text   │────▶│    Motif     │────▶│   Message   │────▶│LiMps │
│  Input  │     │   Detector   │     │ Vectorizer  │     │Memory│
└─────────┘     └──────────────┘     └─────────────┘     └──────┘
                     (Julia)              (Julia)          (Julia)
```

### QVNM Pipeline: Vector Analysis

```
┌─────────┐     ┌──────────────┐     ┌─────────────┐
│ Vectors │────▶│   Intrinsic  │────▶│   Manifold  │
│  .npz   │     │ Dim Estimate │     │   Preview   │
└─────────┘     └──────────────┘     └─────────────┘
                     (Julia)              (Julia)
```

### AL-ULS Pipeline: Adaptive Evolution

```
┌──────────┐   ┌─────────┐   ┌─────────┐   ┌──────────┐
│ Training │──▶│ Neural- │──▶│ Entropy │──▶│  Chaos   │
│   Data   │   │Symbolic │   │   Opt   │   │   RAG    │
└──────────┘   └─────────┘   └─────────┘   └──────────┘
                                                   │
                                                   ▼
┌──────────┐   ┌─────────┐   ┌─────────────────────────┐
│ Evolved  │◀──│ Matrix  │◀──│   Symbolic Evolution    │
│Constrain │   │   Opt   │   │                         │
└──────────┘   └─────────┘   └─────────────────────────┘
```

## Technology Stack

### Backend
- **Julia 1.9+** - Symbolic computation, numerical processing
- **Python 3.9+** - API layer, orchestration, ML training

### Key Libraries (Julia)
- Symbolics.jl - Symbolic mathematics
- HTTP.jl - Web server
- Graphs.jl - Graph operations
- NearestNeighbors.jl - KNN queries

### Key Libraries (Python)
- FastAPI - REST API framework
- PyTorch - Deep learning
- NumPy/SciPy - Numerical computing
- Transformers - NLP models

### Infrastructure
- Docker/Docker Compose - Containerization
- Redis - Caching
- PostgreSQL - Persistence
- Prometheus/Grafana - Monitoring

## Directory Structure

```
eopiez/
├── services/              # Microservices
│   ├── api-gateway/       # Unified API entry point
│   ├── motif-detection/   # Motif detector service
│   ├── message-vectorizer/ # Vectorization service
│   ├── limps/             # Symbolic memory service
│   ├── qvnm/              # Quantum vector neural memory
│   ├── al-uls-evolution/  # AL-ULS adaptive learning
│   ├── context-selector/  # Context selection service
│   └── neutronics/        # Neutronics surrogate
│
├── core/                  # Shared core libraries
│   ├── julia/             # Julia shared modules
│   └── python/            # Python shared modules
│
├── tests/                 # All tests
│   ├── integration/       # End-to-end tests
│   ├── julia/             # Julia unit tests
│   └── python/            # Python unit tests
│
├── docs/                  # Documentation
│   ├── api/               # API documentation
│   ├── guides/            # User guides
│   └── architecture/      # Architecture docs
│
├── examples/              # Usage examples
│   ├── notebooks/         # Jupyter notebooks
│   └── scripts/           # Example scripts
│
├── deployment/            # Deployment configs
│   ├── docker/            # Dockerfiles
│   ├── k8s/               # Kubernetes manifests
│   └── ci-cd/             # CI/CD configs
│
├── docker-compose.yml     # Full system startup
├── ARCHITECTURE.md        # This file
├── GETTING_STARTED.md     # Quick start guide
└── README.md              # Project overview
```

## API Endpoints

### Unified Gateway: `http://localhost:8000`

| Endpoint | Service | Description |
|----------|---------|-------------|
| `POST /api/v1/motif/detect` | Motif Detector | Detect motifs in text |
| `POST /api/v1/vector/encode` | Vectorizer | Encode motifs to vectors |
| `POST /api/v1/limps/store` | LiMps | Store symbolic memory |
| `GET /api/v1/limps/retrieve` | LiMps | Retrieve memories |
| `POST /api/v1/qvnm/upload` | QVNM | Upload vector dataset |
| `GET /api/v1/qvnm/estimate` | QVNM | Estimate intrinsic dimension |
| `POST /api/v1/qvnm/query` | QVNM | Nearest neighbor query |
| `POST /api/v1/al-uls/train` | AL-ULS | Train adaptive model |
| `POST /api/v1/al-uls/infer` | AL-ULS | Run inference |
| `POST /api/v1/context/select` | Context Selector | Select optimal contexts |
| `POST /api/v1/neutronics/predict` | Neutronics | Predict reactor flux |

## Service Communication

All services communicate through the API Gateway using:
- **HTTP/REST** - Request/response patterns
- **WebSocket** - Real-time streaming (planned)
- **Message Queue** - Async processing (planned)

## Data Storage

- **Redis** - Caching, session management
- **PostgreSQL** - Persistent storage for memories, vectors
- **File System** - Vector datasets (.npz), model checkpoints

## Monitoring & Observability

- **Prometheus** - Metrics collection
- **Grafana** - Visualization dashboards
- **Logging** - Structured JSON logs
- **Tracing** - Distributed request tracing (planned)

## Development Workflow

1. **Local Development:** `docker-compose up`
2. **Run Tests:** `make test`
3. **Build Services:** `make build`
4. **Deploy:** `make deploy`

## Deployment Models

### Development
- Docker Compose
- Local Julia/Python processes

### Production
- Kubernetes cluster
- Service mesh (Istio)
- Auto-scaling
- Load balancing

## Security

- API key authentication
- Rate limiting
- Input validation
- HTTPS/TLS encryption

## Performance Considerations

- **Caching:** Redis for frequent queries
- **Batch Processing:** Group similar requests
- **Async Operations:** Non-blocking I/O
- **Resource Limits:** Memory/CPU constraints per service

## Future Roadmap

- [ ] WebSocket support for streaming
- [ ] GraphQL API alternative
- [ ] Enhanced monitoring dashboards
- [ ] Auto-scaling policies
- [ ] Multi-region deployment
- [ ] Advanced security (OAuth2, JWT)
- [ ] ML model versioning
- [ ] A/B testing framework
