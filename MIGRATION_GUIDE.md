# Migration Guide: Eopiez Code Restructuring

This guide helps you transition to the new, improved Eopiez code structure.

## What Changed?

### Overview

The Eopiez codebase has been significantly improved with:
- **Unified Architecture** - Clear separation of concerns
- **Better Documentation** - Comprehensive guides and API docs
- **Improved Developer Experience** - Docker Compose, Makefile, examples
- **Consistent API** - Versioned endpoints with clear routing
- **Enhanced Testing** - Comprehensive test suite

## Key Improvements

### 1. **New API Gateway** (`api_gateway.py`)

**Old Way:**
```python
# Multiple endpoints with inconsistent patterns
POST /dual/select_contexts
POST /qvnm/upload_vectors
POST /qvnm/estimate_id
```

**New Way:**
```python
# Consistent, versioned API
POST /api/v1/context/select
POST /api/v1/qvnm/upload
POST /api/v1/qvnm/estimate/{session_id}
POST /api/v1/motif/detect
POST /api/v1/vector/encode
POST /api/v1/limps/store
GET  /api/v1/limps/retrieve
```

**Migration Steps:**
1. Update your API calls to use the new `/api/v1/*` endpoints
2. Use the new request/response models (see API docs at `/docs`)
3. Legacy endpoints still work for backwards compatibility

### 2. **Docker Compose Setup**

**Old Way:**
- Manual service startup
- Complex dependency management
- Unclear which services are needed

**New Way:**
```bash
# Start everything with one command
docker-compose up

# Or use Makefile
make start
```

**What You Get:**
- ✓ API Gateway on port 8000
- ✓ Julia backend
- ✓ AL-ULS service
- ✓ PostgreSQL database
- ✓ Redis cache
- ✓ Auto-configured networking

### 3. **Makefile Commands**

**New Developer Experience:**
```bash
make help          # See all available commands
make start         # Start services
make test          # Run tests
make logs          # View logs
make notebook      # Start Jupyter
make monitoring    # Start Prometheus + Grafana
```

### 4. **Improved Documentation**

| Document | Purpose |
|----------|---------|
| `README.md` | Project overview |
| `GETTING_STARTED.md` | Quick start guide |
| `ARCHITECTURE.md` | System architecture |
| `MIGRATION_GUIDE.md` | This document |
| `/docs` API | Interactive API docs |

### 5. **Example Scripts**

**New Examples:**
- `examples/motif_detection_example.py` - Motif detection demo
- `examples/full_pipeline_example.py` - End-to-end pipeline
- `examples/qvnm_example.py` - Vector analysis demo

**Run Examples:**
```bash
python examples/motif_detection_example.py
python examples/full_pipeline_example.py
python examples/qvnm_example.py
```

## API Migration

### Context Selection

**Old:**
```python
response = requests.post('http://localhost:8000/dual/select_contexts', json={
    'candidates': [{'id': 'c1', 'text': '...'}, {'id': 'c2', 'text': '...'}],
    'embedding_dim': 64
})
```

**New:**
```python
response = requests.post('http://localhost:8000/api/v1/context/select', json={
    'candidates': [{'id': 'c1', 'text': '...'}, {'id': 'c2', 'text': '...'}],
    'embedding_dim': 64
})
```

### QVNM Upload

**Old:**
```python
files = {'file': open('vectors.npz', 'rb')}
response = requests.post('http://localhost:8000/qvnm/upload_vectors', files=files)
```

**New:**
```python
files = {'file': open('vectors.npz', 'rb')}
response = requests.post('http://localhost:8000/api/v1/qvnm/upload', files=files)
```

### Motif Detection (New!)

**Now Available:**
```python
response = requests.post('http://localhost:8000/api/v1/motif/detect', json={
    'text': 'The soldier stood alone in the wasteland...'
})
```

### Message Vectorization (New!)

**Now Available:**
```python
response = requests.post('http://localhost:8000/api/v1/vector/encode', json={
    'motifs': [...],
    'embedding_dim': 64
})
```

### LiMps Memory (New!)

**Now Available:**
```python
# Store memory
response = requests.post('http://localhost:8000/api/v1/limps/store', json={
    'vector': [...],
    'text': '...',
    'context': {'source': 'user', 'timestamp': '...'}
})

# Retrieve memories
response = requests.get('http://localhost:8000/api/v1/limps/retrieve', params={
    'query': 'memories of war',
    'limit': 5
})
```

## Environment Variables

### Required Environment Variables

Create a `.env` file:
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Backend Services
JULIA_BACKEND_URL=http://julia-backend:8080
AL_ULS_BACKEND_URL=http://al-uls-service:8001

# Database
POSTGRES_PASSWORD=your_secure_password

# Logging
LOG_LEVEL=INFO
```

### Docker Compose Overrides

For custom configurations:
```bash
# Create docker-compose.override.yml
version: '3.8'
services:
  api-gateway:
    environment:
      API_WORKERS: 8
    ports:
      - "8080:8000"
```

## Directory Structure Changes

### Old Structure (Disorganized)
```
├── src/
├── test/
├── tests/
├── limps-aalc/
├── limps-suite-exec-final/
├── al-uls-evolution/
├── file.jl
├── api.py
└── ... (many scattered files)
```

### New Structure (Organized)
```
eopiez/
├── api.py                  # Legacy API (still works)
├── api_gateway.py          # New unified gateway
├── docker-compose.yml      # One-command startup
├── Makefile                # Development commands
├── ARCHITECTURE.md         # Architecture docs
├── GETTING_STARTED.md      # Quick start guide
├── MIGRATION_GUIDE.md      # This file
│
├── deployment/             # Deployment configs
│   ├── docker/             # Dockerfiles
│   └── monitoring/         # Prometheus, Grafana
│
├── examples/               # Usage examples
│   ├── motif_detection_example.py
│   ├── full_pipeline_example.py
│   ├── qvnm_example.py
│   └── notebooks/          # Jupyter notebooks
│
├── src/                    # Julia source code
│   ├── qvnm_server.jl
│   ├── motif_detection/
│   ├── limps/
│   └── quantum_neural/
│
├── al-uls-evolution/       # AL-ULS service
│   └── api/
│
└── tests/                  # All tests
    ├── integration/
    ├── python/
    └── julia/
```

## Breaking Changes

### 1. API Endpoint Paths

All endpoints now use `/api/v1/` prefix. Legacy endpoints still work but are deprecated.

**Action Required:**
- Update client code to use new endpoints
- Update any documentation referencing old endpoints

### 2. Response Formats

Responses are now more consistent with standardized error handling.

**Old Error Response:**
```json
{"error": "missing or invalid sid"}
```

**New Error Response:**
```json
{
  "detail": "Session ID not found",
  "status_code": 404,
  "type": "not_found"
}
```

**Action Required:**
- Update error handling code
- Check for `detail` field instead of `error`

### 3. Configuration

Configuration now uses environment variables instead of hardcoded values.

**Action Required:**
- Create `.env` file with required variables
- Update deployment scripts to set environment variables

## Testing Your Migration

### 1. Check Health

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "services": {
    "julia_backend": "ready",
    "al_uls": "ready"
  },
  "version": "1.0.0"
}
```

### 2. Test Legacy Endpoint

```bash
curl -X POST http://localhost:8000/dual/select_contexts \
  -H "Content-Type: application/json" \
  -d '{"candidates": [{"id": "c1", "text": "test"}], "embedding_dim": 64}'
```

### 3. Test New Endpoint

```bash
curl -X POST http://localhost:8000/api/v1/context/select \
  -H "Content-Type: application/json" \
  -d '{"candidates": [{"id": "c1", "text": "test"}], "embedding_dim": 64}'
```

### 4. Run Example Scripts

```bash
python examples/motif_detection_example.py
python examples/full_pipeline_example.py
```

## Rollback Plan

If you encounter issues:

### Option 1: Use Legacy API

The old `api.py` still works:
```bash
# In docker-compose.yml, use old API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Option 2: Git Revert

```bash
git log --oneline  # Find commit before migration
git revert <commit-hash>
```

## Common Issues & Solutions

### Issue: Port 8000 already in use

**Solution:**
```bash
# Find and kill process
lsof -i :8000
kill -9 <PID>

# Or change port in .env
API_PORT=8001
```

### Issue: Julia dependencies not installing

**Solution:**
```bash
# Rebuild Julia container
docker-compose build --no-cache julia-backend
```

### Issue: Services not starting

**Solution:**
```bash
# Check logs
docker-compose logs -f

# Restart services
docker-compose restart
```

### Issue: Can't connect to backend services

**Solution:**
```bash
# Check network
docker network ls
docker network inspect eopiez-network

# Verify service URLs
echo $JULIA_BACKEND_URL
echo $AL_ULS_BACKEND_URL
```

## Timeline

- **✅ Completed:** Core restructuring, documentation, Docker setup
- **🚧 In Progress:** Test suite migration, monitoring setup
- **📅 Planned:** WebSocket support, GraphQL API, enhanced security

## Getting Help

- **Documentation:** See `GETTING_STARTED.md` and `ARCHITECTURE.md`
- **API Reference:** http://localhost:8000/docs
- **Issues:** https://github.com/9x25dillon/Eopiez/issues
- **Examples:** Check `examples/` directory

## Feedback

Please report issues or suggestions:
- GitHub Issues: https://github.com/9x25dillon/Eopiez/issues
- Pull Requests: https://github.com/9x25dillon/Eopiez/pulls

## Next Steps

1. ✅ Read this migration guide
2. ✅ Review `GETTING_STARTED.md`
3. ✅ Start services: `make start`
4. ✅ Test health: `curl http://localhost:8000/health`
5. ✅ Run examples: `python examples/motif_detection_example.py`
6. ✅ Explore API docs: http://localhost:8000/docs
7. ✅ Update your code to use new endpoints
8. ✅ Test thoroughly
9. ✅ Provide feedback

Welcome to the new and improved Eopiez! 🚀
