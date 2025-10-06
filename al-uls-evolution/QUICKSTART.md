# 🚀 AL-ULS Evolution - Quick Start Guide

## Get Running in 5 Minutes

### Step 1: Start the System
```bash
cd al-uls-evolution
docker-compose up --build
```

Wait for:
```
✅ API server ready at http://localhost:8000
✅ Julia symbolic server ready at http://localhost:8088
```

### Step 2: Initialize
```bash
curl -X POST http://localhost:8000/system/initialize \
  -H "Content-Type: application/json" \
  -d '{
    "input_dim": 64,
    "hidden_dims": [128, 256, 128],
    "output_dim": 10,
    "entropy_target": 3.5,
    "use_chaos_rag": true,
    "use_symbolic_reasoning": true
  }'
```

### Step 3: Run Demo
```bash
docker exec -it al-uls-evolution_api_1 python examples/training_demo.py
```

### Step 4: Monitor
```bash
# Watch system evolve
watch -n 1 'curl -s http://localhost:8000/symbolic/constraints | jq ".top_constraints[:3]"'

# View entropy stats
curl http://localhost:8000/entropy/stats | jq

# See discovered matrix structures
curl http://localhost:8000/matrix/structures | jq
```

## What You'll See

### Self-Evolving Constraints
```json
{
  "expression": "H(s) → 3.50",
  "strength": 0.85,
  "confidence": 0.92
}
```

### Entropy-Guided Learning
```json
{
  "learning_rate": 0.001234,
  "total_entropy": 3.47,
  "chaos_signal": 0.23
}
```

### Discovered Structures
```json
{
  "layer.weight": {
    "type": "low_rank",
    "compression": 0.35,
    "expression": "rank-12 approximation"
  }
}
```

## Next Steps

- 📖 Read **README.md** for complete documentation
- 📊 See **SUMMARY.md** for implementation details
- 🔬 Explore **examples/training_demo.py** for advanced usage
- 🌐 Visit `http://localhost:8000/docs` for API documentation

## Troubleshooting

**Julia server not connecting?**
```bash
docker logs al-uls-evolution_julia_1
```

**API errors?**
```bash
docker logs al-uls-evolution_api_1
```

**Need to rebuild?**
```bash
docker-compose down -v
docker-compose up --build
```

## 🎉 You're Ready!

The AL-ULS Evolution system is now:
- ✅ Self-evolving its learning constraints
- ✅ Using hybrid neural-symbolic reasoning
- ✅ Adapting learning rates via entropy
- ✅ Exploring knowledge with chaos
- ✅ Discovering mathematical structures

**Welcome to emergent AI!** 🌟
