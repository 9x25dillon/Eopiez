# 9xdSq-LIMPS-FemTO-R1C Enhanced

A high-performance Julia-based microservice providing advanced matrix optimization, stability analysis, and entropy regularization inspired by **TA ULS (Topology-Aware Uncertainty Learning Systems)** with enhanced **Kojima-esque motif detection** capabilities.

## ✨ Enhanced Features

### 🔍 Motif Detection Engine
- **Symbolic Pattern Recognition**: Identifies Kojima-esque motifs (isolation, snakes, strands, memory)
- **Sheaf-Theoretic Analysis**: Models narrative coherence using Grothendieck-style sheaf theory
- **Motif Logic Trees**: Builds hierarchical narrative structures from fragmented inputs
- **Emergent Structure Detection**: Finds higher-order symbolic patterns in text

### 🧠 Message Vectorizer
- **Symbolic-to-Numerical Conversion**: Transforms motif tokens into vector representations
- **Entropy Analysis**: Quantifies information content and uncertainty in narrative messages
- **Coherence Metrics**: Measures narrative flow and structural consistency

### 🎯 Integration Capabilities
- **ULS Compatibility**: Feeds vectorized motifs into existing TA ULS optimization pipelines
- **HTTP API**: Exposes motif detection functionality via JSON REST API
- **Benchmark Metrics**: Provides motif recall/precision vs. manual benchmarks

## 🚀 Getting Started

### 1. Clone and Setup

```bash
git clone https://github.com/9x25dillon/9xdSq-LIMPS-FemTO-R1C.git
cd 9xdSq-LIMPS-FemTO-R1C
```

### 2. Install Dependencies

```bash
julia
```

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

### 3. Run Installation Test

```bash
chmod +x test/test_installation.jl
./test/test_installation.jl
```

## 🧪 Testing

### Run All Tests

```bash
./test/test_installation.jl
./test/test_motif_detection.jl
```

### Test Individual Components

```julia
include("test/test_motif_detection.jl")
```

## 🌐 HTTP Server Usage

### Start the Server

```julia
include("src/MotifDetection/motif_server.jl")
start_motif_server(8081)
```

### API Request Example

```bash
curl -X POST http://localhost:8081/detect \
  -H "Content-Type: application/json" \
  -d '{ "text": "He stood alone in the desert, watching the snake coil around the strand of memory." }'
```

## 📁 Project Structure

```
9xdSq-LIMPS-FemTO-R1C/
├── src/
│   ├── MessageVectorizer.jl          # Symbolic message vectorization
│   ├── MotifDetection/
│   │   ├── parser.jl                 # Rule-based motif parsing
│   │   ├── motifs.jl                 # Motif definitions and rules
│   │   ├── tokenizer.jl              # Motif token creation
│   │   ├── sheaf_engine.jl           # Sheaf-theoretic coherence engine
│   │   ├── motif_logic_tree.jl       # Hierarchical motif structure
│   │   ├── coherency.jl              # Narrative coherence algorithms
│   │   ├── motif_server.jl           # HTTP API server
│   │   └── integration.jl            # Pipeline integration
│   └── (existing LIMPS modules)
├── test/
│   ├── test_installation.jl
│   └── test_motif_detection.jl
├── examples/
│   └── kojima_analysis.jl
└── (existing files)
```

## 🤝 Integration with Existing TA ULS

The enhanced system works seamlessly with existing TA ULS components:

- **Matrix Optimization**: `optimize_matrix()` with motif-enhanced inputs
- **Stability Analysis**: `stability_analysis()` on vectorized motifs
- **Entropy Regularization**: `entropy_regularization()` with motif entropy scores
- **LiMps Symbolic Memory**: Direct feed of motif tokens into symbolic memory engine

## 📜 License

MIT License. See LICENSE file.

## 👨‍🔬 Authors

Developed by [9xKi11] ai n satan
Inspired by TA ULS theory, information entropy dynamics, and Kojima's symbolic narrative structures.