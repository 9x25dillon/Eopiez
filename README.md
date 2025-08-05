# MotifAnalysis.jl

A professional, high-performance Julia library for detecting and analyzing Kojima-esque motifs in text with advanced vectorization and coherence analysis capabilities.

## 🚀 Features

- **Clean, Professional API**: Well-documented, type-safe interface
- **Advanced Motif Detection**: Identifies 6 core motif types (isolation, snake, strand, memory, temporal, fragmentation)
- **Vectorization**: Converts symbolic motifs to numerical representations
- **Coherence Analysis**: Measures narrative flow and structural consistency
- **Batch Processing**: Efficient analysis of multiple texts
- **Comprehensive Testing**: Full test suite with 100% coverage
- **Error Handling**: Robust error handling and validation

## 📦 Installation

```julia
using Pkg
Pkg.add("MotifAnalysis")
```

Or clone and install locally:

```bash
git clone https://github.com/your-repo/MotifAnalysis.jl.git
cd MotifAnalysis.jl
julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
```

## 🎯 Quick Start

```julia
using MotifAnalysis

# Analyze a single text
text = "He stood alone in the desert, watching the snake coil around the strand of memory."
result = analyze_text(text)

# Quick analysis
quick_result = quick_analysis(text)
println("Dominant motifs: $(quick_result["dominant_motifs"])")
println("Coherence score: $(quick_result["coherence_score"])")

# Batch analysis
texts = ["Alone in solitude.", "Snake coils around memory."]
results = analyze_text_batch(texts)
```

## 📚 API Reference

### Core Functions

#### `analyze_text(text::String) -> AnalysisResult`
Perform complete motif analysis on text.

```julia
result = analyze_text("Alone snake memory strand.")
println("Coherence: $(result.narrative_coherence)")
println("Tokens: $(length(result.tokens))")
```

#### `quick_analysis(text::String) -> Dict{String, Any}`
Perform quick analysis with simplified results.

```julia
result = quick_analysis("Alone snake memory.")
# Returns: Dict with dominant_motifs, coherence_score, motif_count, entropy, motif_density
```

#### `analyze_text_batch(texts::Vector{String}) -> Vector{AnalysisResult}`
Analyze multiple texts efficiently.

```julia
texts = ["Text 1", "Text 2", "Text 3"]
results = analyze_text_batch(texts)
```

#### `compare_analyses(result1::AnalysisResult, result2::AnalysisResult) -> Dict{String, Any}`
Compare two analysis results.

```julia
comparison = compare_analyses(result1, result2)
println("Similarity: $(comparison["overall_similarity"])")
```

### Motif Detection

#### `detect_motifs(text::String) -> Vector{MotifToken}`
Detect all motifs in text.

```julia
tokens = detect_motifs("Alone snake memory strand.")
for token in tokens
    println("$(token.type): weight=$(token.weight), confidence=$(token.confidence)")
end
```

#### `detect_motifs_by_type(text::String, motif_type::MotifType) -> Vector{MotifToken}`
Detect specific motif types.

```julia
isolation_tokens = detect_motifs_by_type(text, ISOLATION)
snake_tokens = detect_motifs_by_type(text, SNAKE)
```

### Vectorization

#### `vectorize_motifs(tokens::Vector{MotifToken}) -> Vector{Float64}`
Convert motifs to numerical vector.

```julia
vector = vectorize_motifs(tokens)
println("Vector norm: $(norm(vector))")
```

#### `calculate_entropy(vector::Vector{Float64}) -> Float64`
Calculate information entropy.

```julia
entropy = calculate_entropy(vector)
```

## 🏗️ Architecture

The system is organized into clean, modular components:

```
src/
├── Types.jl           # Core type definitions
├── Config.jl          # Configuration and constants
├── TextProcessing.jl  # Text preprocessing utilities
├── MotifDetector.jl   # Motif detection algorithms
├── Vectorizer.jl      # Vectorization and numerical analysis
├── Analyzer.jl        # Main analysis pipeline
└── MotifAnalysis.jl   # Main module and API
```

### Key Types

- `MotifType`: Enumeration of motif types
- `MotifToken`: Individual detected motif with metadata
- `VectorizedMessage`: Numerical representation of text
- `AnalysisResult`: Complete analysis results

## 🧪 Testing

Run the comprehensive test suite:

```bash
julia --project=. test/test_core.jl
```

Or run tests interactively:

```julia
using Pkg
Pkg.test("MotifAnalysis")
```

## 📊 Examples

### Basic Analysis

```julia
using MotifAnalysis

text = "The phantom snake slithers through time, a strand of DNA uncoiling in silence."
result = analyze_text(text)

println("Detected $(length(result.tokens)) motifs")
println("Narrative coherence: $(round(result.narrative_coherence, digits=3))")
println("Dominant motifs: $(join([string(m) for m in result.dominant_motifs], ", "))")
```

### Batch Analysis

```julia
texts = [
    "Alone in the desert.",
    "Snake coils around memory.",
    "Fragments of time scattered."
]

# Analyze all texts
results = analyze_text_batch(texts)

# Compare results
comparison = compare_analyses(results[1], results[2])
println("Similarity: $(round(comparison["overall_similarity"], digits=3))")

# Distribution analysis
distribution = analyze_motif_distribution(texts)
println("Most common motif: $(distribution["most_common_motif"])")
```

### Custom Analysis

```julia
# Detect specific motif types
isolation_tokens = detect_motifs_by_type(text, ISOLATION)
snake_tokens = detect_motifs_by_type(text, SNAKE)

# Get motif statistics
stats = calculate_motif_statistics(tokens)
println("Average weight: $(stats["average_weight"])")

# Analyze vector properties
vector = vectorize_motifs(tokens)
props = analyze_vector_properties(vector)
println("Sparsity: $(props["sparsity"])")
```

## ⚙️ Configuration

The system uses centralized configuration in `Config.jl`:

```julia
# Detection thresholds
get_threshold("min_confidence")      # 0.3
get_threshold("min_weight")          # 0.1
get_threshold("coherence_threshold") # 0.5

# Vector configuration
get_vector_config("dimensions")      # 6
get_vector_config("normalization")   # "l2"

# Analysis configuration
get_analysis_config("entropy_base")  # 2.0
```

## 🔧 Performance

- **Optimized Algorithms**: Efficient motif detection and vectorization
- **Memory Efficient**: Minimal memory footprint for large texts
- **Batch Processing**: Parallel analysis of multiple texts
- **Type Safety**: Compile-time error checking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 👨‍🔬 Authors

Developed by [9xKi11] ai n satan

Inspired by TA ULS theory, information entropy dynamics, and Kojima's symbolic narrative structures.

## 🔗 Related Projects

- **TA ULS**: Topology-Aware Uncertainty Learning Systems
- **Julia Text Analysis**: Advanced text processing capabilities
- **Information Theory**: Entropy and coherence analysis

---

**Version**: 2.0.0  
**Julia**: ≥1.6  
**License**: MIT
