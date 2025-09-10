"""
    Vectorizer.jl

Vectorization and numerical analysis of motif tokens.
"""

module Vectorizer

using LinearAlgebra
using Statistics
using ..Types: MotifType, MotifToken, VectorizedMessage, motif_type_to_index
using ..Config: get_vector_config, get_analysis_config

"""
    vectorize_motifs(tokens::Vector{MotifToken}) -> Vector{Float64}

Convert motif tokens to a numerical vector representation.

# Arguments
- `tokens::Vector{MotifToken}`: Detected motif tokens

# Returns
- `Vector{Float64}`: Numerical vector representation

# Examples
```julia
vector = vectorize_motifs(tokens)
```
"""
function vectorize_motifs(tokens::Vector{MotifToken})::Vector{Float64}
    dimensions = get_vector_config("dimensions")
    vector = zeros(dimensions)
    
    if isempty(tokens)
        return vector
    end
    
    # Aggregate weights by motif type
    for token in tokens
        idx = motif_type_to_index(token.type)
        if 1 ≤ idx ≤ dimensions
            vector[idx] += token.weight * token.confidence
        end
    end
    
    # Normalize vector
    norm_val = norm(vector)
    if norm_val > 0
        vector ./= norm_val
    end
    
    return vector
end

"""
    calculate_entropy(vector::Vector{Float64}, base::Float64=2.0) -> Float64

Calculate information entropy of a vector.

# Arguments
- `vector::Vector{Float64}`: Input vector
- `base::Float64`: Logarithm base for entropy calculation

# Returns
- `Float64`: Entropy value
"""
function calculate_entropy(vector::Vector{Float64}, base::Float64=2.0)::Float64
    # Filter non-zero elements
    non_zero = vector[vector .> 0]
    
    if isempty(non_zero)
        return 0.0
    end
    
    # Normalize to probabilities
    total = sum(non_zero)
    if total == 0
        return 0.0
    end
    
    probabilities = non_zero / total
    
    # Calculate entropy
    entropy = 0.0
    for p in probabilities
        if p > 0
            entropy -= p * log(base, p)
        end
    end
    
    return entropy
end

"""
    calculate_coherence_score(tokens::Vector{MotifToken}) -> Float64

Calculate narrative coherence score based on motif distribution.

# Arguments
- `tokens::Vector{MotifToken}`: Detected motif tokens

# Returns
- `Float64`: Coherence score between 0.0 and 1.0
"""
function calculate_coherence_score(tokens::Vector{MotifToken})::Float64
    if length(tokens) < 2
        return 1.0
    end
    
    # Calculate position-based coherence
    positions = [token.position for token in tokens]
    weights = [token.weight for token in tokens]
    
    # Sort by position
    sorted_indices = sortperm(positions)
    sorted_weights = weights[sorted_indices]
    
    # Calculate smoothness of weight transitions
    if length(sorted_weights) > 1
        weight_diffs = abs.(diff(sorted_weights))
        smoothness = 1.0 - mean(weight_diffs)
        smoothness = max(0.0, min(1.0, smoothness))
    else
        smoothness = 1.0
    end
    
    # Calculate type diversity penalty
    unique_types = length(unique([token.type for token in tokens]))
    total_types = length(instances(MotifType))
    diversity_penalty = unique_types / total_types
    
    # Final coherence score
    coherence = smoothness * (1.0 - 0.5 * diversity_penalty)
    return max(0.0, min(1.0, coherence))
end

"""
    calculate_motif_density(tokens::Vector{MotifToken}, text_length::Int) -> Float64

Calculate the density of motifs in the text.

# Arguments
- `tokens::Vector{MotifToken}`: Detected motif tokens
- `text_length::Int`: Length of original text

# Returns
- `Float64`: Motif density between 0.0 and 1.0
"""
function calculate_motif_density(tokens::Vector{MotifToken}, text_length::Int)::Float64
    if text_length == 0 || isempty(tokens)
        return 0.0
    end
    
    # Count total motif matches
    total_matches = sum(length(token.matches) for token in tokens)
    
    # Normalize by text length
    density = total_matches / text_length
    return min(1.0, density)
end

"""
    create_vectorized_message(tokens::Vector{MotifToken}, text::String) -> VectorizedMessage

Create a complete vectorized message from motif tokens.

# Arguments
- `tokens::Vector{MotifToken}`: Detected motif tokens
- `text::String`: Original text

# Returns
- `VectorizedMessage`: Complete vectorized representation
"""
function create_vectorized_message(tokens::Vector{MotifToken}, text::String)::VectorizedMessage
    # Vectorize motifs
    vector = vectorize_motifs(tokens)
    
    # Calculate metrics
    entropy = calculate_entropy(vector)
    coherence_score = calculate_coherence_score(tokens)
    motif_density = calculate_motif_density(tokens, length(text))
    
    return VectorizedMessage(vector, entropy, coherence_score, motif_density)
end

"""
    analyze_vector_properties(vector::Vector{Float64}) -> Dict{String, Any}

Analyze properties of a vectorized representation.

# Arguments
- `vector::Vector{Float64}`: Input vector

# Returns
- `Dict{String, Any}`: Vector properties
"""
function analyze_vector_properties(vector::Vector{Float64})::Dict{String, Any}
    return Dict{String, Any}(
        "norm" => norm(vector),
        "entropy" => calculate_entropy(vector),
        "sparsity" => count(abs.(vector) .< 1e-6) / length(vector),
        "max_component" => maximum(vector),
        "min_component" => minimum(vector),
        "mean_component" => mean(vector),
        "std_component" => std(vector)
    )
end

export vectorize_motifs, calculate_entropy, calculate_coherence_score
export calculate_motif_density, create_vectorized_message, analyze_vector_properties

end # module