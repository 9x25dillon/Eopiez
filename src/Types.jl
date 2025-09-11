"""
    Types.jl

Core type definitions for the motif detection and analysis system.
"""

module Types

using LinearAlgebra

"""
    MotifType

Enumeration of supported motif types for narrative analysis.
"""
@enum MotifType begin
    ISOLATION
    SNAKE
    STRAND
    MEMORY
    TEMPORAL
    FRAGMENTATION
end

"""
    MotifToken

Represents a detected motif in text with associated metadata.

# Fields
- `type::MotifType`: The type of motif detected
- `weight::Float64`: Normalized weight/importance of the motif (0.0 to 1.0)
- `matches::Vector{String}`: Original text matches that triggered this motif
- `position::Float64`: Relative position in the text (0.0 to 1.0)
- `context::String`: Surrounding context for the motif
- `confidence::Float64`: Detection confidence score (0.0 to 1.0)
"""
struct MotifToken
    type::MotifType
    weight::Float64
    matches::Vector{String}
    position::Float64
    context::String
    confidence::Float64
    
    function MotifToken(
        type::MotifType,
        weight::Float64,
        matches::Vector{String},
        position::Float64,
        context::String,
        confidence::Float64
    )
        @assert 0.0 ≤ weight ≤ 1.0 "Weight must be between 0.0 and 1.0"
        @assert 0.0 ≤ position ≤ 1.0 "Position must be between 0.0 and 1.0"
        @assert 0.0 ≤ confidence ≤ 1.0 "Confidence must be between 0.0 and 1.0"
        new(type, weight, matches, position, context, confidence)
    end
end

"""
    VectorizedMessage

Represents a message converted to numerical vector form for analysis.

# Fields
- `vector::Vector{Float64}`: Numerical representation of the message
- `entropy::Float64`: Information entropy of the vector
- `coherence_score::Float64`: Narrative coherence score
- `motif_density::Float64`: Density of motifs in the text
"""
struct VectorizedMessage
    vector::Vector{Float64}
    entropy::Float64
    coherence_score::Float64
    motif_density::Float64
    
    function VectorizedMessage(
        vector::Vector{Float64},
        entropy::Float64,
        coherence_score::Float64,
        motif_density::Float64
    )
        @assert entropy ≥ 0.0 "Entropy must be non-negative"
        @assert 0.0 ≤ coherence_score ≤ 1.0 "Coherence score must be between 0.0 and 1.0"
        @assert 0.0 ≤ motif_density ≤ 1.0 "Motif density must be between 0.0 and 1.0"
        new(vector, entropy, coherence_score, motif_density)
    end
end

"""
    AnalysisResult

Container for complete motif analysis results.

# Fields
- `tokens::Vector{MotifToken}`: Detected motif tokens
- `vectorized_message::VectorizedMessage`: Vectorized representation
- `dominant_motifs::Vector{MotifType}`: Most prominent motif types
- `narrative_coherence::Float64`: Overall narrative coherence score
- `metadata::Dict{String, Any}`: Additional analysis metadata
"""
struct AnalysisResult
    tokens::Vector{MotifToken}
    vectorized_message::VectorizedMessage
    dominant_motifs::Vector{MotifType}
    narrative_coherence::Float64
    metadata::Dict{String, Any}
    
    function AnalysisResult(
        tokens::Vector{MotifToken},
        vectorized_message::VectorizedMessage,
        dominant_motifs::Vector{MotifType},
        narrative_coherence::Float64,
        metadata::Dict{String, Any}
    )
        @assert 0.0 ≤ narrative_coherence ≤ 1.0 "Narrative coherence must be between 0.0 and 1.0"
        new(tokens, vectorized_message, dominant_motifs, narrative_coherence, metadata)
    end
end

# Utility functions
Base.string(motif::MotifType) = String(motif)
Base.show(io::IO, motif::MotifType) = print(io, string(motif))

"""
    motif_type_to_index(motif::MotifType) -> Int

Convert motif type to vector index for numerical representation.
"""
function motif_type_to_index(motif::MotifType)::Int
    return Int(motif) + 1
end

"""
    index_to_motif_type(index::Int) -> MotifType

Convert vector index to motif type.
"""
function index_to_motif_type(index::Int)::MotifType
    @assert 1 ≤ index ≤ 6 "Index must be between 1 and 6"
    return MotifType(index - 1)
end

export MotifType, MotifToken, VectorizedMessage, AnalysisResult
export motif_type_to_index, index_to_motif_type

end # module