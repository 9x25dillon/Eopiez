"""
    Analyzer.jl

Main analysis pipeline orchestrator.
"""

module Analyzer

using ..Types: MotifType, MotifToken, VectorizedMessage, AnalysisResult
using ..Config: get_threshold, get_analysis_config
using ..MotifDetector: detect_motifs, get_dominant_motifs, calculate_motif_statistics
using ..Vectorizer: create_vectorized_message, analyze_vector_properties

"""
    analyze_text(text::String) -> AnalysisResult

Perform complete motif analysis on text.

# Arguments
- `text::String`: Input text to analyze

# Returns
- `AnalysisResult`: Complete analysis results

# Examples
```julia
result = analyze_text("He stood alone in the desert, watching the snake coil.")
```
"""
function analyze_text(text::String)::AnalysisResult
    # Input validation
    if isempty(text)
        return create_empty_result()
    end
    
    try
        # Step 1: Detect motifs
        tokens = detect_motifs(text)
        
        # Step 2: Create vectorized representation
        vectorized_message = create_vectorized_message(tokens, text)
        
        # Step 3: Identify dominant motifs
        dominant_threshold = get_threshold("dominant_motif_threshold")
        dominant_motifs = get_dominant_motifs(tokens, dominant_threshold)
        
        # Step 4: Calculate narrative coherence
        narrative_coherence = vectorized_message.coherence_score
        
        # Step 5: Gather metadata
        metadata = create_analysis_metadata(tokens, vectorized_message, text)
        
        return AnalysisResult(
            tokens,
            vectorized_message,
            dominant_motifs,
            narrative_coherence,
            metadata
        )
        
    catch e
        @warn "Analysis failed" exception=(e, catch_backtrace())
        return create_empty_result()
    end
end

"""
    analyze_text_batch(texts::Vector{String}) -> Vector{AnalysisResult}

Analyze multiple texts in batch.

# Arguments
- `texts::Vector{String}`: Input texts

# Returns
- `Vector{AnalysisResult}`: Analysis results for each text
"""
function analyze_text_batch(texts::Vector{String})::Vector{AnalysisResult}
    results = AnalysisResult[]
    
    for text in texts
        result = analyze_text(text)
        push!(results, result)
    end
    
    return results
end

"""
    compare_analyses(result1::AnalysisResult, result2::AnalysisResult) -> Dict{String, Any}

Compare two analysis results.

# Arguments
- `result1::AnalysisResult`: First analysis result
- `result2::AnalysisResult`: Second analysis result

# Returns
- `Dict{String, Any}`: Comparison metrics
"""
function compare_analyses(result1::AnalysisResult, result2::AnalysisResult)::Dict{String, Any}
    # Vector similarity
    vector_similarity = cosine_similarity(
        result1.vectorized_message.vector,
        result2.vectorized_message.vector
    )
    
    # Coherence comparison
    coherence_diff = abs(result1.narrative_coherence - result2.narrative_coherence)
    
    # Dominant motif overlap
    motif_overlap = length(
        intersect(result1.dominant_motifs, result2.dominant_motifs)
    ) / max(length(result1.dominant_motifs), length(result2.dominant_motifs), 1)
    
    # Token count comparison
    token_count_ratio = length(result1.tokens) / max(length(result2.tokens), 1)
    
    return Dict{String, Any}(
        "vector_similarity" => vector_similarity,
        "coherence_difference" => coherence_diff,
        "motif_overlap" => motif_overlap,
        "token_count_ratio" => token_count_ratio,
        "overall_similarity" => (vector_similarity + (1.0 - coherence_diff) + motif_overlap) / 3.0
    )
end

"""
    create_analysis_metadata(tokens::Vector{MotifToken}, vectorized_message::VectorizedMessage, text::String) -> Dict{String, Any}

Create comprehensive metadata for analysis results.

# Arguments
- `tokens::Vector{MotifToken}`: Detected tokens
- `vectorized_message::VectorizedMessage`: Vectorized representation
- `text::String`: Original text

# Returns
- `Dict{String, Any}`: Analysis metadata
"""
function create_analysis_metadata(
    tokens::Vector{MotifToken},
    vectorized_message::VectorizedMessage,
    text::String
)::Dict{String, Any}
    
    # Basic statistics
    stats = calculate_motif_statistics(tokens)
    
    # Vector properties
    vector_props = analyze_vector_properties(vectorized_message.vector)
    
    # Text properties
    text_props = Dict{String, Any}(
        "length" => length(text),
        "word_count" => length(split(text)),
        "sentence_count" => length(split(text, r"[.!?]+"))
    )
    
    # Analysis configuration
    config = Dict{String, Any}(
        "analysis_timestamp" => time(),
        "version" => "2.0.0"
    )
    
    return Dict{String, Any}(
        "statistics" => stats,
        "vector_properties" => vector_props,
        "text_properties" => text_props,
        "configuration" => config
    )
end

"""
    create_empty_result() -> AnalysisResult

Create an empty analysis result for error cases.

# Returns
- `AnalysisResult`: Empty result
"""
function create_empty_result()::AnalysisResult
    empty_vector = zeros(6)
    empty_message = VectorizedMessage(empty_vector, 0.0, 1.0, 0.0)
    
    return AnalysisResult(
        MotifToken[],
        empty_message,
        MotifType[],
        1.0,
        Dict{String, Any}("error" => "Empty or invalid input")
    )
end

"""
    cosine_similarity(v1::Vector{Float64}, v2::Vector{Float64}) -> Float64

Calculate cosine similarity between two vectors.

# Arguments
- `v1::Vector{Float64}`: First vector
- `v2::Vector{Float64}`: Second vector

# Returns
- `Float64`: Cosine similarity between 0.0 and 1.0
"""
function cosine_similarity(v1::Vector{Float64}, v2::Vector{Float64})::Float64
    if length(v1) != length(v2)
        return 0.0
    end
    
    dot_product = dot(v1, v2)
    norm1 = norm(v1)
    norm2 = norm(v2)
    
    if norm1 == 0 || norm2 == 0
        return 0.0
    end
    
    return dot_product / (norm1 * norm2)
end

export analyze_text, analyze_text_batch, compare_analyses

end # module