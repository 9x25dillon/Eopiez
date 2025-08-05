"""
    MotifAnalysis.jl

Main module for motif detection and analysis system.

This module provides a clean API for analyzing text and detecting Kojima-esque motifs
with advanced vectorization and coherence analysis capabilities.
"""

module MotifAnalysis

# Import all submodules
include("Types.jl")
include("Config.jl")
include("TextProcessing.jl")
include("MotifDetector.jl")
include("Vectorizer.jl")
include("Analyzer.jl")

using .Types
using .Config
using .TextProcessing
using .MotifDetector
using .Vectorizer
using .Analyzer

# Re-export main functionality
export MotifType, MotifToken, VectorizedMessage, AnalysisResult
export analyze_text, analyze_text_batch, compare_analyses
export detect_motifs, detect_motifs_by_type, get_dominant_motifs
export vectorize_motifs, calculate_entropy, calculate_coherence_score
export preprocess_text, extract_sentences

"""
    version() -> String

Get the current version of the MotifAnalysis system.

# Returns
- `String`: Version string
"""
function version()::String
    return "2.0.0"
end

"""
    supported_motif_types() -> Vector{MotifType}

Get all supported motif types.

# Returns
- `Vector{MotifType}`: All available motif types
"""
function supported_motif_types()::Vector{MotifType}
    return collect(instances(MotifType))
end

"""
    get_motif_patterns(motif_type::MotifType) -> Vector{String}

Get the text patterns associated with a motif type.

# Arguments
- `motif_type::MotifType`: The motif type

# Returns
- `Vector{String}`: Associated text patterns
"""
function get_motif_patterns(motif_type::MotifType)::Vector{String}
    return Config.get_motif_patterns(motif_type)
end

"""
    quick_analysis(text::String) -> Dict{String, Any}

Perform a quick analysis and return simplified results.

# Arguments
- `text::String`: Input text

# Returns
- `Dict{String, Any}`: Simplified analysis results
"""
function quick_analysis(text::String)::Dict{String, Any}
    result = analyze_text(text)
    
    return Dict{String, Any}(
        "dominant_motifs" => [string(m) for m in result.dominant_motifs],
        "coherence_score" => result.narrative_coherence,
        "motif_count" => length(result.tokens),
        "entropy" => result.vectorized_message.entropy,
        "motif_density" => result.vectorized_message.motif_density
    )
end

"""
    analyze_motif_distribution(texts::Vector{String}) -> Dict{String, Any}

Analyze motif distribution across multiple texts.

# Arguments
- `texts::Vector{String}`: Input texts

# Returns
- `Dict{String, Any}`: Distribution analysis
"""
function analyze_motif_distribution(texts::Vector{String})::Dict{String, Any}
    results = analyze_text_batch(texts)
    
    # Aggregate motif types across all texts
    all_motif_types = Dict{MotifType, Int}()
    total_coherence = 0.0
    total_entropy = 0.0
    
    for result in results
        for motif_type in result.dominant_motifs
            all_motif_types[motif_type] = get(all_motif_types, motif_type, 0) + 1
        end
        total_coherence += result.narrative_coherence
        total_entropy += result.vectorized_message.entropy
    end
    
    # Calculate averages
    n_texts = length(texts)
    avg_coherence = n_texts > 0 ? total_coherence / n_texts : 0.0
    avg_entropy = n_texts > 0 ? total_entropy / n_texts : 0.0
    
    # Sort motif types by frequency
    sorted_motifs = sort(collect(all_motif_types), by=x->x[2], rev=true)
    
    return Dict{String, Any}(
        "total_texts" => n_texts,
        "motif_frequency" => Dict(string(m) => c for (m, c) in sorted_motifs),
        "average_coherence" => avg_coherence,
        "average_entropy" => avg_entropy,
        "most_common_motif" => isempty(sorted_motifs) ? nothing : string(sorted_motifs[1][1])
    )
end

# Module initialization
function __init__()
    @info "MotifAnalysis v$(version()) initialized"
    @info "Supported motif types: $(join([string(t) for t in supported_motif_types()], ", "))"
end

end # module