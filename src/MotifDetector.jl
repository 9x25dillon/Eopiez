"""
    MotifDetector.jl

Core motif detection algorithms and token generation.
"""

module MotifDetector

using Statistics
using ..Types: MotifType, MotifToken, ISOLATION, SNAKE, STRAND, MEMORY, TEMPORAL, FRAGMENTATION
using ..Config: get_threshold, get_motif_patterns
using ..TextProcessing: preprocess_text, find_motif_matches, calculate_position_confidence, extract_context

"""
    detect_motifs(text::String) -> Vector{MotifToken}

Detect all motifs in the given text.

# Arguments
- `text::String`: Input text to analyze

# Returns
- `Vector{MotifToken}`: Detected motif tokens

# Examples
```julia
tokens = detect_motifs("He stood alone in the desert, watching the snake coil.")
```
"""
function detect_motifs(text::String)::Vector{MotifToken}
    isempty(text) && return MotifToken[]
    
    # Preprocess text
    tokens = preprocess_text(text)
    isempty(tokens) && return MotifToken[]
    
    # Detect motifs for each type
    motif_tokens = MotifToken[]
    min_confidence = get_threshold("min_confidence")
    min_weight = get_threshold("min_weight")
    
    for motif_type in instances(MotifType)
        matches = find_motif_matches(tokens, motif_type)
        
        if !isempty(matches)
            # Calculate weight and confidence
            weight = length(matches) / length(tokens)
            position, confidence = calculate_position_confidence(tokens, matches)
            
            # Filter by thresholds
            if weight >= min_weight && confidence >= min_confidence
                context = extract_context(text, position)
                
                token = MotifToken(
                    motif_type,
                    weight,
                    matches,
                    position,
                    context,
                    confidence
                )
                push!(motif_tokens, token)
            end
        end
    end
    
    return motif_tokens
end

"""
    detect_motifs_by_type(text::String, motif_type::MotifType) -> Vector{MotifToken}

Detect motifs of a specific type in the text.

# Arguments
- `text::String`: Input text
- `motif_type::MotifType`: Specific motif type to detect

# Returns
- `Vector{MotifToken}`: Detected tokens of the specified type
"""
function detect_motifs_by_type(text::String, motif_type::MotifType)::Vector{MotifToken}
    all_tokens = detect_motifs(text)
    return filter(token -> token.type == motif_type, all_tokens)
end

"""
    get_dominant_motifs(tokens::Vector{MotifToken}, threshold::Float64=0.6) -> Vector{MotifType}

Identify the most dominant motif types in the text.

# Arguments
- `tokens::Vector{MotifToken}`: Detected motif tokens
- `threshold::Float64`: Weight threshold for dominance

# Returns
- `Vector{MotifType}`: Dominant motif types
"""
function get_dominant_motifs(tokens::Vector{MotifToken}, threshold::Float64=0.6)::Vector{MotifType}
    if isempty(tokens)
        return MotifType[]
    end
    
    # Group tokens by type and calculate average weights
    type_weights = Dict{MotifType, Float64}()
    type_counts = Dict{MotifType, Int}()
    
    for token in tokens
        type_weights[token.type] = get(type_weights, token.type, 0.0) + token.weight
        type_counts[token.type] = get(type_counts, token.type, 0) + 1
    end
    
    # Calculate average weights
    for motif_type in keys(type_weights)
        type_weights[motif_type] /= type_counts[motif_type]
    end
    
    # Find dominant motifs
    dominant_motifs = MotifType[]
    for (motif_type, avg_weight) in type_weights
        if avg_weight >= threshold
            push!(dominant_motifs, motif_type)
        end
    end
    
    return dominant_motifs
end

"""
    calculate_motif_statistics(tokens::Vector{MotifToken}) -> Dict{String, Any}

Calculate comprehensive statistics for detected motifs.

# Arguments
- `tokens::Vector{MotifToken}`: Detected motif tokens

# Returns
- `Dict{String, Any}`: Statistical summary
"""
function calculate_motif_statistics(tokens::Vector{MotifToken})::Dict{String, Any}
    if isempty(tokens)
        return Dict{String, Any}(
            "total_tokens" => 0,
            "unique_types" => 0,
            "average_weight" => 0.0,
            "average_confidence" => 0.0,
            "type_distribution" => Dict{String, Int}(),
            "weight_distribution" => Dict{String, Float64}()
        )
    end
    
    # Basic statistics
    total_tokens = length(tokens)
    unique_types = length(unique([token.type for token in tokens]))
    average_weight = mean([token.weight for token in tokens])
    average_confidence = mean([token.confidence for token in tokens])
    
    # Type distribution
    type_counts = Dict{String, Int}()
    type_weights = Dict{String, Float64}()
    
    for token in tokens
        type_str = string(token.type)
        type_counts[type_str] = get(type_counts, type_str, 0) + 1
        type_weights[type_str] = get(type_weights, type_str, 0.0) + token.weight
    end
    
    # Normalize weights
    for type_str in keys(type_weights)
        type_weights[type_str] /= type_counts[type_str]
    end
    
    return Dict{String, Any}(
        "total_tokens" => total_tokens,
        "unique_types" => unique_types,
        "average_weight" => average_weight,
        "average_confidence" => average_confidence,
        "type_distribution" => type_counts,
        "weight_distribution" => type_weights
    )
end

"""
    validate_motif_token(token::MotifToken) -> Bool

Validate a motif token for consistency and correctness.

# Arguments
- `token::MotifToken`: Token to validate

# Returns
- `Bool`: True if token is valid
"""
function validate_motif_token(token::MotifToken)::Bool
    # Check basic constraints
    if token.weight < 0.0 || token.weight > 1.0
        return false
    end
    
    if token.position < 0.0 || token.position > 1.0
        return false
    end
    
    if token.confidence < 0.0 || token.confidence > 1.0
        return false
    end
    
    if isempty(token.matches)
        return false
    end
    
    return true
end

export detect_motifs, detect_motifs_by_type, get_dominant_motifs
export calculate_motif_statistics, validate_motif_token

end # module