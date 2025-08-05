"""
    TextProcessing.jl

Text preprocessing and normalization utilities.
"""

module TextProcessing

using ..Types: MotifType, MotifToken
using ..Config: get_motif_patterns, get_threshold

"""
    preprocess_text(text::String) -> Vector{String}

Preprocess and tokenize text for motif detection.

# Arguments
- `text::String`: Input text to process

# Returns
- `Vector{String}`: Preprocessed tokens

# Examples
```julia
tokens = preprocess_text("He stood alone in the desert.")
```
"""
function preprocess_text(text::String)::Vector{String}
    isempty(text) && return String[]
    
    # Convert to lowercase and normalize whitespace
    text = lowercase(strip(text))
    text = replace(text, r"\s+" => " ")
    
    # Remove punctuation except for sentence boundaries
    text = replace(text, r"[^\w\s.!?]" => " ")
    
    # Split into tokens
    tokens = split(text)
    
    # Filter out stop words and short tokens
    stop_words = Set([
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", 
        "for", "of", "with", "by", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "this", "that", "these", "those"
    ])
    
    filtered_tokens = String[]
    for token in tokens
        if length(token) > 2 && !(token in stop_words)
            push!(filtered_tokens, token)
        end
    end
    
    return filtered_tokens
end

"""
    extract_sentences(text::String) -> Vector{String}

Extract sentences from text for contextual analysis.

# Arguments
- `text::String`: Input text

# Returns
- `Vector{String}`: Individual sentences
"""
function extract_sentences(text::String)::Vector{String}
    isempty(text) && return String[]
    
    # Split on sentence boundaries
    sentences = split(text, r"[.!?]+")
    
    # Clean and filter sentences
    cleaned_sentences = String[]
    for sentence in sentences
        sentence = strip(sentence)
        if !isempty(sentence) && length(sentence) > 10
            push!(cleaned_sentences, sentence)
        end
    end
    
    return cleaned_sentences
end

"""
    find_motif_matches(tokens::Vector{String}, motif_type::MotifType) -> Vector{String}

Find tokens that match patterns for a specific motif type.

# Arguments
- `tokens::Vector{String}`: Preprocessed tokens
- `motif_type::MotifType`: Type of motif to search for

# Returns
- `Vector{String}`: Matching tokens
"""
function find_motif_matches(tokens::Vector{String}, motif_type::MotifType)::Vector{String}
    patterns = get_motif_patterns(motif_type)
    matches = String[]
    
    for token in tokens
        for pattern in patterns
            if occursin(pattern, token)
                push!(matches, token)
                break
            end
        end
    end
    
    return unique(matches)
end

"""
    calculate_position_confidence(tokens::Vector{String}, matches::Vector{String}) -> Tuple{Float64, Float64}

Calculate position and confidence for motif matches.

# Arguments
- `tokens::Vector{String}`: All tokens in the text
- `matches::Vector{String}`: Matching tokens for a motif

# Returns
- `Tuple{Float64, Float64}`: (position, confidence)
"""
function calculate_position_confidence(
    tokens::Vector{String}, 
    matches::Vector{String}
)::Tuple{Float64, Float64}
    
    if isempty(matches) || isempty(tokens)
        return (0.0, 0.0)
    end
    
    # Calculate average position of matches
    positions = Float64[]
    for match in matches
        for (i, token) in enumerate(tokens)
            if token == match
                push!(positions, i / length(tokens))
                break
            end
        end
    end
    
    position = isempty(positions) ? 0.0 : mean(positions)
    
    # Calculate confidence based on match density and uniqueness
    match_density = length(matches) / length(tokens)
    uniqueness = length(unique(matches)) / length(matches)
    confidence = min(1.0, match_density * uniqueness * 2.0)
    
    return (position, confidence)
end

"""
    extract_context(text::String, position::Float64, window_size::Int=10) -> String

Extract contextual text around a given position.

# Arguments
- `text::String`: Full text
- `position::Float64`: Relative position (0.0 to 1.0)
- `window_size::Int`: Number of words to include on each side

# Returns
- `String`: Contextual text snippet
"""
function extract_context(text::String, position::Float64, window_size::Int=10)::String
    tokens = split(text)
    if isempty(tokens)
        return ""
    end
    
    center_idx = max(1, min(length(tokens), round(Int, position * length(tokens))))
    start_idx = max(1, center_idx - window_size)
    end_idx = min(length(tokens), center_idx + window_size)
    
    context_tokens = tokens[start_idx:end_idx]
    return join(context_tokens, " ")
end

export preprocess_text, extract_sentences, find_motif_matches
export calculate_position_confidence, extract_context

end # module