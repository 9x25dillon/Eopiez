using TextAnalysis

# Enhanced tokenizer for Kojima-esque motifs
function tokenize_text(text::String)
    # Basic text preprocessing
    text = lowercase(text)
    text = replace(text, r"[^\w\s]" => " ")
    words = split(text)
    
    # Remove common stop words
    stop_words = Set(["the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"])
    words = [word for word in words if !(word in stop_words) && length(word) > 2]
    
    return words
end

# Create motif tokens with enhanced context
function create_contextual_motif_tokens(text::String)
    words = tokenize_text(text)
    sentences = split(text, r"[.!?]+")
    
    tokens = []
    
    for (sentence_idx, sentence) in enumerate(sentences)
        sentence_words = tokenize_text(sentence)
        
        # Find motif patterns in this sentence
        for (motif_type, patterns) in KOJIMA_MOTIFS
            matches = String[]
            for word in sentence_words
                for pattern in patterns
                    if occursin(pattern, word)
                        push!(matches, word)
                        break
                    end
                end
            end
            
            if !isempty(matches)
                # Calculate contextual weight
                weight = length(matches) / length(sentence_words)
                
                # Add position information
                position = sentence_idx / length(sentences)
                
                # Create enhanced attributes
                attributes = Dict{Symbol, Any}(
                    :matches => matches,
                    :sentence_position => position,
                    :sentence_length => length(sentence_words),
                    :context => sentence
                )
                
                token = MotifToken(motif_type, weight, attributes)
                push!(tokens, token)
            end
        end
    end
    
    return tokens
end

# Export functions
export tokenize_text, create_contextual_motif_tokens