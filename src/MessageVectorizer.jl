module MessageVectorizer

using LinearAlgebra
using Statistics
using Random

# Motif token structure
struct MotifToken
    type::Symbol
    weight::Float64
    attributes::Dict{Symbol, Any}
end

# Message state structure
struct MessageState
    vector_representation::Vector{Float64}
    entropy_score::Float64
    coherence_metrics::Dict{String, Float64}
end

# Enhanced motif detection patterns
const KOJIMA_MOTIFS = Dict{Symbol, Vector{String}}(
    :isolation => ["alone", "solitude", "desert", "empty", "void", "silence"],
    :snake => ["snake", "serpent", "coil", "slither", "ouroboros"],
    :strand => ["strand", "thread", "connection", "link", "chain"],
    :memory => ["memory", "remember", "forgotten", "past", "dream", "phantom"],
    :temporal => ["time", "temporal", "moment", "eternity", "cycle"],
    :fragmentation => ["fragment", "broken", "shattered", "pieces", "scattered"]
)

# Create motif tokens from text
function create_motif_tokens(text::String)
    tokens = MotifToken[]
    words = split(lowercase(text))
    
    for (motif_type, patterns) in KOJIMA_MOTIFS
        matches = String[]
        for word in words
            for pattern in patterns
                if occursin(pattern, word)
                    push!(matches, word)
                    break
                end
            end
        end
        
        if !isempty(matches)
            weight = length(matches) / length(words)
            categories = Set{Symbol}()
            
            # Categorize based on context
            if motif_type == :isolation
                push!(categories, :spatial)
            elseif motif_type == :temporal
                push!(categories, :temporal)
            elseif motif_type == :memory
                push!(categories, :cognitive)
            end
            
            token = MotifToken(motif_type, weight, Dict(
                :matches => matches,
                :categories => categories
            ))
            push!(tokens, token)
        end
    end
    
    return tokens
end

# Vectorize motif tokens
function vectorize_motifs(motif_tokens::Vector{MotifToken})
    if isempty(motif_tokens)
        return zeros(6)  # 6-dimensional vector for 6 motif types
    end
    
    vector = zeros(6)
    motif_types = [:isolation, :snake, :strand, :memory, :temporal, :fragmentation]
    
    for token in motif_tokens
        idx = findfirst(x -> x == token.type, motif_types)
        if idx !== nothing
            vector[idx] = token.weight
        end
    end
    
    return vector
end

# Calculate entropy score
function calculate_entropy(vector::Vector{Float64})
    non_zero = vector[vector .> 0]
    if isempty(non_zero)
        return 0.0
    end
    
    total = sum(non_zero)
    probabilities = non_zero / total
    
    entropy = -sum(p * log(p) for p in probabilities if p > 0)
    return entropy
end

# Calculate coherence metrics
function calculate_coherence_metrics(motif_tokens::Vector{MotifToken})
    if length(motif_tokens) < 2
        return Dict{String, Float64}(
            "semantic_coherence" => 1.0,
            "temporal_consistency" => 1.0
        )
    end
    
    weights = [token.weight for token in motif_tokens]
    
    return Dict{String, Float64}(
        "semantic_coherence" => 1.0 - std(weights) / (mean(weights) + eps()),
        "temporal_consistency" => 1.0 - (length(motif_tokens) == 0 ? 0.0 : std([motif.weight for motif in motif_tokens]) / mean([motif.weight for motif in motif_tokens]))
    )
end

# Interface with al-ULS (Topology-Aware Uncertainty Learning Systems)
function al_uls_interface(message_state::MessageState)
    vector = message_state.vector_representation
    entropy = message_state.entropy_score
    coherence = message_state.coherence_metrics
    
    # Simulate ULS processing
    vector_norm = norm(vector)
    sparsity_ratio = count(abs.(vector) .< 1e-6) / length(vector)
    
    return Dict{String, Any}(
        "original_vector" => vector,
        "vector_norm" => vector_norm,
        "compressed_size" => length(vector) * (1 - sparsity_ratio),
        "information_density" => entropy > 0 ? vector_norm / (entropy + eps()) : 0.0,
        "entropy_score" => entropy,
        "coherence_metrics" => coherence
    )
end

# Enhanced entropy analysis
function motif_entropy_analysis(motif_tokens::Vector{MotifToken})
    if isempty(motif_tokens)
        return Dict("total_entropy" => 0.0, "motif_contributions" => Dict{String, Float64}())
    end
    
    weights = [motif.weight for motif in motif_tokens]
    total_weight = sum(weights)
    
    if total_weight == 0
        return Dict("total_entropy" => 0.0, "motif_contributions" => Dict{String, Float64}())
    end
    
    probabilities = weights / total_weight
    motif_contributions = Dict{String, Float64}()
    
    total_entropy = 0.0
    for (i, motif) in enumerate(motif_tokens)
        p = probabilities[i]
        if p > 0
            contribution = -p * log(p)
            motif_contributions[string(motif.type)] = contribution
            total_entropy += contribution
        end
    end
    
    return Dict(
        "total_entropy" => total_entropy,
        "motif_contributions" => motif_contributions,
        "motif_weights" => Dict(string(motif.type) => motif.weight for motif in motif_tokens)
    )
end

end # module