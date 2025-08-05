using LinearAlgebra
using Statistics

# Sheaf structure for narrative analysis
struct NarrativeSheaf
    open_sets::Vector{Set{Int}}
    sections::Vector{Dict{String, Any}}
    restriction_maps::Dict{Tuple{Int, Int}, Function}
end

# Create sheaf from motif tokens
function create_narrative_sheaf(motif_tokens::Vector{MotifToken})
    if isempty(motif_tokens)
        return NarrativeSheaf([], [], Dict())
    end
    
    # Create open sets based on motif types
    motif_types = unique([token.type for token in motif_tokens])
    open_sets = [Set{Int}() for _ in motif_types]
    
    # Assign tokens to open sets
    for (i, token) in enumerate(motif_tokens)
        type_idx = findfirst(x -> x == token.type, motif_types)
        if type_idx !== nothing
            push!(open_sets[type_idx], i)
        end
    end
    
    # Create sections (local data)
    sections = []
    for (i, open_set) in enumerate(open_sets)
        if !isempty(open_set)
            section_data = Dict{String, Any}(
                "motif_type" => string(motif_types[i]),
                "tokens" => [motif_tokens[j] for j in open_set],
                "total_weight" => sum([motif_tokens[j].weight for j in open_set]),
                "entropy" => calculate_section_entropy([motif_tokens[j] for j in open_set])
            )
            push!(sections, section_data)
        end
    end
    
    # Create restriction maps
    restriction_maps = Dict{Tuple{Int, Int}, Function}()
    
    return NarrativeSheaf(open_sets, sections, restriction_maps)
end

# Calculate entropy for a section
function calculate_section_entropy(tokens::Vector{MotifToken})
    if isempty(tokens)
        return 0.0
    end
    
    weights = [token.weight for token in tokens]
    total_weight = sum(weights)
    
    if total_weight == 0
        return 0.0
    end
    
    probabilities = weights / total_weight
    entropy = -sum(p * log(p) for p in probabilities if p > 0)
    
    return entropy
end

# Find global sections (coherent narrative elements)
function find_global_sections(sheaf::NarrativeSheaf)
    global_sections = []
    
    for section in sheaf.sections
        if section["total_weight"] > 0.1  # Threshold for significance
            push!(global_sections, section)
        end
    end
    
    return global_sections
end

# Calculate sheaf cohomology (narrative coherence)
function calculate_sheaf_cohomology(sheaf::NarrativeSheaf)
    if isempty(sheaf.sections)
        return Dict{String, Float64}(
            "h0" => 0.0,  # Global sections
            "h1" => 0.0,  # Obstructions to coherence
            "coherence_score" => 0.0
        )
    end
    
    # H^0: Global sections
    global_sections = find_global_sections(sheaf)
    h0 = length(global_sections)
    
    # H^1: Estimate obstructions (simplified)
    total_entropy = sum([section["entropy"] for section in sheaf.sections])
    h1 = total_entropy / length(sheaf.sections)
    
    # Coherence score
    coherence_score = h0 / (h0 + h1 + eps())
    
    return Dict{String, Float64}(
        "h0" => Float64(h0),
        "h1" => h1,
        "coherence_score" => coherence_score
    )
end

# Export functions
export create_narrative_sheaf, find_global_sections, calculate_sheaf_cohomology