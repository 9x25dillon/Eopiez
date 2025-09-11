include("tokenizer.jl")
include("sheaf_engine.jl")
include("../MessageVectorizer.jl")

using .MessageVectorizer

# Enhanced motif analysis pipeline
function enhanced_motif_analysis(text::String)
    # Step 1: Create motif tokens
    motif_tokens = create_motif_tokens(text)
    
    # Step 2: Create sheaf structure
    sheaf = create_narrative_sheaf(motif_tokens)
    
    # Step 3: Calculate sheaf cohomology
    cohomology = calculate_sheaf_cohomology(sheaf)
    
    # Step 4: Vectorize motifs
    vector = vectorize_motifs(motif_tokens)
    
    # Step 5: Calculate entropy and coherence
    entropy = calculate_entropy(vector)
    coherence_metrics = calculate_coherence_metrics(motif_tokens)
    
    # Step 6: Create message state
    message_state = MessageState(vector, entropy, coherence_metrics)
    
    # Step 7: Interface with ULS
    uls_data = al_uls_interface(message_state)
    
    # Step 8: Find emergent structures
    emergent_structures = detect_emergent_structure(sheaf)
    
    # Step 9: Build coherent narrative paths
    coherent_paths = build_coherent_narrative(sheaf)
    
    return Dict{String, Any}(
        "motif_tokens" => motif_tokens,
        "sheaf_info" => Dict(
            "total_sections" => length(sheaf.sections),
            "global_sections" => length(find_global_sections(sheaf)),
            "cohomology" => cohomology
        ),
        "vector_metrics" => Dict(
            "vector_norm" => uls_data["vector_norm"],
            "entropy_score" => entropy,
            "motif_count" => length(motif_tokens)
        ),
        "emergent_structures" => emergent_structures,
        "coherent_paths" => coherent_paths,
        "uls_interface" => uls_data
    )
end

# Detect emergent structures in the sheaf
function detect_emergent_structure(sheaf::NarrativeSheaf)
    emergent_structures = Dict{String, Vector{Any}}()
    
    if isempty(sheaf.sections)
        return emergent_structures
    end
    
    # Find patterns across sections
    motif_types = [section["motif_type"] for section in sheaf.sections]
    weights = [section["total_weight"] for section in sheaf.sections]
    
    # High-weight motifs
    high_weight_threshold = 0.5 * maximum(weights)
    high_weight_motifs = [motif_types[i] for i in 1:length(weights) if weights[i] > high_weight_threshold]
    
    if !isempty(high_weight_motifs)
        emergent_structures["dominant_motifs"] = high_weight_motifs
    end
    
    # Temporal patterns (if position data available)
    temporal_motifs = []
    for section in sheaf.sections
        if haskey(section, "temporal_position")
            push!(temporal_motifs, section)
        end
    end
    
    if !isempty(temporal_motifs)
        emergent_structures["temporal_patterns"] = temporal_motifs
    end
    
    return emergent_structures
end

# Build coherent narrative paths
function build_coherent_narrative(sheaf::NarrativeSheaf)
    coherent_paths = []
    
    if isempty(sheaf.sections)
        return coherent_paths
    end
    
    # Simple path: connect sections by motif type similarity
    for (i, section1) in enumerate(sheaf.sections)
        for (j, section2) in enumerate(sheaf.sections)
            if i != j
                # Calculate coherence between sections
                coherence_score = calculate_section_coherence(section1, section2)
                
                if coherence_score > 0.3  # Threshold for coherence
                    path = [section1, section2]
                    push!(coherent_paths, (path, coherence_score))
                end
            end
        end
    end
    
    return coherent_paths
end

# Calculate coherence between two sections
function calculate_section_coherence(section1::Dict{String, Any}, section2::Dict{String, Any})
    # Simple coherence based on motif type similarity
    if section1["motif_type"] == section2["motif_type"]
        return 0.8
    else
        return 0.2
    end
end

# Process document with sheaf analysis
function process_document_sheaf(text::String)
    motif_tokens = create_motif_tokens(text)
    return create_narrative_sheaf(motif_tokens)
end

# Export main functions
export enhanced_motif_analysis, process_document_sheaf, detect_emergent_structure, build_coherent_narrative