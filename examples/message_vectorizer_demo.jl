using Pkg
Pkg.activate(".")

using MessageVectorizer
using JSON3
using Printf

"""
    create_sample_motifs()

Create sample motif tokens for demonstration.
"""
function create_sample_motifs()
    motifs = [
        # Isolation + Time motif
        MotifToken(
            :isolation_time,
            Dict{Symbol, Any}(
                :intensity => 0.8,
                :duration => 24.0,
                :spatial_separation => 0.9
            ),
            0.7,
            [:temporal, :spatial, :emotional]
        ),
        
        # Decay + Memory motif
        MotifToken(
            :decay_memory,
            Dict{Symbol, Any}(
                :decay_rate => 0.3,
                :memory_strength => 0.6,
                :forgetting_curve => "exponential"
            ),
            0.6,
            [:cognitive, :temporal, :neural]
        ),
        
        # Connection + Network motif
        MotifToken(
            :connection_network,
            Dict{Symbol, Any}(
                :connectivity => 0.9,
                :network_density => 0.7,
                :information_flow => "bidirectional"
            ),
            0.8,
            [:social, :informational, :structural]
        ),
        
        # Transformation + Emergence motif
        MotifToken(
            :transformation_emergence,
            Dict{Symbol, Any}(
                :transformation_rate => 0.4,
                :emergence_threshold => 0.6,
                :complexity_growth => "nonlinear"
            ),
            0.5,
            [:evolutionary, :systemic, :adaptive]
        )
    ]
    
    return motifs
end

"""
    demonstrate_vectorization()

Demonstrate the message vectorization process.
"""
function demonstrate_vectorization()
    println("=== Message Vectorizer Demonstration ===\n")
    
    # Initialize vectorizer
    vectorizer = MessageVectorizer(32, entropy_threshold=0.6, compression_ratio=0.75)
    println("✓ Initialized MessageVectorizer with 32-dimensional embeddings")
    
    # Create sample motifs
    motifs = create_sample_motifs()
    println("✓ Created $(length(motifs)) sample motif tokens")
    
    # Add motif embeddings to vectorizer
    for motif in motifs
        add_motif_embedding!(vectorizer, motif)
    end
    println("✓ Added motif embeddings to vectorizer\n")
    
    # Demonstrate individual motif embeddings
    println("--- Individual Motif Embeddings ---")
    for motif in motifs
        embedding = vectorizer.motif_embeddings[motif.name]
        println("$(motif.name): $(round(norm(embedding), digits=3)) norm, $(length(embedding)) dimensions")
    end
    println()
    
    # Vectorize different motif combinations
    println("--- Message Vectorization Examples ---")
    
    # Example 1: Isolation + Decay
    isolation_decay = [motifs[1], motifs[2]]
    state1 = vectorize_message(isolation_decay, vectorizer)
    println("1. Isolation + Decay Message:")
    println("   Entropy: $(round(state1.entropy_score, digits=3))")
    println("   Vector norm: $(round(norm(state1.vector_representation), digits=3))")
    println("   Motif config: $(state1.motif_configuration)")
    println()
    
    # Example 2: Connection + Transformation
    connection_transform = [motifs[3], motifs[4]]
    state2 = vectorize_message(connection_transform, vectorizer)
    println("2. Connection + Transformation Message:")
    println("   Entropy: $(round(state2.entropy_score, digits=3))")
    println("   Vector norm: $(round(norm(state2.vector_representation), digits=3))")
    println("   Motif config: $(state2.motif_configuration)")
    println()
    
    # Example 3: All motifs combined
    all_motifs = motifs
    state3 = vectorize_message(all_motifs, vectorizer)
    println("3. All Motifs Combined:")
    println("   Entropy: $(round(state3.entropy_score, digits=3))")
    println("   Vector norm: $(round(norm(state3.vector_representation), digits=3))")
    println("   Motif config: $(state3.motif_configuration)")
    println()
    
    # Demonstrate al-ULS interface
    println("--- al-ULS Interface Output ---")
    for (i, state) in enumerate([state1, state2, state3])
        uls_output = al_uls_interface(state)
        println("Message $i:")
        println("   Symbolic expression: $(uls_output["symbolic_expression"][1:50])...")
        println("   Compressed size: $(uls_output["compressed_size"])")
        println("   Information density: $(round(uls_output["information_density"], digits=4))")
        println()
    end
    
    return vectorizer, [state1, state2, state3]
end

"""
    analyze_entropy_distribution(states::Vector{MessageState})

Analyze entropy distribution across message states.
"""
function analyze_entropy_distribution(states::Vector{MessageState})
    println("--- Entropy Analysis ---")
    
    entropies = [state.entropy_score for state in states]
    
    println("Entropy Statistics:")
    println("   Mean: $(round(mean(entropies), digits=3))")
    println("   Std: $(round(std(entropies), digits=3))")
    println("   Min: $(round(minimum(entropies), digits=3))")
    println("   Max: $(round(maximum(entropies), digits=3))")
    println()
    
    # Entropy ranking
    println("Entropy Ranking (highest to lowest):")
    for (i, state) in enumerate(sort(states, by=s->s.entropy_score, rev=true))
        motif_names = collect(keys(state.motif_configuration))
        println("   $(i). $(join(motif_names, " + ")): $(round(state.entropy_score, digits=3))")
    end
    println()
end

"""
    save_results(vectorizer::MessageVectorizer, states::Vector{MessageState})

Save results to JSON for external consumption.
"""
function save_results(vectorizer::MessageVectorizer, states::Vector{MessageState})
    results = Dict{String, Any}(
        "vectorizer_config" => Dict{String, Any}(
            "embedding_dim" => vectorizer.embedding_dim,
            "entropy_threshold" => vectorizer.entropy_threshold,
            "compression_ratio" => vectorizer.compression_ratio
        ),
        "message_states" => [al_uls_interface(state) for state in states],
        "motif_embeddings" => Dict{String, Vector{Float64}}(
            string(k) => v for (k, v) in vectorizer.motif_embeddings
        )
    )
    
    # Save to file
    open("message_vectorizer_results.json", "w") do f
        write(f, JSON3.write(results, pretty=true))
    end
    
    println("✓ Results saved to message_vectorizer_results.json")
end

# Run demonstration
if abspath(PROGRAM_FILE) == @__FILE__
    println("Starting Message Vectorizer Demonstration...\n")
    
    vectorizer, states = demonstrate_vectorization()
    analyze_entropy_distribution(states)
    save_results(vectorizer, states)
    
    println("=== Demonstration Complete ===")
end