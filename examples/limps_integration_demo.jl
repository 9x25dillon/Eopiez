using Pkg
Pkg.activate(".")

using JSON3
using Printf

# Include all modules
include("../src/MessageVectorizer.jl")
include("../src/motif_detection/motifs.jl")
include("../src/motif_detection/parser.jl")
include("../src/motif_detection/motif_server.jl")
include("../src/limps/symbolic_memory.jl")

using .MessageVectorizer
using .MotifDefinitions
using .MotifParser
using .MotifServer
using .LiMpsSymbolicMemory

"""
    create_kojima_narrative_texts()

Create a series of Kojima-esque narrative texts for demonstration.
"""
function create_kojima_narrative_texts()
    texts = [
        # Episode 1: The Awakening
        """
        In the vast digital desert, Snake found himself alone, disconnected from the network. 
        The silence was absolute, broken only by the hiss of static from his radio. 
        He remembered the past, the memories of his comrades now lost in the void. 
        The snake coiled around his thoughts, a symbol of the endless cycle of war and peace.
        """,
        
        # Episode 2: The Connection
        """
        The strands of memory connected him to his past, like DNA encoding his identity. 
        He recalled the flashback of his childhood, the snake that had taught him about survival. 
        His consciousness was fragmented, pieces of himself scattered across the digital landscape. 
        The ouroboros of his existence continued its eternal dance.
        """,
        
        # Episode 3: The Battlefield
        """
        The battlefield was a web of connections, each soldier a node in the network. 
        Messages flowed through the air like invisible strands, carrying information and orders. 
        The snake of war slithered through the jungle, its venom spreading chaos and destruction. 
        In the isolation of combat, each fighter became a ghost, a phantom of their former self.
        """,
        
        # Episode 4: The Transformation
        """
        The forest whispered with ancient memories, its roots entwined like strands of DNA. 
        The snake emerged from the shadows, its scales reflecting the moonlight. 
        Technology and nature clashed in this sacred space, where the digital and organic merged. 
        The transformation was complete - man had become machine, and machine had become nature.
        """,
        
        # Episode 5: The Resolution
        """
        In the final moments, Snake understood the true nature of his isolation. 
        The memories that had haunted him were not chains but threads connecting him to humanity. 
        The snake, once a symbol of fear, became a bridge between past and future. 
        The strands of his existence wove a tapestry of meaning in the void.
        """
    ]
    
    return texts
end

"""
    demonstrate_complete_integration()

Demonstrate the complete integration of all three systems.
"""
function demonstrate_complete_integration()
    println("🔥 === LiMps Integration Demo - Weaving the Tapestry === 🔥\n")
    
    # Initialize all systems
    println("🎯 Initializing Systems...")
    
    # 1. Initialize LiMps Engine
    limps_engine = LiMpsEngine(
        coherence_threshold = 0.7,
        narrative_weaving_factor = 0.9,
        memory_decay_rate = 0.05,
        context_window_size = 15,
        max_memory_entities = 500
    )
    println("✓ LiMps Engine initialized")
    
    # 2. Initialize Message Vectorizer
    vectorizer = MessageVectorizer(64, entropy_threshold=0.6, compression_ratio=0.8)
    println("✓ Message Vectorizer initialized")
    
    # 3. Get narrative texts
    texts = create_kojima_narrative_texts()
    println("✓ Created $(length(texts)) narrative episodes")
    
    # Process each episode
    for (episode_num, text) in enumerate(texts)
        println("\n📖 === Processing Episode $episode_num ===")
        println("Text: $(text[1:80])...")
        
        # Step 1: Motif Detection
        println("\n🔍 Step 1: Motif Detection")
        analysis = parse_document(text)
        motif_tokens = extract_motif_tokens(analysis)
        println("  - Detected $(length(motif_tokens)) motif tokens")
        
        # Display detected motifs
        for token in motif_tokens
            println("    • $(token.name): weight $(round(token.weight, digits=3)), context: $(join([string(ctx) for ctx in token.context], ", "))")
        end
        
        # Step 2: Message Vectorization
        println("\n🧬 Step 2: Message Vectorization")
        for token in motif_tokens
            add_motif_embedding!(vectorizer, token)
        end
        
        message_state = vectorize_message(motif_tokens, vectorizer)
        uls_output = al_uls_interface(message_state)
        println("  - Vectorized message with entropy: $(round(message_state.entropy_score, digits=3))")
        println("  - Information density: $(round(uls_output["information_density"], digits=4))")
        
        # Step 3: LiMps Memory Storage
        println("\n🧠 Step 3: LiMps Memory Storage")
        for token in motif_tokens
            # Create motif data for LiMps
            motif_data = Dict{String, Any}(
                "id" => "$(token.name)_episode_$(episode_num)",
                "type" => "motif",
                "properties" => Dict{String, Any}(
                    "frequency" => token.properties[:frequency],
                    "confidence" => token.properties[:confidence],
                    "weight" => token.weight,
                    "episode" => episode_num,
                    "symbolic_expression" => string(message_state.symbolic_expression)
                ),
                "weight" => token.weight,
                "context" => [string(ctx) for ctx in token.context]
            )
            
            # Store in LiMps
            memory_entity = store_motif_memory(limps_engine, motif_data)
            println("    • Stored $(memory_entity.id) with coherence: $(round(memory_entity.coherence_score, digits=3))")
        end
        
        # Step 4: Memory Tapestry Weaving
        println("\n🕸️ Step 4: Memory Tapestry Weaving")
        focus_context = ["isolation", "memory", "snake", "strand"]
        tapestry = weave_memory_tapestry(limps_engine, focus_context)
        println("  - Wove tapestry with $(tapestry["relevant_memories"]) relevant memories")
        println("  - Coherence score: $(round(tapestry["coherence_score"], digits=3))")
        println("  - Narrative complexity: $(round(tapestry["narrative_complexity"], digits=3))")
    end
    
    return limps_engine, vectorizer, texts
end

"""
    demonstrate_memory_retrieval(limps_engine::LiMpsEngine)

Demonstrate advanced memory retrieval and narrative generation.
"""
function demonstrate_memory_retrieval(limps_engine::LiMpsEngine)
    println("\n🎭 === Memory Retrieval & Narrative Generation ===\n")
    
    # Test different retrieval contexts
    retrieval_contexts = [
        (["isolation"], "Isolation-focused memories"),
        (["snake", "strand"], "Snake and strand connections"),
        (["memory", "identity"], "Memory and identity themes"),
        (["technology", "war"], "Technology and war motifs"),
        (["isolation", "memory", "snake", "strand"], "Complete narrative context")
    ]
    
    for (context, description) in retrieval_contexts
        println("🔍 $description")
        println("Context: $(join(context, ", "))")
        
        # Retrieve contextual memories
        memories = retrieve_contextual_memories(limps_engine, context, limit = 8)
        println("  - Retrieved $(length(memories)) memories")
        
        # Show top memories
        for (i, memory) in enumerate(memories[1:min(3, length(memories))])
            println("    $(i). $(memory.id): importance $(round(memory.narrative_importance, digits=3))")
        end
        
        # Generate symbolic narrative
        narrative = generate_symbolic_narrative(limps_engine, context)
        println("  - Generated narrative with $(length(narrative["memories"])) memory entities")
        println("  - Temporal flow: $(narrative["temporal_flow"]["temporal_flow"])")
        
        # Show symbolic themes
        if !isempty(narrative["symbolic_themes"])
            top_theme = narrative["symbolic_themes"][1]
            println("  - Dominant theme: $(top_theme["theme"]) (frequency: $(top_theme["frequency"]))")
        end
        
        println()
    end
end

"""
    demonstrate_memory_analysis(limps_engine::LiMpsEngine)

Demonstrate comprehensive memory analysis capabilities.
"""
function demonstrate_memory_analysis(limps_engine::LiMpsEngine)
    println("\n📊 === Memory Analysis & Pattern Recognition ===\n")
    
    # Analyze memory patterns
    patterns = analyze_memory_patterns(limps_engine)
    
    println("📈 Memory Statistics:")
    println("  - Total entities: $(patterns["total_entities"])")
    println("  - Total relationships: $(patterns["total_relationships"])")
    
    # Type distribution
    println("\n🎭 Type Distribution:")
    for (type, count) in sort(collect(patterns["type_distribution"]), by = x -> x[2], rev = true)
        println("  - $type: $count entities")
    end
    
    # Context distribution
    println("\n🏷️ Context Distribution:")
    for (context, count) in sort(collect(patterns["context_distribution"]), by = x -> x[2], rev = true)[1:8]
        println("  - $context: $count occurrences")
    end
    
    # Relationship types
    println("\n🔗 Relationship Types:")
    for (rel_type, count) in sort(collect(patterns["relationship_types"]), by = x -> x[2], rev = true)
        println("  - $rel_type: $count relationships")
    end
    
    # Coherence statistics
    coherence_stats = patterns["coherence_stats"]
    println("\n🧠 Coherence Statistics:")
    println("  - Mean: $(round(coherence_stats["mean"], digits=3))")
    println("  - Std: $(round(coherence_stats["std"], digits=3))")
    println("  - Range: $(round(coherence_stats["min"], digits=3)) - $(round(coherence_stats["max"], digits=3))")
    
    # Importance statistics
    importance_stats = patterns["importance_stats"]
    println("\n⭐ Importance Statistics:")
    println("  - Mean: $(round(importance_stats["mean"], digits=3))")
    println("  - Std: $(round(importance_stats["std"], digits=3))")
    println("  - Range: $(round(importance_stats["min"], digits=3)) - $(round(importance_stats["max"], digits=3))")
    
    return patterns
end

"""
    demonstrate_memory_graph(limps_engine::LiMpsEngine)

Demonstrate memory graph creation and analysis.
"""
function demonstrate_memory_graph(limps_engine::LiMpsEngine)
    println("\n🕸️ === Memory Graph Analysis ===\n")
    
    # Create memory graph
    graph = create_memory_graph(limps_engine)
    
    println("📊 Graph Statistics:")
    println("  - Nodes: $(graph["total_nodes"])")
    println("  - Edges: $(graph["total_edges"])")
    println("  - Density: $(round(graph["graph_density"], digits=4))")
    
    # Show some node examples
    println("\n🎯 Sample Nodes:")
    for (i, node) in enumerate(graph["nodes"][1:min(5, length(graph["nodes"]))])
        println("  $(i). $(node["id"])")
        println("     Type: $(node["type"]), Weight: $(round(node["weight"], digits=3))")
        println("     Context: $(join(node["context"], ", "))")
        println("     Importance: $(round(node["importance"], digits=3))")
    end
    
    # Show some edge examples
    println("\n🔗 Sample Edges:")
    for (i, edge) in enumerate(graph["edges"][1:min(5, length(graph["edges"]))])
        println("  $(i). $(edge["source"]) → $(edge["target"])")
        println("     Type: $(edge["type"]), Strength: $(round(edge["strength"], digits=3))")
        println("     Context overlap: $(join(edge["context_overlap"], ", "))")
    end
    
    return graph
end

"""
    demonstrate_symbolic_narrative_generation(limps_engine::LiMpsEngine)

Demonstrate advanced symbolic narrative generation.
"""
function demonstrate_symbolic_narrative_generation(limps_engine::LiMpsEngine)
    println("\n📖 === Symbolic Narrative Generation ===\n")
    
    # Generate narratives for different contexts
    narrative_contexts = [
        (["isolation"], "The Solitude Narrative"),
        (["snake", "memory"], "The Serpent's Memory"),
        (["strand", "technology"], "The Digital Web"),
        (["identity", "memory"], "The Self's Echo"),
        (["isolation", "snake", "strand", "memory"], "The Complete Tapestry")
    ]
    
    for (context, title) in narrative_contexts
        println("📚 $title")
        println("Context: $(join(context, ", "))")
        
        # Generate narrative
        narrative = generate_symbolic_narrative(limps_engine, context)
        
        # Display narrative structure
        tapestry = narrative["tapestry"]
        println("  - Tapestry coherence: $(round(tapestry["coherence_score"], digits=3))")
        println("  - Narrative complexity: $(round(tapestry["narrative_complexity"], digits=3))")
        println("  - Temporal span: $(round(tapestry["temporal_span"], digits=3))")
        
        # Show temporal flow
        temporal_flow = narrative["temporal_flow"]
        println("  - Temporal flow: $(temporal_flow["temporal_flow"])")
        println("  - Total events: $(temporal_flow["total_events"])")
        
        # Show symbolic themes
        themes = narrative["symbolic_themes"]
        if !isempty(themes)
            println("  - Symbolic themes:")
            for theme in themes[1:min(3, length(themes))]
                println("    • $(theme["theme"]): $(theme["frequency"]) occurrences")
            end
        end
        
        # Show key relationships
        relationships = narrative["relationships"]
        if !isempty(relationships)
            println("  - Key relationships:")
            for rel in relationships[1:min(3, length(relationships))]
                println("    • $(rel["source"]) → $(rel["target"]) ($(rel["type"]))")
            end
        end
        
        println()
    end
end

"""
    save_integration_results(limps_engine::LiMpsEngine, vectorizer::MessageVectorizer, 
                           patterns::Dict{String, Any}, graph::Dict{String, Any})

Save all integration results to files.
"""
function save_integration_results(limps_engine::LiMpsEngine, vectorizer::MessageVectorizer, 
                                patterns::Dict{String, Any}, graph::Dict{String, Any})
    println("\n💾 === Saving Integration Results ===\n")
    
    # Export LiMps data
    limps_data = export_limps_data(limps_engine)
    open("limps_memory_export.json", "w") do f
        write(f, JSON3.write(limps_data, pretty=true))
    end
    println("✓ Saved LiMps memory export to limps_memory_export.json")
    
    # Save memory patterns
    open("memory_patterns.json", "w") do f
        write(f, JSON3.write(patterns, pretty=true))
    end
    println("✓ Saved memory patterns to memory_patterns.json")
    
    # Save memory graph
    open("memory_graph.json", "w") do f
        write(f, JSON3.write(graph, pretty=true))
    end
    println("✓ Saved memory graph to memory_graph.json")
    
    # Generate narrative for complete context
    complete_narrative = generate_symbolic_narrative(limps_engine, ["isolation", "snake", "strand", "memory", "identity"])
    open("complete_narrative.json", "w") do f
        write(f, JSON3.write(complete_narrative, pretty=true))
    end
    println("✓ Saved complete narrative to complete_narrative.json")
    
    # Create integration summary
    summary = Dict{String, Any}(
        "integration_summary" => Dict{String, Any}(
            "total_memory_entities" => length(limps_engine.memory_entities),
            "total_relationships" => length(limps_engine.relationships),
            "graph_density" => graph["graph_density"],
            "avg_coherence" => patterns["coherence_stats"]["mean"],
            "avg_importance" => patterns["importance_stats"]["mean"],
            "dominant_themes" => sort(collect(patterns["context_distribution"]), by = x -> x[2], rev = true)[1:5]
        ),
        "system_configuration" => Dict{String, Any}(
            "coherence_threshold" => limps_engine.coherence_threshold,
            "narrative_weaving_factor" => limps_engine.narrative_weaving_factor,
            "memory_decay_rate" => limps_engine.memory_decay_rate,
            "context_window_size" => limps_engine.context_window_size
        ),
        "timestamp" => time()
    )
    
    open("integration_summary.json", "w") do f
        write(f, JSON3.write(summary, pretty=true))
    end
    println("✓ Saved integration summary to integration_summary.json")
end

# Run the complete integration demonstration
if abspath(PROGRAM_FILE) == @__FILE__
    println("🚀 Starting LiMps Integration Demo...\n")
    
    # Run complete integration
    limps_engine, vectorizer, texts = demonstrate_complete_integration()
    
    # Demonstrate memory retrieval
    demonstrate_memory_retrieval(limps_engine)
    
    # Demonstrate memory analysis
    patterns = demonstrate_memory_analysis(limps_engine)
    
    # Demonstrate memory graph
    graph = demonstrate_memory_graph(limps_engine)
    
    # Demonstrate symbolic narrative generation
    demonstrate_symbolic_narrative_generation(limps_engine)
    
    # Save results
    save_integration_results(limps_engine, vectorizer, patterns, graph)
    
    println("\n🎉 === LiMps Integration Complete ===")
    println("🔥 The tapestry has been woven! 🔥")
    println("All three systems are now seamlessly integrated:")
    println("  • Motif Detection Engine → Message Vectorizer → LiMps Symbolic Memory")
    println("  • $(length(limps_engine.memory_entities)) memory entities stored")
    println("  • $(length(limps_engine.relationships)) relationships created")
    println("  • Graph density: $(round(graph["graph_density"], digits=4))")
    println("  • Average coherence: $(round(patterns["coherence_stats"]["mean"], digits=3))")
    println("\nThe symbolic memory tapestry is ready for exploration! 🕸️✨")
end