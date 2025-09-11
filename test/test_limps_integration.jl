using Test
using Pkg
Pkg.activate("..")

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

@testset "LiMps Integration Tests" begin
    
    @testset "LiMpsEngine Initialization" begin
        engine = LiMpsEngine()
        
        @test engine isa LiMpsEngine
        @test engine.coherence_threshold == 0.6
        @test engine.narrative_weaving_factor == 0.8
        @test engine.memory_decay_rate == 0.1
        @test engine.context_window_size == 10
        @test engine.max_memory_entities == 1000
        @test isempty(engine.memory_entities)
        @test isempty(engine.relationships)
        @test !isempty(engine.symbolic_variables)
    end
    
    @testset "Memory Entity Creation" begin
        # Test memory entity creation
        content = Dict{String, Any}("frequency" => 2, "confidence" => 0.8)
        context = ["isolation", "memory"]
        weight = 0.9
        
        # Create symbolic expression
        @variables m, c, p, t
        symbolic_expr = m + c + p + t
        
        entity = create_memory_entity("test_entity", "motif", content, symbolic_expr, weight, context)
        
        @test entity isa MemoryEntity
        @test entity.id == "test_entity"
        @test entity.type == "motif"
        @test entity.content == content
        @test entity.symbolic_expression == symbolic_expr
        @test entity.weight == weight
        @test entity.context == context
        @test entity.coherence_score > 0.0
        @test entity.narrative_importance > 0.0
        @test entity.timestamp > 0.0
    end
    
    @testset "Motif Memory Storage" begin
        engine = LiMpsEngine()
        
        # Create test motif data
        motif_data = Dict{String, Any}(
            "id" => "test_snake",
            "type" => "motif",
            "properties" => Dict{String, Any}(
                "frequency" => 3,
                "confidence" => 0.85,
                "weight" => 0.8
            ),
            "weight" => 0.8,
            "context" => ["snake", "nature", "symbolic"]
        )
        
        # Store motif memory
        memory_entity = store_motif_memory(engine, motif_data)
        
        @test memory_entity isa MemoryEntity
        @test haskey(engine.memory_entities, "test_snake")
        @test engine.memory_entities["test_snake"] == memory_entity
        @test memory_entity.id == "test_snake"
        @test memory_entity.type == "motif"
        @test memory_entity.weight == 0.8
        @test memory_entity.context == ["snake", "nature", "symbolic"]
    end
    
    @testset "Memory Relationships" begin
        engine = LiMpsEngine()
        
        # Store multiple motif memories
        motif1_data = Dict{String, Any}(
            "id" => "isolation_motif",
            "type" => "motif",
            "properties" => Dict{String, Any}("frequency" => 2),
            "weight" => 0.9,
            "context" => ["isolation", "memory"]
        )
        
        motif2_data = Dict{String, Any}(
            "id" => "snake_motif",
            "type" => "motif",
            "properties" => Dict{String, Any}("frequency" => 1),
            "weight" => 0.8,
            "context" => ["snake", "nature"]
        )
        
        motif3_data = Dict{String, Any}(
            "id" => "strand_motif",
            "type" => "motif",
            "properties" => Dict{String, Any}("frequency" => 1),
            "weight" => 0.7,
            "context" => ["strand", "technology"]
        )
        
        # Store memories
        entity1 = store_motif_memory(engine, motif1_data)
        entity2 = store_motif_memory(engine, motif2_data)
        entity3 = store_motif_memory(engine, motif3_data)
        
        @test length(engine.memory_entities) == 3
        @test length(engine.relationships) > 0
        
        # Test relationship types
        relationship_types = [rel.relationship_type for rel in engine.relationships]
        @test !isempty(relationship_types)
        @test all(rel_type in ["homogeneous", "isolated_connection", "memory_link", 
                              "identity_mirror", "symbolic_coil", "network_connection", "contextual"] 
                  for rel_type in relationship_types)
    end
    
    @testset "Contextual Memory Retrieval" begin
        engine = LiMpsEngine()
        
        # Store memories with different contexts
        memories_data = [
            Dict{String, Any}(
                "id" => "isolation_1",
                "type" => "motif",
                "properties" => Dict{String, Any}("frequency" => 2),
                "weight" => 0.9,
                "context" => ["isolation", "memory"]
            ),
            Dict{String, Any}(
                "id" => "snake_1",
                "type" => "motif",
                "properties" => Dict{String, Any}("frequency" => 1),
                "weight" => 0.8,
                "context" => ["snake", "nature"]
            ),
            Dict{String, Any}(
                "id" => "strand_1",
                "type" => "motif",
                "properties" => Dict{String, Any}("frequency" => 1),
                "weight" => 0.7,
                "context" => ["strand", "technology"]
            )
        ]
        
        for data in memories_data
            store_motif_memory(engine, data)
        end
        
        # Test retrieval with different contexts
        isolation_memories = retrieve_contextual_memories(engine, ["isolation"], limit = 5)
        @test !isempty(isolation_memories)
        @test any("isolation" in memory.context for memory in isolation_memories)
        
        snake_memories = retrieve_contextual_memories(engine, ["snake"], limit = 5)
        @test !isempty(snake_memories)
        @test any("snake" in memory.context for memory in snake_memories)
        
        # Test retrieval with multiple contexts
        multi_context_memories = retrieve_contextual_memories(engine, ["isolation", "memory"], limit = 5)
        @test !isempty(multi_context_memories)
    end
    
    @testset "Memory Tapestry Weaving" begin
        engine = LiMpsEngine()
        
        # Store a set of related memories
        related_memories = [
            Dict{String, Any}(
                "id" => "isolation_ep1",
                "type" => "motif",
                "properties" => Dict{String, Any}("frequency" => 2, "episode" => 1),
                "weight" => 0.9,
                "context" => ["isolation", "memory"]
            ),
            Dict{String, Any}(
                "id" => "snake_ep1",
                "type" => "motif",
                "properties" => Dict{String, Any}("frequency" => 1, "episode" => 1),
                "weight" => 0.8,
                "context" => ["snake", "nature"]
            ),
            Dict{String, Any}(
                "id" => "strand_ep1",
                "type" => "motif",
                "properties" => Dict{String, Any}("frequency" => 1, "episode" => 1),
                "weight" => 0.7,
                "context" => ["strand", "technology"]
            )
        ]
        
        for data in related_memories
            store_motif_memory(engine, data)
        end
        
        # Weave memory tapestry
        focus_context = ["isolation", "memory", "snake"]
        tapestry = weave_memory_tapestry(engine, focus_context)
        
        @test tapestry isa Dict{String, Any}
        @test haskey(tapestry, "symbolic_tapestry")
        @test haskey(tapestry, "relevant_memories")
        @test haskey(tapestry, "coherence_score")
        @test haskey(tapestry, "narrative_complexity")
        @test haskey(tapestry, "temporal_span")
        @test tapestry["relevant_memories"] > 0
        @test tapestry["coherence_score"] >= 0.0
        @test tapestry["coherence_score"] <= 1.0
    end
    
    @testset "Symbolic Narrative Generation" begin
        engine = LiMpsEngine()
        
        # Store memories for narrative generation
        narrative_memories = [
            Dict{String, Any}(
                "id" => "isolation_narr1",
                "type" => "motif",
                "properties" => Dict{String, Any}("frequency" => 3),
                "weight" => 0.9,
                "context" => ["isolation", "memory"]
            ),
            Dict{String, Any}(
                "id" => "snake_narr1",
                "type" => "motif",
                "properties" => Dict{String, Any}("frequency" => 2),
                "weight" => 0.8,
                "context" => ["snake", "nature"]
            ),
            Dict{String, Any}(
                "id" => "strand_narr1",
                "type" => "motif",
                "properties" => Dict{String, Any}("frequency" => 2),
                "weight" => 0.7,
                "context" => ["strand", "technology"]
            )
        ]
        
        for data in narrative_memories
            store_motif_memory(engine, data)
        end
        
        # Generate symbolic narrative
        focus_context = ["isolation", "snake", "strand"]
        narrative = generate_symbolic_narrative(engine, focus_context)
        
        @test narrative isa Dict{String, Any}
        @test haskey(narrative, "tapestry")
        @test haskey(narrative, "memories")
        @test haskey(narrative, "relationships")
        @test haskey(narrative, "symbolic_themes")
        @test haskey(narrative, "temporal_flow")
        
        # Test narrative components
        @test !isempty(narrative["memories"])
        @test narrative["temporal_flow"]["total_events"] > 0
        @test !isempty(narrative["temporal_flow"]["temporal_flow"])
    end
    
    @testset "Memory Graph Creation" begin
        engine = LiMpsEngine()
        
        # Store memories for graph creation
        graph_memories = [
            Dict{String, Any}(
                "id" => "node1",
                "type" => "motif",
                "properties" => Dict{String, Any}("frequency" => 2),
                "weight" => 0.9,
                "context" => ["isolation", "memory"]
            ),
            Dict{String, Any}(
                "id" => "node2",
                "type" => "motif",
                "properties" => Dict{String, Any}("frequency" => 1),
                "weight" => 0.8,
                "context" => ["snake", "nature"]
            ),
            Dict{String, Any}(
                "id" => "node3",
                "type" => "motif",
                "properties" => Dict{String, Any}("frequency" => 1),
                "weight" => 0.7,
                "context" => ["strand", "technology"]
            )
        ]
        
        for data in graph_memories
            store_motif_memory(engine, data)
        end
        
        # Create memory graph
        graph = create_memory_graph(engine)
        
        @test graph isa Dict{String, Any}
        @test haskey(graph, "nodes")
        @test haskey(graph, "edges")
        @test haskey(graph, "total_nodes")
        @test haskey(graph, "total_edges")
        @test haskey(graph, "graph_density")
        
        @test graph["total_nodes"] == 3
        @test graph["total_edges"] > 0
        @test graph["graph_density"] >= 0.0
        @test graph["graph_density"] <= 1.0
        
        # Test node structure
        for node in graph["nodes"]
            @test haskey(node, "id")
            @test haskey(node, "type")
            @test haskey(node, "weight")
            @test haskey(node, "context")
            @test haskey(node, "importance")
            @test haskey(node, "coherence")
        end
        
        # Test edge structure
        for edge in graph["edges"]
            @test haskey(edge, "source")
            @test haskey(edge, "target")
            @test haskey(edge, "type")
            @test haskey(edge, "strength")
            @test haskey(edge, "context_overlap")
        end
    end
    
    @testset "Memory Pattern Analysis" begin
        engine = LiMpsEngine()
        
        # Store diverse memories for pattern analysis
        diverse_memories = [
            Dict{String, Any}(
                "id" => "isolation_pattern1",
                "type" => "motif",
                "properties" => Dict{String, Any}("frequency" => 3),
                "weight" => 0.9,
                "context" => ["isolation", "memory"]
            ),
            Dict{String, Any}(
                "id" => "snake_pattern1",
                "type" => "motif",
                "properties" => Dict{String, Any}("frequency" => 2),
                "weight" => 0.8,
                "context" => ["snake", "nature"]
            ),
            Dict{String, Any}(
                "id" => "strand_pattern1",
                "type" => "motif",
                "properties" => Dict{String, Any}("frequency" => 2),
                "weight" => 0.7,
                "context" => ["strand", "technology"]
            ),
            Dict{String, Any}(
                "id" => "identity_pattern1",
                "type" => "motif",
                "properties" => Dict{String, Any}("frequency" => 1),
                "weight" => 0.9,
                "context" => ["identity", "memory"]
            )
        ]
        
        for data in diverse_memories
            store_motif_memory(engine, data)
        end
        
        # Analyze memory patterns
        patterns = analyze_memory_patterns(engine)
        
        @test patterns isa Dict{String, Any}
        @test haskey(patterns, "type_distribution")
        @test haskey(patterns, "context_distribution")
        @test haskey(patterns, "relationship_types")
        @test haskey(patterns, "coherence_stats")
        @test haskey(patterns, "importance_stats")
        @test haskey(patterns, "total_entities")
        @test haskey(patterns, "total_relationships")
        
        # Test statistics
        @test patterns["total_entities"] == 4
        @test patterns["total_relationships"] > 0
        
        # Test coherence stats
        coherence_stats = patterns["coherence_stats"]
        @test haskey(coherence_stats, "mean")
        @test haskey(coherence_stats, "std")
        @test haskey(coherence_stats, "min")
        @test haskey(coherence_stats, "max")
        @test coherence_stats["mean"] >= 0.0
        @test coherence_stats["mean"] <= 1.0
        
        # Test importance stats
        importance_stats = patterns["importance_stats"]
        @test haskey(importance_stats, "mean")
        @test haskey(importance_stats, "std")
        @test haskey(importance_stats, "min")
        @test haskey(importance_stats, "max")
        @test importance_stats["mean"] >= 0.0
        @test importance_stats["mean"] <= 1.0
    end
    
    @testset "LiMps Data Export" begin
        engine = LiMpsEngine()
        
        # Store test memories
        export_memories = [
            Dict{String, Any}(
                "id" => "export_test1",
                "type" => "motif",
                "properties" => Dict{String, Any}("frequency" => 2),
                "weight" => 0.9,
                "context" => ["isolation", "memory"]
            ),
            Dict{String, Any}(
                "id" => "export_test2",
                "type" => "motif",
                "properties" => Dict{String, Any}("frequency" => 1),
                "weight" => 0.8,
                "context" => ["snake", "nature"]
            )
        ]
        
        for data in export_memories
            store_motif_memory(engine, data)
        end
        
        # Export LiMps data
        export_data = export_limps_data(engine)
        
        @test export_data isa Dict{String, Any}
        @test haskey(export_data, "memory_entities")
        @test haskey(export_data, "relationships")
        @test haskey(export_data, "engine_config")
        @test haskey(export_data, "metadata")
        
        # Test metadata
        metadata = export_data["metadata"]
        @test haskey(metadata, "total_entities")
        @test haskey(metadata, "total_relationships")
        @test haskey(metadata, "export_timestamp")
        @test haskey(metadata, "version")
        @test metadata["total_entities"] == 2
        @test metadata["version"] == "1.0.0"
        
        # Test engine config
        engine_config = export_data["engine_config"]
        @test haskey(engine_config, "coherence_threshold")
        @test haskey(engine_config, "narrative_weaving_factor")
        @test haskey(engine_config, "memory_decay_rate")
        @test haskey(engine_config, "context_window_size")
        @test haskey(engine_config, "max_memory_entities")
    end
    
    @testset "Integration Pipeline" begin
        # Test complete integration pipeline
        engine = LiMpsEngine()
        vectorizer = MessageVectorizer(32)
        
        # Create test text
        test_text = "In the digital desert, Snake found himself alone, disconnected from the network. The snake coiled around his memories."
        
        # Step 1: Motif Detection
        analysis = parse_document(test_text)
        motif_tokens = extract_motif_tokens(analysis)
        @test !isempty(motif_tokens)
        
        # Step 2: Message Vectorization
        for token in motif_tokens
            add_motif_embedding!(vectorizer, token)
        end
        message_state = vectorize_message(motif_tokens, vectorizer)
        @test message_state isa MessageState
        
        # Step 3: LiMps Storage
        for token in motif_tokens
            motif_data = Dict{String, Any}(
                "id" => string(token.name),
                "type" => "motif",
                "properties" => Dict{String, Any}(
                    "frequency" => token.properties[:frequency],
                    "confidence" => token.properties[:confidence],
                    "weight" => token.weight,
                    "symbolic_expression" => string(message_state.symbolic_expression)
                ),
                "weight" => token.weight,
                "context" => [string(ctx) for ctx in token.context]
            )
            
            memory_entity = store_motif_memory(engine, motif_data)
            @test memory_entity isa MemoryEntity
        end
        
        # Step 4: Memory Retrieval and Narrative Generation
        memories = retrieve_contextual_memories(engine, ["isolation", "snake"], limit = 5)
        @test !isempty(memories)
        
        narrative = generate_symbolic_narrative(engine, ["isolation", "snake"])
        @test narrative isa Dict{String, Any}
        @test haskey(narrative, "tapestry")
        @test haskey(narrative, "memories")
        
        # Step 5: Export
        export_data = export_limps_data(engine)
        @test export_data isa Dict{String, Any}
        @test export_data["metadata"]["total_entities"] > 0
    end
    
    @testset "Edge Cases" begin
        engine = LiMpsEngine()
        
        # Test empty context retrieval
        empty_memories = retrieve_contextual_memories(engine, String[], limit = 5)
        @test isempty(empty_memories)
        
        # Test single memory tapestry
        single_memory_data = Dict{String, Any}(
            "id" => "single_test",
            "type" => "motif",
            "properties" => Dict{String, Any}("frequency" => 1),
            "weight" => 0.8,
            "context" => ["isolation"]
        )
        store_motif_memory(engine, single_memory_data)
        
        tapestry = weave_memory_tapestry(engine, ["isolation"])
        @test tapestry["relevant_memories"] == 1
        @test tapestry["coherence_score"] == 1.0  # Single memory has perfect coherence
        
        # Test memory with no relationships
        isolated_memory_data = Dict{String, Any}(
            "id" => "isolated_test",
            "type" => "motif",
            "properties" => Dict{String, Any}("frequency" => 1),
            "weight" => 0.5,
            "context" => ["unique_context"]
        )
        store_motif_memory(engine, isolated_memory_data)
        
        # Test pattern analysis with minimal data
        patterns = analyze_memory_patterns(engine)
        @test patterns["total_entities"] == 2
        @test patterns["total_relationships"] >= 0
    end
    
    @testset "Performance Tests" begin
        engine = LiMpsEngine()
        
        # Test with larger dataset
        large_memories = []
        for i in 1:50
            push!(large_memories, Dict{String, Any}(
                "id" => "perf_test_$i",
                "type" => "motif",
                "properties" => Dict{String, Any}("frequency" => rand(1:5)),
                "weight" => rand() * 0.5 + 0.5,  # 0.5 to 1.0
                "context" => rand(["isolation", "snake", "strand", "memory", "identity"], rand(1:3))
            ))
        end
        
        # Store memories
        start_time = time()
        for data in large_memories
            store_motif_memory(engine, data)
        end
        storage_time = time() - start_time
        
        @test storage_time < 5.0  # Should store 50 memories in under 5 seconds
        
        # Test retrieval performance
        start_time = time()
        memories = retrieve_contextual_memories(engine, ["isolation", "memory"], limit = 20)
        retrieval_time = time() - start_time
        
        @test retrieval_time < 1.0  # Should retrieve in under 1 second
        
        # Test narrative generation performance
        start_time = time()
        narrative = generate_symbolic_narrative(engine, ["isolation", "snake", "strand"])
        narrative_time = time() - start_time
        
        @test narrative_time < 2.0  # Should generate narrative in under 2 seconds
        
        # Test graph creation performance
        start_time = time()
        graph = create_memory_graph(engine)
        graph_time = time() - start_time
        
        @test graph_time < 1.0  # Should create graph in under 1 second
    end
end

println("All LiMps Integration tests passed!")