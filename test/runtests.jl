using Test
using Pkg
Pkg.activate("..")

using MessageVectorizer
using LinearAlgebra

@testset "MessageVectorizer Tests" begin
    
    @testset "MotifToken Creation" begin
        motif = MotifToken(
            :test_motif,
            Dict{Symbol, Any}(:intensity => 0.5, :duration => 10.0),
            0.8,
            [:temporal, :spatial]
        )
        
        @test motif.name == :test_motif
        @test motif.weight == 0.8
        @test length(motif.context) == 2
        @test motif.properties[:intensity] == 0.5
    end
    
    @testset "MessageVectorizer Initialization" begin
        vectorizer = MessageVectorizer(64)
        
        @test vectorizer.embedding_dim == 64
        @test vectorizer.entropy_threshold == 0.5
        @test vectorizer.compression_ratio == 0.8
        @test isempty(vectorizer.motif_embeddings)
        @test length(vectorizer.symbolic_variables) == 4
    end
    
    @testset "Motif Embedding Creation" begin
        motif = MotifToken(
            :test_embedding,
            Dict{Symbol, Any}(
                :intensity => 0.7,
                :duration => 15.0,
                :type => "temporal"
            ),
            0.9,
            [:context1, :context2]
        )
        
        embedding = create_motif_embedding(motif, 32)
        
        @test length(embedding) == 32
        @test norm(embedding) ≈ 1.0 atol=1e-6
        @test all(isfinite, embedding)
    end
    
    @testset "Symbolic State Compression" begin
        vectorizer = MessageVectorizer(16)
        
        motifs = [
            MotifToken(:motif1, Dict{Symbol, Any}(:val => 0.5), 0.6, [:ctx1]),
            MotifToken(:motif2, Dict{Symbol, Any}(:val => 0.3), 0.4, [:ctx2])
        ]
        
        # Add embeddings
        for motif in motifs
            add_motif_embedding!(vectorizer, motif)
        end
        
        symbolic_state = symbolic_state_compression(motifs, vectorizer)
        
        @test symbolic_state isa Num
        @test !iszero(symbolic_state)
    end
    
    @testset "Message Vectorization" begin
        vectorizer = MessageVectorizer(16)
        
        motifs = [
            MotifToken(:test1, Dict{Symbol, Any}(:intensity => 0.8), 0.7, [:temporal]),
            MotifToken(:test2, Dict{Symbol, Any}(:connectivity => 0.6), 0.5, [:spatial])
        ]
        
        # Add embeddings
        for motif in motifs
            add_motif_embedding!(vectorizer, motif)
        end
        
        message_state = vectorize_message(motifs, vectorizer)
        
        @test message_state isa MessageState
        @test length(message_state.vector_representation) == 16
        @test message_state.entropy_score >= 0.0
        @test length(message_state.motif_configuration) == 2
        @test haskey(message_state.metadata, "num_motifs")
    end
    
    @testset "Entropy Computation" begin
        # Test with uniform distribution
        uniform_vector = ones(8)
        motif_config = Dict{Symbol, Float64}(:test => 1.0)
        
        entropy = compute_entropy(uniform_vector, motif_config)
        
        @test entropy > 0.0
        @test isfinite(entropy)
        
        # Test with zero vector
        zero_vector = zeros(8)
        entropy_zero = compute_entropy(zero_vector, motif_config)
        
        @test entropy_zero == 0.0
    end
    
    @testset "al-ULS Interface" begin
        vectorizer = MessageVectorizer(8)
        
        motif = MotifToken(:test_uls, Dict{Symbol, Any}(:val => 0.5), 0.8, [:ctx])
        add_motif_embedding!(vectorizer, motif)
        
        message_state = vectorize_message([motif], vectorizer)
        uls_output = al_uls_interface(message_state)
        
        @test uls_output isa Dict{String, Any}
        @test haskey(uls_output, "symbolic_expression")
        @test haskey(uls_output, "vector_representation")
        @test haskey(uls_output, "entropy_score")
        @test haskey(uls_output, "compressed_size")
        @test haskey(uls_output, "information_density")
        @test uls_output["compressed_size"] == 8
        @test uls_output["information_density"] >= 0.0
    end
    
    @testset "Complex Motif Configurations" begin
        vectorizer = MessageVectorizer(32)
        
        # Create complex motif configuration
        complex_motifs = [
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
            MotifToken(
                :decay_memory,
                Dict{Symbol, Any}(
                    :decay_rate => 0.3,
                    :memory_strength => 0.6,
                    :forgetting_curve => "exponential"
                ),
                0.6,
                [:cognitive, :temporal, :neural]
            )
        ]
        
        # Add embeddings
        for motif in complex_motifs
            add_motif_embedding!(vectorizer, motif)
        end
        
        # Test vectorization
        state = vectorize_message(complex_motifs, vectorizer)
        
        @test state isa MessageState
        @test length(state.vector_representation) == 32
        @test state.entropy_score > 0.0
        @test length(state.motif_configuration) == 2
        
        # Test that embeddings are properly stored
        @test haskey(vectorizer.motif_embeddings, :isolation_time)
        @test haskey(vectorizer.motif_embeddings, :decay_memory)
        @test length(vectorizer.motif_embeddings[:isolation_time]) == 32
        @test length(vectorizer.motif_embeddings[:decay_memory]) == 32
    end
    
    @testset "Edge Cases" begin
        vectorizer = MessageVectorizer(4)
        
        # Test with empty motif list
        empty_state = vectorize_message(MotifToken[], vectorizer)
        @test empty_state isa MessageState
        @test all(iszero, empty_state.vector_representation)
        @test empty_state.entropy_score == 0.0
        
        # Test with single motif
        single_motif = [MotifToken(:single, Dict{Symbol, Any}(:val => 1.0), 1.0, [:ctx])]
        add_motif_embedding!(vectorizer, single_motif[1])
        
        single_state = vectorize_message(single_motif, vectorizer)
        @test single_state isa MessageState
        @test length(single_state.motif_configuration) == 1
    end
end

println("All tests passed!")