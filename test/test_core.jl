#!/usr/bin/env julia

"""
Test suite for the refactored MotifAnalysis system.
"""

using Test
using Logging

# Suppress info messages during testing
global_logger(ConsoleLogger(stderr, Logging.Error))

# Include the main module
include("../src/MotifAnalysis.jl")
using .MotifAnalysis

@testset "Core Types" begin
    @testset "MotifType enum" begin
        @test length(instances(MotifType)) == 6
        @test ISOLATION in instances(MotifType)
        @test SNAKE in instances(MotifType)
        @test string(ISOLATION) == "ISOLATION"
    end
    
    @testset "MotifToken creation" begin
        token = MotifToken(ISOLATION, 0.5, ["alone"], 0.3, "He stood alone", 0.8)
        @test token.type == ISOLATION
        @test token.weight == 0.5
        @test token.confidence == 0.8
        @test validate_motif_token(token)
    end
    
    @testset "MotifToken validation" begin
        # Invalid weight
        @test_throws AssertionError MotifToken(ISOLATION, 1.5, ["alone"], 0.3, "context", 0.8)
        # Invalid position
        @test_throws AssertionError MotifToken(ISOLATION, 0.5, ["alone"], 1.5, "context", 0.8)
        # Invalid confidence
        @test_throws AssertionError MotifToken(ISOLATION, 0.5, ["alone"], 0.3, "context", 1.5)
    end
end

@testset "Text Processing" begin
    @testset "Text preprocessing" begin
        text = "He stood alone in the desert, watching the snake coil."
        tokens = preprocess_text(text)
        @test !isempty(tokens)
        @test "alone" in tokens
        @test "snake" in tokens
        @test !("the" in tokens)  # Stop word should be removed
    end
    
    @testset "Sentence extraction" begin
        text = "First sentence. Second sentence. Third sentence."
        sentences = extract_sentences(text)
        @test length(sentences) == 3
        @test all(s -> length(s) > 10, sentences)
    end
    
    @testset "Motif pattern matching" begin
        tokens = ["alone", "snake", "memory", "time"]
        isolation_matches = find_motif_matches(tokens, ISOLATION)
        snake_matches = find_motif_matches(tokens, SNAKE)
        
        @test "alone" in isolation_matches
        @test "snake" in snake_matches
        @test isempty(find_motif_matches(tokens, FRAGMENTATION))
    end
end

@testset "Motif Detection" begin
    @testset "Basic motif detection" begin
        text = "He stood alone in the desert, watching the snake coil around the strand of memory."
        tokens = detect_motifs(text)
        
        @test !isempty(tokens)
        @test any(t -> t.type == ISOLATION, tokens)
        @test any(t -> t.type == SNAKE, tokens)
        @test any(t -> t.type == STRAND, tokens)
        @test any(t -> t.type == MEMORY, tokens)
    end
    
    @testset "Motif detection by type" begin
        text = "The snake coils in silence, a phantom memory."
        isolation_tokens = detect_motifs_by_type(text, ISOLATION)
        snake_tokens = detect_motifs_by_type(text, SNAKE)
        
        @test all(t -> t.type == ISOLATION, isolation_tokens)
        @test all(t -> t.type == SNAKE, snake_tokens)
    end
    
    @testset "Dominant motif identification" begin
        text = "Alone in solitude, the snake coils around broken fragments of memory."
        tokens = detect_motifs(text)
        dominant = get_dominant_motifs(tokens, 0.3)
        
        @test !isempty(dominant)
        @test all(m -> m in instances(MotifType), dominant)
    end
    
    @testset "Motif statistics" begin
        text = "Alone snake memory strand temporal fragmentation."
        tokens = detect_motifs(text)
        stats = calculate_motif_statistics(tokens)
        
        @test haskey(stats, "total_tokens")
        @test haskey(stats, "unique_types")
        @test haskey(stats, "average_weight")
        @test stats["total_tokens"] > 0
    end
end

@testset "Vectorization" begin
    @testset "Basic vectorization" begin
        text = "Alone snake memory."
        tokens = detect_motifs(text)
        vector = vectorize_motifs(tokens)
        
        @test length(vector) == 6
        @test norm(vector) > 0
        @test all(x -> x >= 0, vector)
    end
    
    @testset "Entropy calculation" begin
        # Test uniform distribution
        uniform_vector = [0.25, 0.25, 0.25, 0.25, 0.0, 0.0]
        entropy = calculate_entropy(uniform_vector)
        @test entropy > 0
        
        # Test single peak
        peak_vector = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        entropy = calculate_entropy(peak_vector)
        @test entropy == 0.0
    end
    
    @testset "Coherence calculation" begin
        text = "Alone snake memory strand."
        tokens = detect_motifs(text)
        coherence = calculate_coherence_score(tokens)
        
        @test 0.0 <= coherence <= 1.0
    end
    
    @testset "Vectorized message creation" begin
        text = "Alone in the desert, snake coils around memory."
        tokens = detect_motifs(text)
        message = create_vectorized_message(tokens, text)
        
        @test length(message.vector) == 6
        @test message.entropy >= 0.0
        @test 0.0 <= message.coherence_score <= 1.0
        @test 0.0 <= message.motif_density <= 1.0
    end
end

@testset "Analysis Pipeline" begin
    @testset "Complete analysis" begin
        text = "He stood alone in the desert, watching the snake coil around the strand of memory."
        result = analyze_text(text)
        
        @test result isa AnalysisResult
        @test !isempty(result.tokens)
        @test result.vectorized_message isa VectorizedMessage
        @test 0.0 <= result.narrative_coherence <= 1.0
        @test haskey(result.metadata, "statistics")
    end
    
    @testset "Empty text handling" begin
        result = analyze_text("")
        @test result isa AnalysisResult
        @test isempty(result.tokens)
        @test result.narrative_coherence == 1.0
    end
    
    @testset "Batch analysis" begin
        texts = [
            "Alone in the desert.",
            "Snake coils around memory.",
            "Fragments of time scattered."
        ]
        results = analyze_text_batch(texts)
        
        @test length(results) == 3
        @test all(r -> r isa AnalysisResult, results)
    end
    
    @testset "Analysis comparison" begin
        text1 = "Alone snake memory."
        text2 = "Alone snake memory strand."
        
        result1 = analyze_text(text1)
        result2 = analyze_text(text2)
        
        comparison = compare_analyses(result1, result2)
        
        @test haskey(comparison, "vector_similarity")
        @test haskey(comparison, "overall_similarity")
        @test 0.0 <= comparison["overall_similarity"] <= 1.0
    end
end

@testset "API Functions" begin
    @testset "Version and configuration" begin
        @test version() == "2.0.0"
        @test length(supported_motif_types()) == 6
    end
    
    @testset "Quick analysis" begin
        text = "Alone snake memory strand temporal fragmentation."
        quick_result = quick_analysis(text)
        
        @test haskey(quick_result, "dominant_motifs")
        @test haskey(quick_result, "coherence_score")
        @test haskey(quick_result, "motif_count")
    end
    
    @testset "Motif distribution analysis" begin
        texts = [
            "Alone in solitude.",
            "Snake coils around memory.",
            "Fragments of time."
        ]
        distribution = analyze_motif_distribution(texts)
        
        @test distribution["total_texts"] == 3
        @test haskey(distribution, "motif_frequency")
        @test haskey(distribution, "average_coherence")
    end
end

@testset "Error Handling" begin
    @testset "Invalid inputs" begin
        # Test with very long text
        long_text = repeat("Alone snake memory. ", 1000)
        result = analyze_text(long_text)
        @test result isa AnalysisResult
        
        # Test with special characters
        special_text = "Alone 🐍 memory @#$%^&*()"
        result = analyze_text(special_text)
        @test result isa AnalysisResult
    end
end

println("✅ All tests passed!")