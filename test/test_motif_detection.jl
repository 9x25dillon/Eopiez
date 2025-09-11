using Test
using Pkg
Pkg.activate("..")

# Include motif detection modules
include("../src/motif_detection/motifs.jl")
include("../src/motif_detection/parser.jl")
include("../src/motif_detection/motif_server.jl")

using .MotifDefinitions
using .MotifParser
using .MotifServer

@testset "Motif Detection Engine Tests" begin
    
    @testset "MotifDefinitions Tests" begin
        @test length(MOTIF_RULES) > 0
        @test haskey(MOTIF_RULES, "isolation")
        @test haskey(MOTIF_RULES, "snake")
        @test haskey(MOTIF_RULES, "strand")
        @test haskey(MOTIF_RULES, "memory")
        
        @test length(MOTIF_WEIGHTS) == length(MOTIF_RULES)
        @test all(0.0 <= w <= 1.0 for w in values(MOTIF_WEIGHTS))
        
        @test length(MOTIF_CONTEXTS) > 0
        @test haskey(MOTIF_CONTEXTS, "isolation")
    end
    
    @testset "Motif Detection Tests" begin
        # Test basic motif detection
        test_text = "The snake coiled in isolation, its memory fading like strands of DNA."
        tokens = detect_motifs(test_text)
        
        @test haskey(tokens, "snake")
        @test haskey(tokens, "isolation")
        @test haskey(tokens, "memory")
        @test haskey(tokens, "strand")
        
        # Test confidence calculation
        confidence = calculate_motif_confidence(tokens, length(test_text))
        @test all(0.0 <= c <= 1.0 for c in values(confidence))
    end
    
    @testset "Document Parsing Tests" begin
        test_text = "In the digital desert, Snake found himself alone. The snake coiled around his memories."
        
        # Test document parsing
        analysis = parse_document(test_text)
        
        @test analysis isa DocumentAnalysis
        @test analysis.text == lowercase(test_text)
        @test !isempty(analysis.motif_tokens)
        @test !isempty(analysis.confidence_scores)
        @test !isempty(analysis.document_metrics)
        @test analysis.timestamp > 0
        
        # Test document metrics
        metrics = analysis.document_metrics
        @test metrics["text_length"] > 0
        @test metrics["word_count"] > 0
        @test metrics["motif_density"] >= 0.0
        @test metrics["unique_motifs"] >= 0
    end
    
    @testset "Motif Token Extraction Tests" begin
        test_text = "The snake and isolation created a web of memory strands."
        analysis = parse_document(test_text)
        
        motif_tokens = extract_motif_tokens(analysis)
        
        @test motif_tokens isa Vector{MotifToken}
        @test length(motif_tokens) > 0
        
        for token in motif_tokens
            @test token isa MotifToken
            @test !isempty(token.name)
            @test token.weight >= 0.0
            @test token.weight <= 1.0
            @test !isempty(token.properties)
        end
    end
    
    @testset "Document Structure Analysis Tests" begin
        test_text = "Snake found isolation in the digital desert. Memory strands connected everything."
        
        # Create mock motif tokens
        motif_tokens = Dict{String, Vector{String}}(
            "snake" => ["snake"],
            "isolation" => ["isolation"],
            "memory" => ["memory"],
            "strand" => ["strands"],
            "technology" => ["digital"]
        )
        
        metrics = analyze_document_structure(test_text, motif_tokens)
        
        @test metrics["text_length"] == length(test_text)
        @test metrics["word_count"] > 0
        @test metrics["motif_density"] > 0.0
        @test metrics["unique_motifs"] == 5
        @test haskey(metrics, "motif_entropy")
        @test haskey(metrics, "avg_word_length")
        @test haskey(metrics, "vocabulary_size")
    end
    
    @testset "Motif Relationships Tests" begin
        # Create mock data
        motif_tokens = Dict{String, Vector{String}}(
            "isolation" => ["isolation"],
            "technology" => ["digital"],
            "snake" => ["snake"],
            "memory" => ["memory"]
        )
        
        confidence_scores = Dict{String, Float64}(
            "isolation" => 0.8,
            "technology" => 0.6,
            "snake" => 0.7,
            "memory" => 0.5
        )
        
        relationships = find_motif_relationships(motif_tokens, confidence_scores)
        
        @test relationships isa Dict{String, Vector{String}}
        @test !isempty(relationships)
    end
    
    @testset "Metrics Calculation Tests" begin
        # Create mock analysis
        test_text = "Snake in isolation with digital memories."
        analysis = parse_document(test_text)
        
        # Test without benchmarks
        metrics = calculate_motif_metrics(analysis)
        
        @test haskey(metrics, "total_detected_motifs")
        @test haskey(metrics, "total_motif_occurrences")
        @test haskey(metrics, "avg_confidence")
        @test haskey(metrics, "max_confidence")
        @test haskey(metrics, "min_confidence")
        @test haskey(metrics, "confidence_std")
        
        # Test with benchmarks
        manual_benchmarks = Dict{String, Vector{String}}(
            "snake" => ["snake"],
            "isolation" => ["isolation"],
            "memory" => ["memories"]
        )
        
        metrics_with_benchmarks = calculate_motif_metrics(analysis, manual_benchmarks=manual_benchmarks)
        
        @test haskey(metrics_with_benchmarks, "avg_precision")
        @test haskey(metrics_with_benchmarks, "avg_recall")
        @test haskey(metrics_with_benchmarks, "f1_score")
    end
    
    @testset "Report Generation Tests" begin
        test_text = "Digital snake in isolated memory strands."
        analysis = parse_document(test_text)
        
        report = create_motif_report(analysis)
        
        @test haskey(report, "document_info")
        @test haskey(report, "detected_motifs")
        @test haskey(report, "summary")
        @test haskey(report["document_info"], "text_length")
        @test haskey(report["document_info"], "word_count")
        @test haskey(report["summary"], "total_motifs_detected")
        @test haskey(report["summary"], "total_occurrences")
    end
    
    @testset "Batch Processing Tests" begin
        texts = [
            "Snake in isolation.",
            "Digital memories and strands.",
            "Technology and communication."
        ]
        
        analyses = batch_parse_documents(texts)
        
        @test length(analyses) == length(texts)
        @test all(a isa DocumentAnalysis for a in analyses)
    end
    
    @testset "LiMps Integration Tests" begin
        # Create mock motif tokens
        motif_tokens = [
            MotifToken(:snake, Dict{Symbol, Any}(:frequency => 2), 0.8, [:nature, :symbolic]),
            MotifToken(:isolation, Dict{Symbol, Any}(:frequency => 1), 0.9, [:emotional, :spatial])
        ]
        
        limps_data = create_limps_integration(motif_tokens)
        
        @test haskey(limps_data, "motif_entities")
        @test haskey(limps_data, "relationships")
        @test haskey(limps_data, "metadata")
        @test length(limps_data["motif_entities"]) == length(motif_tokens)
        @test limps_data["metadata"]["total_motifs"] == length(motif_tokens)
    end
    
    @testset "HTTP API Simulation Tests" begin
        # Test request structure
        sample_request = Dict{String, Any}(
            "text" => "Snake in digital isolation",
            "custom_rules" => nothing,
            "weights" => nothing
        )
        
        @test haskey(sample_request, "text")
        @test !isempty(sample_request["text"])
        
        # Test response structure (simulated)
        analysis = parse_document(sample_request["text"])
        motif_tokens = extract_motif_tokens(analysis)
        
        response = Dict{String, Any}(
            "status" => "success",
            "document_analysis" => create_motif_report(analysis),
            "motif_tokens" => [
                Dict{String, Any}(
                    "name" => string(token.name),
                    "properties" => token.properties,
                    "weight" => token.weight,
                    "context" => [string(ctx) for ctx in token.context]
                ) for token in motif_tokens
            ],
            "metrics" => calculate_motif_metrics(analysis),
            "timestamp" => time()
        )
        
        @test response["status"] == "success"
        @test haskey(response, "document_analysis")
        @test haskey(response, "motif_tokens")
        @test haskey(response, "metrics")
        @test haskey(response, "timestamp")
    end
    
    @testset "Edge Cases Tests" begin
        # Test empty text
        empty_analysis = parse_document("")
        @test empty_analysis isa DocumentAnalysis
        @test isempty(empty_analysis.motif_tokens)
        
        # Test text with no motifs
        no_motif_text = "This is a simple text without any specific motifs."
        no_motif_analysis = parse_document(no_motif_text)
        @test no_motif_analysis isa DocumentAnalysis
        
        # Test very long text
        long_text = repeat("Snake in isolation with digital memories. ", 100)
        long_analysis = parse_document(long_text)
        @test long_analysis isa DocumentAnalysis
        @test long_analysis.document_metrics["text_length"] > 1000
    end
    
    @testset "Performance Tests" begin
        # Test processing time for reasonable text
        test_text = repeat("Snake in isolation with digital memories and strands. ", 50)
        
        start_time = time()
        analysis = parse_document(test_text)
        end_time = time()
        
        processing_time = end_time - start_time
        @test processing_time < 1.0  # Should process in under 1 second
        
        # Test memory usage (basic check)
        @test analysis isa DocumentAnalysis
        @test !isempty(analysis.motif_tokens)
    end
end

println("All Motif Detection Engine tests passed!")