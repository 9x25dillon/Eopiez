using Pkg
Pkg.activate(".")

using JSON3
using Printf

# Include motif detection modules
include("../src/motif_detection/motifs.jl")
include("../src/motif_detection/parser.jl")
include("../src/motif_detection/motif_server.jl")

using .MotifDefinitions
using .MotifParser
using .MotifServer

"""
    create_sample_texts()

Create sample Kojima-esque texts for demonstration.
"""
function create_sample_texts()
    texts = [
        # Sample 1: Isolation and Technology
        """
        In the vast digital desert, Snake found himself alone, disconnected from the network. 
        The silence was absolute, broken only by the hiss of static from his radio. 
        He remembered the past, the memories of his comrades now lost in the void. 
        The snake coiled around his thoughts, a symbol of the endless cycle of war and peace.
        """,
        
        # Sample 2: Memory and Identity
        """
        The strands of memory connected him to his past, like DNA encoding his identity. 
        He recalled the flashback of his childhood, the snake that had taught him about survival. 
        His consciousness was fragmented, pieces of himself scattered across the digital landscape. 
        The ouroboros of his existence continued its eternal dance.
        """,
        
        # Sample 3: Communication and War
        """
        The battlefield was a web of connections, each soldier a node in the network. 
        Messages flowed through the air like invisible strands, carrying information and orders. 
        The snake of war slithered through the jungle, its venom spreading chaos and destruction. 
        In the isolation of combat, each fighter became a ghost, a phantom of their former self.
        """,
        
        # Sample 4: Nature and Transformation
        """
        The forest whispered with ancient memories, its roots entwined like strands of DNA. 
        The snake emerged from the shadows, its scales reflecting the moonlight. 
        Technology and nature clashed in this sacred space, where the digital and organic merged. 
        The transformation was complete - man had become machine, and machine had become nature.
        """
    ]
    
    return texts
end

"""
    demonstrate_motif_detection()

Demonstrate the motif detection capabilities.
"""
function demonstrate_motif_detection()
    println("=== Motif Detection Engine Demonstration ===\n")
    
    # Create sample texts
    texts = create_sample_texts()
    println("✓ Created $(length(texts)) sample Kojima-esque texts")
    
    # Analyze each text
    for (i, text) in enumerate(texts)
        println("\n--- Sample Text $i ---")
        println("Text: $(text[1:100])...")
        
        # Parse document
        analysis = parse_document(text)
        
        # Create report
        report = create_motif_report(analysis)
        
        # Display results
        println("Document Analysis:")
        println("  - Words: $(report["document_info"]["word_count"])")
        println("  - Motif density: $(round(report["summary"]["motif_density"], digits=4))")
        println("  - Total motifs: $(report["summary"]["total_motifs_detected"])")
        println("  - Avg confidence: $(round(report["summary"]["avg_confidence"], digits=3))")
        
        # Show detected motifs
        println("Detected Motifs:")
        for (motif, data) in report["detected_motifs"]
            println("  - $(motif): $(data["count"]) occurrences, confidence $(round(data["confidence"], digits=3))")
            println("    Tokens: $(join(data["tokens"][1:min(3, length(data["tokens"]))], ", "))")
        end
        
        # Show relationships
        if !isempty(report["motif_relationships"])
            println("Motif Relationships:")
            for (motif, related) in report["motif_relationships"]
                println("  - $(motif) ↔ $(join(related, ", "))")
            end
        end
    end
    
    return texts
end

"""
    demonstrate_batch_processing()

Demonstrate batch processing capabilities.
"""
function demonstrate_batch_processing()
    println("\n=== Batch Processing Demonstration ===\n")
    
    texts = create_sample_texts()
    
    # Process all texts in batch
    analyses = batch_parse_documents(texts)
    
    println("Batch Analysis Results:")
    println("  - Total documents: $(length(analyses))")
    
    # Calculate batch metrics
    total_motifs = sum([a.document_metrics["unique_motifs"] for a in analyses])
    avg_density = mean([a.document_metrics["motif_density"] for a in analyses])
    avg_confidence = mean([mean(values(a.confidence_scores)) for a in analyses])
    
    println("  - Total unique motifs: $total_motifs")
    println("  - Average motif density: $(round(avg_density, digits=4))")
    println("  - Average confidence: $(round(avg_confidence, digits=3))")
    
    # Show motif distribution across all documents
    println("\nMotif Distribution Across Documents:")
    all_motifs = Set{String}()
    for analysis in analyses
        for motif in keys(analysis.motif_tokens)
            if !isempty(analysis.motif_tokens[motif])
                push!(all_motifs, motif)
            end
        end
    end
    
    for motif in sort(collect(all_motifs))
        count = sum([!isempty(a.motif_tokens[motif]) for a in analyses if haskey(a.motif_tokens, motif)])
        println("  - $(motif): present in $(count)/$(length(analyses)) documents")
    end
    
    return analyses
end

"""
    demonstrate_vectorization_integration()

Demonstrate integration with Message Vectorizer.
"""
function demonstrate_vectorization_integration()
    println("\n=== Message Vectorizer Integration ===\n")
    
    # Use the first sample text
    text = create_sample_texts()[1]
    analysis = parse_document(text)
    
    # Extract motif tokens
    motif_tokens = extract_motif_tokens(analysis)
    println("✓ Extracted $(length(motif_tokens)) motif tokens")
    
    # Show motif tokens
    for token in motif_tokens
        println("Motif: $(token.name)")
        println("  - Weight: $(round(token.weight, digits=3))")
        println("  - Context: $(join([string(ctx) for ctx in token.context], ", "))")
        println("  - Properties: $(token.properties)")
    end
    
    # Create LiMps integration data
    limps_data = create_limps_integration(motif_tokens)
    
    println("\nLiMps Integration Data:")
    println("  - Motif entities: $(length(limps_data["motif_entities"]))")
    println("  - Relationships: $(length(limps_data["relationships"]))")
    
    # Show relationships
    for rel in limps_data["relationships"]
        println("  - $(rel["source"]) → $(rel["target"]) (strength: $(round(rel["strength"], digits=3)))")
    end
    
    return motif_tokens, limps_data
end

"""
    demonstrate_metrics_calculation()

Demonstrate precision/recall metrics calculation.
"""
function demonstrate_metrics_calculation()
    println("\n=== Metrics Calculation Demonstration ===\n")
    
    # Create manual benchmarks for testing
    manual_benchmarks = Dict{String, Vector{String}}(
        "isolation" => ["alone", "disconnected", "silence", "void"],
        "snake" => ["snake", "coiled", "hiss"],
        "memory" => ["remembered", "memories", "past", "flashback"],
        "strand" => ["strands", "connections", "network"],
        "technology" => ["digital", "network", "radio"],
        "identity" => ["identity", "consciousness", "self"],
        "communication" => ["radio", "messages", "network"],
        "war" => ["battlefield", "soldier", "combat"],
        "nature" => ["forest", "jungle", "nature"]
    )
    
    # Analyze sample text
    text = create_sample_texts()[1]
    analysis = parse_document(text)
    
    # Calculate metrics
    metrics = calculate_motif_metrics(analysis, manual_benchmarks=manual_benchmarks)
    
    println("Performance Metrics:")
    println("  - Total detected motifs: $(metrics["total_detected_motifs"])")
    println("  - Total occurrences: $(metrics["total_motif_occurrences"])")
    println("  - Average confidence: $(round(metrics["avg_confidence"], digits=3))")
    println("  - Confidence std: $(round(metrics["confidence_std"], digits=3))")
    
    if haskey(metrics, "avg_precision")
        println("  - Average precision: $(round(metrics["avg_precision"], digits=3))")
        println("  - Average recall: $(round(metrics["avg_recall"], digits=3))")
        println("  - F1 score: $(round(metrics["f1_score"], digits=3))")
    end
    
    return metrics
end

"""
    demonstrate_http_api()

Demonstrate HTTP API functionality (simulated).
"""
function demonstrate_http_api()
    println("\n=== HTTP API Demonstration ===\n")
    
    # Simulate API requests
    sample_request = Dict{String, Any}(
        "text" => create_sample_texts()[1],
        "custom_rules" => nothing,
        "weights" => nothing
    )
    
    println("Sample API Request:")
    println("  - Endpoint: POST /detect")
    println("  - Text length: $(length(sample_request["text"])) characters")
    
    # Simulate response structure
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
    
    println("Sample API Response:")
    println("  - Status: $(response["status"])")
    println("  - Motif tokens: $(length(response["motif_tokens"]))")
    println("  - Total motifs detected: $(response["document_analysis"]["summary"]["total_motifs_detected"])")
    
    return response
end

"""
    save_demonstration_results(texts, analyses, motif_tokens, limps_data, metrics, api_response)

Save demonstration results to files.
"""
function save_demonstration_results(texts, analyses, motif_tokens, limps_data, metrics, api_response)
    println("\n=== Saving Results ===\n")
    
    # Save sample texts
    open("sample_texts.json", "w") do f
        write(f, JSON3.write(Dict("texts" => texts), pretty=true))
    end
    println("✓ Saved sample texts to sample_texts.json")
    
    # Save batch analysis results
    batch_results = []
    for (i, analysis) in enumerate(analyses)
        push!(batch_results, Dict{String, Any}(
            "document_id" => i,
            "analysis" => create_motif_report(analysis)
        ))
    end
    
    open("batch_analysis_results.json", "w") do f
        write(f, JSON3.write(Dict("results" => batch_results), pretty=true))
    end
    println("✓ Saved batch analysis to batch_analysis_results.json")
    
    # Save LiMps integration data
    open("limps_integration.json", "w") do f
        write(f, JSON3.write(limps_data, pretty=true))
    end
    println("✓ Saved LiMps integration data to limps_integration.json")
    
    # Save metrics
    open("performance_metrics.json", "w") do f
        write(f, JSON3.write(metrics, pretty=true))
    end
    println("✓ Saved performance metrics to performance_metrics.json")
    
    # Save API response
    open("api_response.json", "w") do f
        write(f, JSON3.write(api_response, pretty=true))
    end
    println("✓ Saved API response to api_response.json")
end

# Run demonstration
if abspath(PROGRAM_FILE) == @__FILE__
    println("Starting Motif Detection Engine Demonstration...\n")
    
    # Run all demonstrations
    texts = demonstrate_motif_detection()
    analyses = demonstrate_batch_processing()
    motif_tokens, limps_data = demonstrate_vectorization_integration()
    metrics = demonstrate_metrics_calculation()
    api_response = demonstrate_http_api()
    
    # Save results
    save_demonstration_results(texts, analyses, motif_tokens, limps_data, metrics, api_response)
    
    println("\n=== Demonstration Complete ===")
    println("The Motif Detection Engine is ready for integration with LiMps!")
end