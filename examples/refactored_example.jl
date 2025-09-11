#!/usr/bin/env julia

"""
Example usage of the refactored MotifAnalysis system.

This example demonstrates the clean, professional API for analyzing
Kojima-esque motifs in text.
"""

using Logging
global_logger(ConsoleLogger(stderr, Logging.Info))

# Include the main module
include("../src/MotifAnalysis.jl")
using .MotifAnalysis

println("🎭 MotifAnalysis v$(version()) Example")
println("=" ^ 50)

# Sample texts for analysis
texts = [
    "He stood alone in the desert, watching the snake coil around the strand of memory.",
    "The phantom snake slithers through time, a strand of DNA uncoiling in silence.",
    "Memory fragments connect like threads in a tapestry of isolation.",
    "The past coils around the present, serpents dancing through forgotten dreams.",
    "Connection forms between these symbols, creating a coherent narrative thread."
]

println("\n📊 Analyzing individual texts...")
println("-" ^ 40)

for (i, text) in enumerate(texts)
    println("\nText $i: \"$(text[1:min(50, length(text))])...\"")
    
    # Perform quick analysis
    result = quick_analysis(text)
    
    println("  Dominant motifs: $(join(result["dominant_motifs"], ", "))")
    println("  Coherence score: $(round(result["coherence_score"], digits=3))")
    println("  Motif count: $(result["motif_count"])")
    println("  Entropy: $(round(result["entropy"], digits=3))")
    println("  Motif density: $(round(result["motif_density"], digits=3))")
end

println("\n🔍 Detailed analysis of first text...")
println("-" ^ 40)

# Detailed analysis
detailed_result = analyze_text(texts[1])
println("Full analysis result:")
println("  Total tokens: $(length(detailed_result.tokens))")
println("  Vector norm: $(round(norm(detailed_result.vectorized_message.vector), digits=3))")
println("  Narrative coherence: $(round(detailed_result.narrative_coherence, digits=3))")

println("\n  Detected motifs:")
for token in detailed_result.tokens
    println("    $(string(token.type)): weight=$(round(token.weight, digits=2)), confidence=$(round(token.confidence, digits=2))")
    println("      Matches: $(join(token.matches, ", "))")
    println("      Context: \"$(token.context[1:min(40, length(token.context))])...\"")
end

println("\n📈 Batch analysis and comparison...")
println("-" ^ 40)

# Batch analysis
batch_results = analyze_text_batch(texts)
println("Batch analysis completed for $(length(batch_results)) texts")

# Compare first two texts
comparison = compare_analyses(batch_results[1], batch_results[2])
println("\nComparison between texts 1 and 2:")
println("  Vector similarity: $(round(comparison["vector_similarity"], digits=3))")
println("  Coherence difference: $(round(comparison["coherence_difference"], digits=3))")
println("  Motif overlap: $(round(comparison["motif_overlap"], digits=3))")
println("  Overall similarity: $(round(comparison["overall_similarity"], digits=3))")

println("\n🌐 Motif distribution analysis...")
println("-" ^ 40)

# Distribution analysis
distribution = analyze_motif_distribution(texts)
println("Distribution across all texts:")
println("  Total texts: $(distribution["total_texts"])")
println("  Average coherence: $(round(distribution["average_coherence"], digits=3))")
println("  Average entropy: $(round(distribution["average_entropy"], digits=3))")
println("  Most common motif: $(distribution["most_common_motif"])")

println("\n  Motif frequencies:")
for (motif, count) in sort(collect(distribution["motif_frequency"]), by=x->x[2], rev=true)
    println("    $motif: $count occurrences")
end

println("\n🔧 System information...")
println("-" ^ 40)

println("Supported motif types:")
for motif_type in supported_motif_types()
    patterns = get_motif_patterns(motif_type)
    println("  $(string(motif_type)): $(join(patterns[1:min(3, length(patterns))], ", "))...")
end

println("\n✅ Example completed successfully!")
println("The refactored system provides a clean, professional API for motif analysis.")