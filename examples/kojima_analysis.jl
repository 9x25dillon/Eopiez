#!/usr/bin/env julia

println("Kojima Narrative Analysis Example")

include("../src/MotifDetection/integration.jl")
include("../src/MessageVectorizer.jl")

using .MessageVectorizer

# Sample Kojima-style text
kojima_text = """
The phantom snake slithers through time, a strand of DNA uncoiling in silence.
He stands alone in the desert, watching the ouroboros consume its own tail.
Memory fragments connect like threads in a tapestry of isolation.
Connection forms between these symbols, creating a coherent narrative thread.
The past coils around the present, serpents dancing through forgotten dreams.
"""

println("Analyzing text for Kojima motifs...")
results = enhanced_motif_analysis(kojima_text)

println("\n=== MOTIF DETECTION RESULTS ===")
println("Total global sections: $(results["sheaf_info"]["global_sections"])")

println("\n=== EMERGENT STRUCTURES ===")
for (structure, instances) in results["emergent_structures"]
    println("$structure: $(length(instances)) instances")
end

println("\n=== VECTOR METRICS ===")
for (section_id, metrics) in results["vector_metrics"]
    println("$section_id:")
    println("  Vector norm: $(round(metrics["vector_norm"], digits=3))")
    println("  Entropy score: $(round(metrics["entropy_score"], digits=3))")
    println("  Motif count: $(metrics["motif_count"])")
end

println("\n=== COHERENT PATHS ===")
for (i, (path, score)) in enumerate(results["coherent_paths"])
    println("Path $i (coherence: $(round(score, digits=3))):")
    for section in path
        println("  $(section["motif_type"]): weight=$(round(section["total_weight"], digits=3))")
    end
    println()
end