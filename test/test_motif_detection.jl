#!/usr/bin/env julia

println("Running Motif Detection Tests...")

include("../src/MotifDetection/tokenizer.jl")
include("../src/MotifDetection/integration.jl")

# Test case 1: Basic motif detection
test_text1 = "He stood alone in the desert, watching the snake coil around the strand of memory."
motif_tokens1 = create_motif_tokens(test_text1)

println("Test 1 - Basic motif detection:")
println("  Found $(length(motif_tokens1)) motif tokens")
for token in motif_tokens1
    println("    $(token.type): weight=$(round(token.weight, digits=2)), categories=$(get(token.attributes, :categories, Set()))")
end

# Test case 2: Sheaf construction
sheaf = process_document_sheaf(test_text1)
println("\nTest 2 - Sheaf construction:")
println("  Created $(length(sheaf.sections)) sections")

# Test case 3: Coherent narrative building
coherent_paths = build_coherent_narrative(sheaf)
println("\nTest 3 - Coherent narrative:")
println("  Found $(length(coherent_paths)) coherent paths")

# Test case 4: Emergent structure detection
emergent_structures = detect_emergent_structure(sheaf)
println("\nTest 4 - Emergent structures:")
for (structure, instances) in emergent_structures
    println("  $structure: $(length(instances)) instances")
end

println("\n✓ All motif detection tests completed successfully")