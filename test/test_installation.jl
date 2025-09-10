#!/usr/bin/env julia

println("🔧 Testing 9xdSq-LIMPS-FemTO-R1C Enhanced Installation...")

try
    # Test basic Julia functionality
    println("✓ Julia environment: OK")
    
    # Test required packages
    println("📦 Testing required packages...")
    
    using LinearAlgebra
    println("✓ LinearAlgebra: OK")
    
    using Statistics
    println("✓ Statistics: OK")
    
    using Random
    println("✓ Random: OK")
    
    # Test optional packages
    try
        using TextAnalysis
        println("✓ TextAnalysis: OK")
    catch e
        println("⚠ TextAnalysis: Not available (optional)")
    end
    
    try
        using HTTP
        println("✓ HTTP: OK")
    catch e
        println("⚠ HTTP: Not available (optional)")
    end
    
    try
        using JSON
        println("✓ JSON: OK")
    catch e
        println("⚠ JSON: Not available (optional)")
    end
    
    # Test basic motif detection
    println("🧠 Testing motif detection...")
    
    # Include our modules
    include("../src/MessageVectorizer.jl")
    include("../src/MotifDetection/integration.jl")
    
    using .MessageVectorizer
    
    # Test with sample text
    test_text = "He stood alone in the desert, watching the snake coil around the strand of memory."
    motif_tokens = create_motif_tokens(test_text)
    
    println("✓ Motif detection: Found $(length(motif_tokens)) tokens")
    
    # Test vectorization
    vector = vectorize_motifs(motif_tokens)
    println("✓ Vectorization: Vector norm = $(round(norm(vector), digits=3))")
    
    # Test entropy calculation
    entropy = calculate_entropy(vector)
    println("✓ Entropy calculation: $(round(entropy, digits=3))")
    
    # Test sheaf analysis
    sheaf = create_narrative_sheaf(motif_tokens)
    println("✓ Sheaf analysis: $(length(sheaf.sections)) sections")
    
    println("\n🎉 All tests passed! Installation is complete.")
    println("🚀 Ready to run enhanced motif detection and analysis.")
    
catch e
    println("❌ Installation test failed:")
    println("Error: $e")
    println("\nPlease check your Julia installation and dependencies.")
    exit(1)
end