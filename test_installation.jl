#!/usr/bin/env julia

println("Testing Message Vectorizer Installation...")

try
    using Pkg
    Pkg.activate(".")
    Pkg.instantiate()
    
    println("✓ Dependencies installed successfully")
    
    # Test basic functionality
    include("src/MessageVectorizer.jl")
    
    using .MessageVectorizer
    
    println("✓ MessageVectorizer module loaded successfully")
    
    # Create a simple test
    vectorizer = MessageVectorizer(16)
    println("✓ Vectorizer initialized with 16 dimensions")
    
    # Create a test motif
    test_motif = MotifToken(
        :test,
        Dict{Symbol, Any}(:intensity => 0.5),
        0.8,
        [:temporal]
    )
    
    add_motif_embedding!(vectorizer, test_motif)
    println("✓ Test motif embedding created")
    
    # Test vectorization
    message_state = vectorize_message([test_motif], vectorizer)
    println("✓ Message vectorization successful")
    println("  - Entropy score: $(round(message_state.entropy_score, digits=3))")
    println("  - Vector norm: $(round(norm(message_state.vector_representation), digits=3))")
    
    # Test al-ULS interface
    uls_output = al_uls_interface(message_state)
    println("✓ al-ULS interface working")
    println("  - Compressed size: $(uls_output["compressed_size"])")
    println("  - Information density: $(round(uls_output["information_density"], digits=4))")
    
    println("\n🎉 All tests passed! Message Vectorizer is ready to use.")
    
catch e
    println("❌ Installation test failed:")
    println("Error: $e")
    println("\nPlease check your Julia installation and try again.")
    exit(1)
end