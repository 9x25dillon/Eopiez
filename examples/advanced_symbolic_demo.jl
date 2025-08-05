using Pkg
Pkg.activate(".")

using MessageVectorizer
using Symbolics
using SymbolicNumericIntegration
using LinearAlgebra
using JSON3

"""
    demonstrate_symbolic_integration()

Demonstrate symbolic integration and manipulation capabilities.
"""
function demonstrate_symbolic_integration()
    println("=== Advanced Symbolic Computation Demo ===\n")
    
    # Initialize vectorizer with symbolic variables
    vectorizer = MessageVectorizer(16)
    
    # Create complex motif configurations
    motifs = [
        # Temporal evolution motif
        MotifToken(
            :temporal_evolution,
            Dict{Symbol, Any}(
                :evolution_rate => 0.4,
                :time_scale => 100.0,
                :stability => 0.7
            ),
            0.8,
            [:temporal, :evolutionary, :systemic]
        ),
        
        # Spatial diffusion motif
        MotifToken(
            :spatial_diffusion,
            Dict{Symbol, Any}(
                :diffusion_coefficient => 0.3,
                :spatial_dimensions => 3,
                :boundary_conditions => "periodic"
            ),
            0.6,
            [:spatial, :physical, :boundary]
        ),
        
        # Cognitive processing motif
        MotifToken(
            :cognitive_processing,
            Dict{Symbol, Any}(
                :processing_speed => 0.9,
                :memory_capacity => 0.8,
                :attention_focus => 0.7
            ),
            0.9,
            [:cognitive, :neural, :attentional]
        )
    ]
    
    # Add embeddings
    for motif in motifs
        add_motif_embedding!(vectorizer, motif)
    end
    
    println("✓ Created $(length(motifs)) complex motif configurations")
    
    # Create message state
    message_state = vectorize_message(motifs, vectorizer)
    
    println("✓ Generated symbolic message state")
    println("Symbolic expression: $(message_state.symbolic_expression)")
    println()
    
    # Demonstrate symbolic manipulation
    println("--- Symbolic Manipulation ---")
    
    # Extract symbolic expression
    sym_expr = message_state.symbolic_expression
    
    # Create symbolic variables for integration
    @variables t, x, y, z
    
    # Demonstrate symbolic differentiation
    if haskey(vectorizer.symbolic_variables, :s)
        s = vectorizer.symbolic_variables[:s]
        derivative = Symbolics.derivative(sym_expr, s)
        println("Derivative with respect to s: $derivative")
    end
    
    # Demonstrate symbolic integration (if possible)
    try
        # Create a simple integrand
        integrand = sym_expr * t
        integral = SymbolicNumericIntegration.integrate(integrand, t)
        println("Integral with respect to t: $integral")
    catch e
        println("Integration result: Complex expression (simplified)")
    end
    
    println()
    
    # Analyze vector properties
    println("--- Vector Analysis ---")
    vector = message_state.vector_representation
    
    println("Vector properties:")
    println("  - Dimension: $(length(vector))")
    println("  - Norm: $(round(norm(vector), digits=4))")
    println("  - Mean: $(round(mean(vector), digits=4))")
    println("  - Std: $(round(std(vector), digits=4))")
    println("  - Min: $(round(minimum(vector), digits=4))")
    println("  - Max: $(round(maximum(vector), digits=4))")
    println()
    
    # Demonstrate motif interaction analysis
    println("--- Motif Interaction Analysis ---")
    
    motif_config = message_state.motif_configuration
    motif_names = collect(keys(motif_config))
    
    println("Motif interactions:")
    for (i, motif1) in enumerate(motif_names)
        for (j, motif2) in enumerate(motif_names)
            if i < j
                weight1 = motif_config[motif1]
                weight2 = motif_config[motif2]
                interaction_strength = weight1 * weight2
                println("  $(motif1) × $(motif2): $(round(interaction_strength, digits=3))")
            end
        end
    end
    println()
    
    # Demonstrate entropy analysis
    println("--- Entropy Analysis ---")
    
    entropy = message_state.entropy_score
    println("Message entropy: $(round(entropy, digits=4))")
    
    # Calculate information density
    info_density = entropy / length(vector)
    println("Information density: $(round(info_density, digits=4))")
    
    # Entropy interpretation
    if entropy > 2.0
        println("High complexity message (entropy > 2.0)")
    elseif entropy > 1.0
        println("Medium complexity message (entropy 1.0-2.0)")
    else
        println("Low complexity message (entropy < 1.0)")
    end
    println()
    
    # Demonstrate al-ULS interface with detailed output
    println("--- al-ULS Interface Details ---")
    uls_output = al_uls_interface(message_state)
    
    println("Complete al-ULS output:")
    for (key, value) in uls_output
        if value isa Vector && length(value) > 10
            println("  $key: [$(value[1:5])...] (length: $(length(value)))")
        else
            println("  $key: $value")
        end
    end
    println()
    
    return message_state, vectorizer
end

"""
    demonstrate_narrative_states()

Demonstrate how motif combinations create different narrative states.
"""
function demonstrate_narrative_states()
    println("=== Narrative State Analysis ===\n")
    
    vectorizer = MessageVectorizer(32)
    
    # Define different narrative motifs
    narrative_motifs = [
        # Conflict motif
        MotifToken(:conflict, Dict{Symbol, Any}(:intensity => 0.9, :resolution => 0.3), 0.8, [:dramatic, :tension]),
        
        # Resolution motif
        MotifToken(:resolution, Dict{Symbol, Any}(:clarity => 0.8, :satisfaction => 0.7), 0.6, [:closure, :harmony]),
        
        # Transformation motif
        MotifToken(:transformation, Dict{Symbol, Any}(:change_magnitude => 0.7, :irreversibility => 0.8), 0.9, [:evolution, :growth]),
        
        # Stasis motif
        MotifToken(:stasis, Dict{Symbol, Any}(:stability => 0.9, :duration => 0.6), 0.4, [:equilibrium, :persistence])
    ]
    
    # Add embeddings
    for motif in narrative_motifs
        add_motif_embedding!(vectorizer, motif)
    end
    
    # Create different narrative combinations
    narrative_combinations = [
        ("Conflict → Resolution", [narrative_motifs[1], narrative_motifs[2]]),
        ("Transformation", [narrative_motifs[3]]),
        ("Stasis → Conflict", [narrative_motifs[4], narrative_motifs[1]]),
        ("Full Arc", narrative_motifs)
    ]
    
    println("Narrative State Analysis:")
    println()
    
    for (name, motifs) in narrative_combinations
        state = vectorize_message(motifs, vectorizer)
        
        println("$(name):")
        println("  - Entropy: $(round(state.entropy_score, digits=3))")
        println("  - Complexity: $(length(motifs)) motifs")
        println("  - Vector norm: $(round(norm(state.vector_representation), digits=3))")
        
        # Analyze motif balance
        weights = collect(values(state.motif_configuration))
        balance = std(weights)
        println("  - Motif balance (std): $(round(balance, digits=3))")
        println()
    end
end

# Run demonstrations
if abspath(PROGRAM_FILE) == @__FILE__
    println("Starting Advanced Symbolic Demo...\n")
    
    message_state, vectorizer = demonstrate_symbolic_integration()
    demonstrate_narrative_states()
    
    println("=== Advanced Demo Complete ===")
end