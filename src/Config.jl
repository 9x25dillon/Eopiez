"""
    Config.jl

Configuration and constants for the motif detection system.
"""

module Config

using ..Types: MotifType, ISOLATION, SNAKE, STRAND, MEMORY, TEMPORAL, FRAGMENTATION

"""
    MotifPatterns

Mapping of motif types to their associated text patterns.
"""
const MOTIF_PATTERNS = Dict{MotifType, Vector{String}}(
    ISOLATION => [
        "alone", "solitude", "desert", "empty", "void", "silence",
        "isolated", "separate", "detached", "remote", "distant"
    ],
    SNAKE => [
        "snake", "serpent", "coil", "slither", "ouroboros",
        "viper", "python", "cobra", "winding", "twisting"
    ],
    STRAND => [
        "strand", "thread", "connection", "link", "chain",
        "fiber", "wire", "cord", "bond", "tie"
    ],
    MEMORY => [
        "memory", "remember", "forgotten", "past", "dream",
        "phantom", "recollection", "reminiscence", "nostalgia"
    ],
    TEMPORAL => [
        "time", "temporal", "moment", "eternity", "cycle",
        "duration", "period", "epoch", "era", "chronology"
    ],
    FRAGMENTATION => [
        "fragment", "broken", "shattered", "pieces", "scattered",
        "split", "divided", "dispersed", "splintered"
    ]
)

"""
    DetectionThresholds

Thresholds for motif detection and analysis.
"""
const DETECTION_THRESHOLDS = Dict{String, Float64}(
    "min_confidence" => 0.3,
    "min_weight" => 0.1,
    "coherence_threshold" => 0.5,
    "dominant_motif_threshold" => 0.6
)

"""
    VectorConfig

Configuration for vectorization and numerical analysis.
"""
const VECTOR_CONFIG = Dict{String, Any}(
    "dimensions" => 6,
    "normalization" => "l2",
    "sparsity_threshold" => 1e-6
)

"""
    AnalysisConfig

Configuration for analysis algorithms.
"""
const ANALYSIS_CONFIG = Dict{String, Any}(
    "entropy_base" => 2.0,
    "coherence_window_size" => 5,
    "max_path_length" => 10
)

"""
    get_motif_patterns(motif_type::MotifType) -> Vector{String}

Get patterns associated with a specific motif type.
"""
function get_motif_patterns(motif_type::MotifType)::Vector{String}
    return get(MOTIF_PATTERNS, motif_type, String[])
end

"""
    get_threshold(key::String) -> Float64

Get a detection threshold by key.
"""
function get_threshold(key::String)::Float64
    return get(DETECTION_THRESHOLDS, key, 0.0)
end

"""
    get_vector_config(key::String) -> Any

Get vector configuration by key.
"""
function get_vector_config(key::String)
    return get(VECTOR_CONFIG, key, nothing)
end

"""
    get_analysis_config(key::String) -> Any

Get analysis configuration by key.
"""
function get_analysis_config(key::String)
    return get(ANALYSIS_CONFIG, key, nothing)
end

export MOTIF_PATTERNS, DETECTION_THRESHOLDS, VECTOR_CONFIG, ANALYSIS_CONFIG
export get_motif_patterns, get_threshold, get_vector_config, get_analysis_config

end # module