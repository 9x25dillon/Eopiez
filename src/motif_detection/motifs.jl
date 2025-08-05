module MotifDefinitions

using Regex
using StringDistances

export MOTIF_RULES, MOTIF_WEIGHTS, MOTIF_CONTEXTS, detect_motifs, calculate_motif_confidence

"""
    MOTIF_RULES

Comprehensive dictionary of Kojima-esque motif patterns with regex rules.
"""
const MOTIF_RULES = Dict{String, Vector{Regex}}(
    # Core Kojima motifs
    "isolation" => [
        r"\bisolat(?:ed|ion|ing)\b"i,
        r"\balone\b"i,
        r"\bdesert(?:ed)?\b"i,
        r"\bsilence\b"i,
        r"\bempty\b"i,
        r"\bdisconnect(?:ed|ion)\b"i,
        r"\bseparat(?:ed|ion)\b"i,
        r"\bcut\s+off\b"i,
        r"\bsolitary\b"i,
        r"\blonely\b"i,
        r"\bvoid\b"i,
        r"\bvacuum\b"i,
        r"\babsence\b"i,
        r"\bmissing\b"i,
        r"\bexiled\b"i,
        r"\bcast\s+out\b"i
    ],
    
    "snake" => [
        r"\bsnake(?:s)?\b"i,
        r"\bouroboros\b"i,
        r"\bserpent(?:s)?\b"i,
        r"\bslither(?:ing|ed)?\b"i,
        r"\bcoil(?:ing|ed)?\b"i,
        r"\bviper\b"i,
        r"\bcobra\b"i,
        r"\bpython\b"i,
        r"\bconstrict(?:or|ing)\b"i,
        r"\bvenom(?:ous)?\b"i,
        r"\bfang(?:s)?\b"i,
        r"\bscales?\b"i,
        r"\bsss\b"i,  # Snake sound
        r"\bhiss(?:ing)?\b"i
    ],
    
    "strand" => [
        r"\bstrand(?:s)?\b"i,
        r"\bthread(?:s)?\b"i,
        r"\bfiber(?:s)?\b"i,
        r"\btwine(?:d)?\b"i,
        r"\bDNA\b"i,
        r"\bconnection(?:s)?\b"i,
        r"\blink(?:s|ed)?\b"i,
        r"\bchain(?:s)?\b"i,
        r"\bweb(?:s)?\b"i,
        r"\bnetwork(?:s)?\b"i,
        r"\bwire(?:s)?\b"i,
        r"\bcable(?:s)?\b"i,
        r"\bline(?:s)?\b"i,
        r"\bstring(?:s)?\b"i,
        r"\bcord(?:s)?\b"i,
        r"\btape\b"i,
        r"\bribbon(?:s)?\b"i,
        r"\bweave(?:d|ing)?\b"i,
        r"\bknit(?:ted)?\b"i,
        r"\bentangle(?:d|ment)?\b"i
    ],
    
    "memory" => [
        r"\bmemory(?:ies)?\b"i,
        r"\brecall(?:ing|ed)?\b"i,
        r"\bpast\b"i,
        r"\bnostalgia\b"i,
        r"\bremember(?:ing|ed)?\b"i,
        r"\bflashback(?:s)?\b"i,
        r"\bforget(?:ting|ten)?\b"i,
        r"\bamnesia\b"i,
        r"\bremembrance\b"i,
        r"\bcommemorat(?:ion|ed)\b"i,
        r"\bheritage\b"i,
        r"\blegacy\b"i,
        r"\bancestry\b"i,
        r"\blineage\b"i,
        r"\bhistory\b"i,
        r"\btradition\b"i,
        r"\barchive(?:s)?\b"i,
        r"\brecord(?:s)?\b"i,
        r"\bimprint(?:ed)?\b"i,
        r"\bengrave(?:d)?\b"i,
        r"\betch(?:ed)?\b"i,
        r"\bcarve(?:d)?\b"i
    ],
    
    # Additional Kojima themes
    "technology" => [
        r"\bcyber(?:punk|netic)?\b"i,
        r"\bdigital\b"i,
        r"\bvirtual\b"i,
        r"\bAI\b"i,
        r"\bartificial\s+intelligence\b"i,
        r"\bneural\b"i,
        r"\bnetwork(?:ed)?\b"i,
        r"\bconnected\b"i,
        r"\bwireless\b"i,
        r"\bwire(?:d|less)?\b"i,
        r"\bterminal\b"i,
        r"\binterface\b"i,
        r"\bprogram(?:med|ming)?\b"i,
        r"\bcode(?:d)?\b"i,
        r"\bdata\b"i,
        r"\binformation\b"i,
        r"\btransmission\b"i,
        r"\bsignal(?:s)?\b"i,
        r"\bfrequency\b"i,
        r"\bwavelength\b"i,
        r"\bantenna\b"i,
        r"\bsatellite\b"i,
        r"\borbital\b"i
    ],
    
    "war" => [
        r"\bwar(?:fare)?\b"i,
        r"\bbattle(?:field)?\b"i,
        r"\bcombat\b"i,
        r"\bfight(?:ing|er)?\b"i,
        r"\bsoldier(?:s)?\b"i,
        r"\barmy\b"i,
        r"\bmilitary\b"i,
        r"\bweapon(?:s)?\b"i,
        r"\bgun(?:s)?\b"i,
        r"\bbullet(?:s)?\b"i,
        r"\bexplosion\b"i,
        r"\bbomb(?:s)?\b"i,
        r"\bmissile(?:s)?\b"i,
        r"\btank(?:s)?\b"i,
        r"\bhelicopter(?:s)?\b"i,
        r"\baircraft\b"i,
        r"\bstealth\b"i,
        r"\bcamouflage\b"i,
        r"\btactical\b"i,
        r"\bstrategic\b"i,
        r"\bmission(?:s)?\b"i,
        r"\boperation(?:s)?\b"i,
        r"\bdeployment\b"i,
        r"\bretreat\b"i,
        r"\bsurrender\b"i
    ],
    
    "identity" => [
        r"\bidentity\b"i,
        r"\bself\b"i,
        r"\bego\b"i,
        r"\bpersona\b"i,
        r"\bcharacter\b"i,
        r"\bpersonality\b"i,
        r"\bconsciousness\b"i,
        r"\bawareness\b"i,
        r"\bexistence\b"i,
        r"\bbeing\b"i,
        r"\bessence\b"i,
        r"\bnature\b"i,
        r"\bsoul\b"i,
        r"\bspirit\b"i,
        r"\bghost\b"i,
        r"\bphantom\b"i,
        r"\bshadow\b"i,
        r"\breflection\b"i,
        r"\bmirror\b"i,
        r"\bclone(?:s)?\b"i,
        r"\bcopy(?:ies)?\b"i,
        r"\bduplicate(?:s)?\b"i,
        r"\btwin(?:s)?\b"i,
        r"\bdouble\b"i,
        r"\bfake\b"i,
        r"\bimpostor\b"i,
        r"\bpretender\b"i
    ],
    
    "communication" => [
        r"\bcommunicat(?:ion|ed|ing)\b"i,
        r"\bmessage(?:s)?\b"i,
        r"\bsignal(?:s)?\b"i,
        r"\btransmission\b"i,
        r"\bbroadcast\b"i,
        r"\bradio\b"i,
        r"\btelephone\b"i,
        r"\bphone\b"i,
        r"\bcode(?:d)?\b"i,
        r"\bcipher\b"i,
        r"\bencrypt(?:ed|ion)?\b"i,
        r"\bdecrypt(?:ed|ion)?\b"i,
        r"\bkey\b"i,
        r"\bpassword\b"i,
        r"\baccess\b"i,
        r"\bconnect(?:ion|ed)?\b"i,
        r"\blink(?:ed)?\b"i,
        r"\bbridge\b"i,
        r"\bgateway\b"i,
        r"\bportal\b"i,
        r"\bchannel\b"i,
        r"\bfrequency\b"i,
        r"\bwavelength\b"i,
        r"\bantenna\b"i,
        r"\breceiver\b"i,
        r"\btransmitter\b"i
    ],
    
    "nature" => [
        r"\bnature\b"i,
        r"\bwild(?:erness)?\b"i,
        r"\bforest\b"i,
        r"\bjungle\b"i,
        r"\bdesert\b"i,
        r"\bmountain(?:s)?\b"i,
        r"\bocean\b"i,
        r"\bsea\b"i,
        r"\briver\b"i,
        r"\bstream\b"i,
        r"\bwater\b"i,
        r"\bfire\b"i,
        r"\bearth\b"i,
        r"\bwind\b"i,
        r"\bair\b"i,
        r"\bweather\b"i,
        r"\bclimate\b"i,
        r"\bseason(?:s)?\b"i,
        r"\bday\b"i,
        r"\bnight\b"i,
        r"\bsun\b"i,
        r"\bmoon\b"i,
        r"\bstars?\b"i,
        r"\bcloud(?:s)?\b"i,
        r"\brain\b"i,
        r"\bstorm\b"i,
        r"\bthunder\b"i,
        r"\blightning\b"i
    ]
)

"""
    MOTIF_WEIGHTS

Default weights for different motif categories based on Kojima's thematic importance.
"""
const MOTIF_WEIGHTS = Dict{String, Float64}(
    "isolation" => 0.9,      # Core Kojima theme
    "snake" => 0.8,          # Strong symbolic element
    "strand" => 0.7,         # Connection/networking theme
    "memory" => 0.8,         # Central to narrative
    "technology" => 0.6,     # Background element
    "war" => 0.5,            # Setting element
    "identity" => 0.9,       # Core philosophical theme
    "communication" => 0.7,  # Important plot element
    "nature" => 0.4          # Environmental element
)

"""
    MOTIF_CONTEXTS

Contextual relationships between motifs for enhanced detection.
"""
const MOTIF_CONTEXTS = Dict{String, Vector{String}}(
    "isolation" => ["technology", "war", "identity"],
    "snake" => ["nature", "strand", "memory"],
    "strand" => ["technology", "communication", "snake"],
    "memory" => ["identity", "isolation", "communication"],
    "technology" => ["communication", "war", "strand"],
    "war" => ["technology", "isolation", "identity"],
    "identity" => ["memory", "isolation", "communication"],
    "communication" => ["technology", "strand", "identity"],
    "nature" => ["snake", "isolation", "memory"]
)

"""
    detect_motifs(text::String, rules::Dict{String, Vector{Regex}})

Detect motifs in text using regex patterns.
"""
function detect_motifs(text::String, rules::Dict{String, Vector{Regex}} = MOTIF_RULES)
    tokens = Dict{String, Vector{String}}()
    
    for (motif, patterns) in rules
        matches = String[]
        for pattern in patterns
            for match in eachmatch(pattern, text)
                push!(matches, match.match)
            end
        end
        tokens[motif] = unique(matches)
    end
    
    return tokens
end

"""
    calculate_motif_confidence(motif_tokens::Dict{String, Vector{String}}, 
                              text_length::Int)

Calculate confidence scores for detected motifs.
"""
function calculate_motif_confidence(motif_tokens::Dict{String, Vector{String}}, 
                                  text_length::Int)
    confidence = Dict{String, Float64}()
    
    for (motif, tokens) in motif_tokens
        # Base confidence on frequency and text length
        frequency = length(tokens) / text_length * 1000  # Normalize per 1000 chars
        weight = get(MOTIF_WEIGHTS, motif, 0.5)
        
        # Contextual boost
        context_boost = 0.0
        if haskey(MOTIF_CONTEXTS, motif)
            for context_motif in MOTIF_CONTEXTS[motif]
                if haskey(motif_tokens, context_motif) && !isempty(motif_tokens[context_motif])
                    context_boost += 0.1
                end
            end
        end
        
        confidence[motif] = min(1.0, frequency * weight + context_boost)
    end
    
    return confidence
end

end # module