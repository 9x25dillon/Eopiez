module MotifParser

using TextAnalysis
using StringDistances
using Statistics
using JSON3

export parse_document, extract_motif_tokens, analyze_document_structure, 
       calculate_motif_metrics, create_motif_report

include("motifs.jl")
using .MotifDefinitions

"""
    DocumentAnalysis

Structure to hold comprehensive document analysis results.
"""
struct DocumentAnalysis
    text::String
    motif_tokens::Dict{String, Vector{String}}
    confidence_scores::Dict{String, Float64}
    document_metrics::Dict{String, Any}
    motif_relationships::Dict{String, Vector{String}}
    timestamp::Float64
end

"""
    parse_document(text::String; 
                  custom_rules::Dict{String, Vector{Regex}} = MOTIF_RULES,
                  weights::Dict{String, Float64} = MOTIF_WEIGHTS)

Parse a document and extract motif tokens with comprehensive analysis.
"""
function parse_document(text::String; 
                       custom_rules::Dict{String, Vector{Regex}} = MOTIF_RULES,
                       weights::Dict{String, Float64} = MOTIF_WEIGHTS)
    
    # Basic text preprocessing
    cleaned_text = preprocess_text(text)
    
    # Detect motifs using regex patterns
    motif_tokens = detect_motifs(cleaned_text, custom_rules)
    
    # Calculate confidence scores
    confidence_scores = calculate_motif_confidence(motif_tokens, length(cleaned_text))
    
    # Analyze document structure
    doc_metrics = analyze_document_structure(cleaned_text, motif_tokens)
    
    # Find motif relationships
    motif_relationships = find_motif_relationships(motif_tokens, confidence_scores)
    
    return DocumentAnalysis(
        cleaned_text,
        motif_tokens,
        confidence_scores,
        doc_metrics,
        motif_relationships,
        time()
    )
end

"""
    preprocess_text(text::String)

Preprocess text for better motif detection.
"""
function preprocess_text(text::String)
    # Convert to lowercase for case-insensitive matching
    text = lowercase(text)
    
    # Remove excessive whitespace
    text = replace(text, r"\s+" => " ")
    
    # Remove common punctuation that might interfere
    text = replace(text, r"[^\w\s\-]" => " ")
    
    # Normalize whitespace
    text = strip(text)
    
    return text
end

"""
    extract_motif_tokens(analysis::DocumentAnalysis)

Extract motif tokens in a format suitable for the Message Vectorizer.
"""
function extract_motif_tokens(analysis::DocumentAnalysis)
    motif_tokens = MotifToken[]
    
    for (motif_name, tokens) in analysis.motif_tokens
        if !isempty(tokens)
            # Create properties dictionary
            properties = Dict{Symbol, Any}(
                :frequency => length(tokens),
                :confidence => analysis.confidence_scores[motif_name],
                :weight => get(MOTIF_WEIGHTS, motif_name, 0.5)
            )
            
            # Add contextual information
            context = Symbol[]
            if haskey(MOTIF_CONTEXTS, motif_name)
                for context_motif in MOTIF_CONTEXTS[motif_name]
                    if analysis.confidence_scores[context_motif] > 0.3
                        push!(context, Symbol(context_motif))
                    end
                end
            end
            
            # Create MotifToken
            token = MotifToken(
                Symbol(motif_name),
                properties,
                analysis.confidence_scores[motif_name],
                context
            )
            
            push!(motif_tokens, token)
        end
    end
    
    return motif_tokens
end

"""
    analyze_document_structure(text::String, motif_tokens::Dict{String, Vector{String}})

Analyze document structure and calculate various metrics.
"""
function analyze_document_structure(text::String, motif_tokens::Dict{String, Vector{String}})
    metrics = Dict{String, Any}()
    
    # Basic text metrics
    metrics["text_length"] = length(text)
    metrics["word_count"] = length(split(text))
    metrics["sentence_count"] = length(split(text, r"[.!?]+"))
    
    # Motif density metrics
    total_motifs = sum(length(tokens) for tokens in values(motif_tokens))
    metrics["motif_density"] = total_motifs / metrics["word_count"]
    metrics["unique_motifs"] = length([motif for motif in keys(motif_tokens) if !isempty(motif_tokens[motif])])
    
    # Motif distribution
    motif_counts = Dict{String, Int}()
    for (motif, tokens) in motif_tokens
        motif_counts[motif] = length(tokens)
    end
    metrics["motif_distribution"] = motif_counts
    
    # Calculate motif diversity (Shannon entropy)
    if total_motifs > 0
        probabilities = [count / total_motifs for count in values(motif_counts) if count > 0]
        entropy = -sum(p * log2(p) for p in probabilities)
        metrics["motif_entropy"] = entropy
    else
        metrics["motif_entropy"] = 0.0
    end
    
    # Text complexity metrics
    metrics["avg_word_length"] = mean(length(word) for word in split(text))
    metrics["vocabulary_size"] = length(unique(split(text)))
    
    return metrics
end

"""
    find_motif_relationships(motif_tokens::Dict{String, Vector{String}}, 
                           confidence_scores::Dict{String, Float64})

Find relationships between detected motifs.
"""
function find_motif_relationships(motif_tokens::Dict{String, Vector{String}}, 
                                confidence_scores::Dict{String, Float64})
    relationships = Dict{String, Vector{String}}()
    
    # Find motifs with high confidence
    high_confidence_motifs = [motif for (motif, score) in confidence_scores if score > 0.5]
    
    for motif in high_confidence_motifs
        related = String[]
        
        # Check contextual relationships
        if haskey(MOTIF_CONTEXTS, motif)
            for context_motif in MOTIF_CONTEXTS[motif]
                if haskey(confidence_scores, context_motif) && 
                   confidence_scores[context_motif] > 0.3
                    push!(related, context_motif)
                end
            end
        end
        
        # Check co-occurrence patterns
        for other_motif in high_confidence_motifs
            if motif != other_motif && confidence_scores[other_motif] > 0.4
                # Simple co-occurrence boost
                if !isempty(motif_tokens[motif]) && !isempty(motif_tokens[other_motif])
                    push!(related, other_motif)
                end
            end
        end
        
        relationships[motif] = unique(related)
    end
    
    return relationships
end

"""
    calculate_motif_metrics(analysis::DocumentAnalysis)

Calculate precision and recall metrics against manual benchmarks.
"""
function calculate_motif_metrics(analysis::DocumentAnalysis; 
                               manual_benchmarks::Dict{String, Vector{String}} = Dict{String, Vector{String}}())
    
    metrics = Dict{String, Any}()
    
    # Calculate overall motif statistics
    detected_motifs = [motif for (motif, tokens) in analysis.motif_tokens if !isempty(tokens)]
    metrics["total_detected_motifs"] = length(detected_motifs)
    metrics["total_motif_occurrences"] = sum(length(tokens) for tokens in values(analysis.motif_tokens))
    
    # Calculate confidence statistics
    confidence_values = collect(values(analysis.confidence_scores))
    metrics["avg_confidence"] = mean(confidence_values)
    metrics["max_confidence"] = maximum(confidence_values)
    metrics["min_confidence"] = minimum(confidence_values)
    metrics["confidence_std"] = std(confidence_values)
    
    # Compare with manual benchmarks if provided
    if !isempty(manual_benchmarks)
        precision_scores = Float64[]
        recall_scores = Float64[]
        
        for (motif, manual_tokens) in manual_benchmarks
            if haskey(analysis.motif_tokens, motif)
                detected_tokens = analysis.motif_tokens[motif]
                
                # Calculate precision (detected correct / total detected)
                correct_detections = length(intersect(detected_tokens, manual_tokens))
                precision = correct_detections / max(length(detected_tokens), 1)
                push!(precision_scores, precision)
                
                # Calculate recall (detected correct / total correct)
                recall = correct_detections / max(length(manual_tokens), 1)
                push!(recall_scores, recall)
            end
        end
        
        if !isempty(precision_scores)
            metrics["avg_precision"] = mean(precision_scores)
            metrics["avg_recall"] = mean(recall_scores)
            metrics["f1_score"] = 2 * metrics["avg_precision"] * metrics["avg_recall"] / 
                                 (metrics["avg_precision"] + metrics["avg_recall"])
        end
    end
    
    return metrics
end

"""
    create_motif_report(analysis::DocumentAnalysis; 
                       include_metrics::Bool = true,
                       include_relationships::Bool = true)

Create a comprehensive motif detection report.
"""
function create_motif_report(analysis::DocumentAnalysis; 
                           include_metrics::Bool = true,
                           include_relationships::Bool = true)
    
    report = Dict{String, Any}()
    
    # Basic document info
    report["document_info"] = Dict{String, Any}(
        "text_length" => analysis.document_metrics["text_length"],
        "word_count" => analysis.document_metrics["word_count"],
        "sentence_count" => analysis.document_metrics["sentence_count"],
        "timestamp" => analysis.timestamp
    )
    
    # Detected motifs
    report["detected_motifs"] = Dict{String, Any}()
    for (motif, tokens) in analysis.motif_tokens
        if !isempty(tokens)
            report["detected_motifs"][motif] = Dict{String, Any}(
                "tokens" => tokens,
                "count" => length(tokens),
                "confidence" => analysis.confidence_scores[motif],
                "weight" => get(MOTIF_WEIGHTS, motif, 0.5)
            )
        end
    end
    
    # Include metrics if requested
    if include_metrics
        report["metrics"] = analysis.document_metrics
    end
    
    # Include relationships if requested
    if include_relationships
        report["motif_relationships"] = analysis.motif_relationships
    end
    
    # Summary statistics
    report["summary"] = Dict{String, Any}(
        "total_motifs_detected" => length([motif for motif in keys(analysis.motif_tokens) 
                                         if !isempty(analysis.motif_tokens[motif])]),
        "total_occurrences" => sum(length(tokens) for tokens in values(analysis.motif_tokens)),
        "avg_confidence" => mean(values(analysis.confidence_scores)),
        "motif_density" => analysis.document_metrics["motif_density"]
    )
    
    return report
end

"""
    batch_parse_documents(texts::Vector{String}; 
                         custom_rules::Dict{String, Vector{Regex}} = MOTIF_RULES)

Parse multiple documents in batch.
"""
function batch_parse_documents(texts::Vector{String}; 
                             custom_rules::Dict{String, Vector{Regex}} = MOTIF_RULES)
    
    analyses = DocumentAnalysis[]
    
    for (i, text) in enumerate(texts)
        try
            analysis = parse_document(text, custom_rules=custom_rules)
            push!(analyses, analysis)
        catch e
            println("Error parsing document $i: $e")
        end
    end
    
    return analyses
end

end # module