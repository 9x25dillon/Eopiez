module MotifServer

using HTTP
using JSON3
using JSON

export start_motif_server, motif_handler, batch_motif_handler, 
       health_check_handler, metrics_handler

include("parser.jl")
using .MotifParser

# Include Message Vectorizer for integration
include("../MessageVectorizer.jl")
using .MessageVectorizer

"""
    motif_handler(req::HTTP.Request)

Handle individual motif detection requests.
"""
function motif_handler(req::HTTP.Request)
    try
        # Parse request body
        body = JSON.parse(String(req.body))
        
        # Extract text and optional parameters
        text = get(body, "text", "")
        custom_rules = get(body, "custom_rules", nothing)
        weights = get(body, "weights", nothing)
        
        if isempty(text)
            return HTTP.Response(400, JSON.json(Dict("error" => "No text provided")))
        end
        
        # Parse document
        analysis = parse_document(text)
        
        # Extract motif tokens for Message Vectorizer
        motif_tokens = extract_motif_tokens(analysis)
        
        # Create response
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
        
        return HTTP.Response(200, JSON.json(response))
        
    catch e
        return HTTP.Response(500, JSON.json(Dict("error" => "Internal server error: $(string(e))")))
    end
end

"""
    batch_motif_handler(req::HTTP.Request)

Handle batch motif detection requests.
"""
function batch_motif_handler(req::HTTP.Request)
    try
        # Parse request body
        body = JSON.parse(String(req.body))
        
        # Extract documents and optional parameters
        documents = get(body, "documents", [])
        custom_rules = get(body, "custom_rules", nothing)
        weights = get(body, "weights", nothing)
        
        if isempty(documents)
            return HTTP.Response(400, JSON.json(Dict("error" => "No documents provided")))
        end
        
        # Parse all documents
        analyses = batch_parse_documents(documents)
        
        # Create response for each document
        results = []
        for (i, analysis) in enumerate(analyses)
            motif_tokens = extract_motif_tokens(analysis)
            
            push!(results, Dict{String, Any}(
                "document_id" => i,
                "document_analysis" => create_motif_report(analysis),
                "motif_tokens" => [
                    Dict{String, Any}(
                        "name" => string(token.name),
                        "properties" => token.properties,
                        "weight" => token.weight,
                        "context" => [string(ctx) for ctx in token.context]
                    ) for token in motif_tokens
                ],
                "metrics" => calculate_motif_metrics(analysis)
            ))
        end
        
        # Calculate batch metrics
        batch_metrics = Dict{String, Any}(
            "total_documents" => length(results),
            "avg_motif_density" => mean([r["document_analysis"]["summary"]["motif_density"] for r in results]),
            "avg_confidence" => mean([r["document_analysis"]["summary"]["avg_confidence"] for r in results]),
            "total_motifs_detected" => sum([r["document_analysis"]["summary"]["total_motifs_detected"] for r in results])
        )
        
        response = Dict{String, Any}(
            "status" => "success",
            "results" => results,
            "batch_metrics" => batch_metrics,
            "timestamp" => time()
        )
        
        return HTTP.Response(200, JSON.json(response))
        
    catch e
        return HTTP.Response(500, JSON.json(Dict("error" => "Internal server error: $(string(e))")))
    end
end

"""
    vectorize_motifs_handler(req::HTTP.Request)

Handle motif vectorization requests using Message Vectorizer.
"""
function vectorize_motifs_handler(req::HTTP.Request)
    try
        # Parse request body
        body = JSON.parse(String(req.body))
        
        # Extract motif tokens and vectorizer parameters
        motif_tokens_data = get(body, "motif_tokens", [])
        embedding_dim = get(body, "embedding_dim", 64)
        entropy_threshold = get(body, "entropy_threshold", 0.5)
        compression_ratio = get(body, "compression_ratio", 0.8)
        
        if isempty(motif_tokens_data)
            return HTTP.Response(400, JSON.json(Dict("error" => "No motif tokens provided")))
        end
        
        # Convert JSON motif tokens to MotifToken objects
        motif_tokens = MotifToken[]
        for token_data in motif_tokens_data
            token = MotifToken(
                Symbol(token_data["name"]),
                Dict{Symbol, Any}(Symbol(k) => v for (k, v) in token_data["properties"]),
                token_data["weight"],
                [Symbol(ctx) for ctx in token_data["context"]]
            )
            push!(motif_tokens, token)
        end
        
        # Initialize vectorizer
        vectorizer = MessageVectorizer(embedding_dim, 
                                     entropy_threshold=entropy_threshold, 
                                     compression_ratio=compression_ratio)
        
        # Add motif embeddings
        for token in motif_tokens
            add_motif_embedding!(vectorizer, token)
        end
        
        # Vectorize message
        message_state = vectorize_message(motif_tokens, vectorizer)
        
        # Get al-ULS compatible output
        uls_output = al_uls_interface(message_state)
        
        response = Dict{String, Any}(
            "status" => "success",
            "message_state" => uls_output,
            "vectorizer_config" => Dict{String, Any}(
                "embedding_dim" => vectorizer.embedding_dim,
                "entropy_threshold" => vectorizer.entropy_threshold,
                "compression_ratio" => vectorizer.compression_ratio
            ),
            "timestamp" => time()
        )
        
        return HTTP.Response(200, JSON.json(response))
        
    catch e
        return HTTP.Response(500, JSON.json(Dict("error" => "Internal server error: $(string(e))")))
    end
end

"""
    health_check_handler(req::HTTP.Request)

Handle health check requests.
"""
function health_check_handler(req::HTTP.Request)
    response = Dict{String, Any}(
        "status" => "healthy",
        "service" => "Motif Detection Engine",
        "version" => "1.0.0",
        "timestamp" => time()
    )
    
    return HTTP.Response(200, JSON.json(response))
end

"""
    metrics_handler(req::HTTP.Request)

Handle metrics requests.
"""
function metrics_handler(req::HTTP.Request)
    response = Dict{String, Any}(
        "motif_categories" => length(MotifDefinitions.MOTIF_RULES),
        "available_motifs" => collect(keys(MotifDefinitions.MOTIF_RULES)),
        "motif_weights" => MotifDefinitions.MOTIF_WEIGHTS,
        "motif_contexts" => MotifDefinitions.MOTIF_CONTEXTS,
        "timestamp" => time()
    )
    
    return HTTP.Response(200, JSON.json(response))
end

"""
    start_motif_server(port::Int=8081; host::String="0.0.0.0")

Start the motif detection HTTP server.
"""
function start_motif_server(port::Int=8081; host::String="0.0.0.0")
    println("Starting Motif Detection Server on $host:$port")
    println("Available endpoints:")
    println("  POST /detect - Detect motifs in single document")
    println("  POST /batch - Detect motifs in multiple documents")
    println("  POST /vectorize - Vectorize motif tokens")
    println("  GET  /health - Health check")
    println("  GET  /metrics - Server metrics")
    
    HTTP.serve(host, port) do req
        try
            if req.method == "POST"
                if req.target == "/detect"
                    return motif_handler(req)
                elseif req.target == "/batch"
                    return batch_motif_handler(req)
                elseif req.target == "/vectorize"
                    return vectorize_motifs_handler(req)
                else
                    return HTTP.Response(404, JSON.json(Dict("error" => "Endpoint not found")))
                end
            elseif req.method == "GET"
                if req.target == "/health"
                    return health_check_handler(req)
                elseif req.target == "/metrics"
                    return metrics_handler(req)
                else
                    return HTTP.Response(404, JSON.json(Dict("error" => "Endpoint not found")))
                end
            else
                return HTTP.Response(405, JSON.json(Dict("error" => "Method not allowed")))
            end
        catch e
            return HTTP.Response(500, JSON.json(Dict("error" => "Internal server error: $(string(e))")))
        end
    end
end

"""
    create_limps_integration(motif_tokens::Vector{MotifToken})

Create LiMps symbolic memory engine integration format.
"""
function create_limps_integration(motif_tokens::Vector{MotifToken})
    limps_data = Dict{String, Any}()
    
    # Convert motif tokens to LiMps format
    limps_data["motif_entities"] = [
        Dict{String, Any}(
            "id" => string(token.name),
            "type" => "motif",
            "properties" => token.properties,
            "weight" => token.weight,
            "context" => [string(ctx) for ctx in token.context],
            "timestamp" => time()
        ) for token in motif_tokens
    ]
    
    # Add symbolic relationships
    limps_data["relationships"] = []
    for (i, token1) in enumerate(motif_tokens)
        for (j, token2) in enumerate(motif_tokens)
            if i < j
                # Check for contextual relationships
                common_context = intersect(token1.context, token2.context)
                if !isempty(common_context)
                    push!(limps_data["relationships"], Dict{String, Any}(
                        "source" => string(token1.name),
                        "target" => string(token2.name),
                        "type" => "contextual",
                        "strength" => length(common_context) / max(length(token1.context), length(token2.context)),
                        "shared_context" => [string(ctx) for ctx in common_context]
                    ))
                end
            end
        end
    end
    
    # Add metadata
    limps_data["metadata"] = Dict{String, Any}(
        "total_motifs" => length(motif_tokens),
        "total_relationships" => length(limps_data["relationships"]),
        "source" => "motif_detection_engine",
        "version" => "1.0.0"
    )
    
    return limps_data
end

end # module