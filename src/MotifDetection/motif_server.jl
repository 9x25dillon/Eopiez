using HTTP
using JSON

include("integration.jl")
include("../MessageVectorizer.jl")

using .MessageVectorizer

function motif_handler(req::HTTP.Request)
    try
        body = JSON.parse(String(req.body))
        text = body["text"]
        
        # Perform enhanced motif analysis
        results = enhanced_motif_analysis(text)
        
        # Format response
        response_data = Dict{String, Any}()
        
        # Convert coherent paths to serializable format
        paths_data = []
        for (path, score) in results["coherent_paths"]
            path_data = [Dict(
                "motif_type" => section["motif_type"],
                "total_weight" => section["total_weight"],
                "entropy" => section["entropy"]
            ) for section in path]
            push!(paths_data, Dict("path" => path_data, "coherence_score" => score))
        end
        
        response_data["coherent_paths"] = paths_data
        response_data["emergent_structures"] = results["emergent_structures"]
        response_data["sheaf_info"] = results["sheaf_info"]
        response_data["vector_metrics"] = results["vector_metrics"]
        
        return HTTP.Response(200, JSON.json(response_data))
    catch e
        return HTTP.Response(500, JSON.json(Dict("error" => string(e))))
    end
end

function start_motif_server(port::Int=8081)
    HTTP.serve(port) do req
        if req.method == "POST" && req.target == "/detect"
            return motif_handler(req)
        else
            return HTTP.Response(404, "Not Found")
        end
    end
end

export start_motif_server