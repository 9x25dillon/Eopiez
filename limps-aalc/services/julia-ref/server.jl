using HTTP, JSON3, Symbolics, WebSockets

function simplify_expr(s)
    @variables x y z
    try
        ex = Meta.parse(s)
        # basic: return string, real impl would map to Symbolics expressions safely
        return string(ex)
    catch
        return s
    end
end

HTTP.@register(HTTP.Router(), "POST", "/simplify") do req
    body = JSON3.read(String(req.body))
    exprs = get(body, :exprs, String[])
    res = [simplify_expr(e) for e in exprs]
    return HTTP.Response(200, JSON3.write(Dict(:result=>res)))
end

function ws_handler(ws)
    for msg in WebSockets.readguarded(ws)
        data = JSON3.read(String(msg))
        fn = get(data, :fn, "")
        if fn == "simplify"
            ex = get(data, :expr, "")
            res = simplify_expr(ex)
            WebSockets.send(ws, JSON3.write(Dict(:result=>res)))
        else
            WebSockets.send(ws, JSON3.write(Dict(:error=>"unknown fn")))
        end
    end
end

HTTP.serve(host="0.0.0.0", port=8008) do req::HTTP.Request
    if HTTP.URIs.path(req.target) == "/ws"
        return WebSockets.upgrade(ws_handler, req)
    else
        return HTTP.Response(200, "JuliaRef alive")
    end
end