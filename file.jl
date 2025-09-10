#!/usr/bin/env julia
# Dianne PolyServe + QVNM + CPLearn + Preview + Query + DP (compact)
using HTTP, JSON3, LinearAlgebra, Statistics, Random

##############################
# Util
##############################
epsf() = 1e-12f0
function l2norm!(v::Vector{Float32})
    n = sqrt(sum(x*x for x in v))
    if n > 1e-12f0
        @inbounds for i in eachindex(v); v[i] /= n; end
    end
    v
end

##############################
# (A) Costa–Hero: ID/Entropy
##############################
module CostaHeroID
using NearestNeighbors, Statistics, StatsBase, LinearAlgebra, Random
export id_entropy_global, id_entropy_local, knn_distances

function knn_distances(X::AbstractMatrix{<:Real}; k::Int=10)
    d,N = size(X)
    tree = KDTree(Matrix{Float64}(X))
    D,I = knn(tree, Matrix{Float64}(X), k+1, true) # includes self
    idx = [vec(I[i][2:end]) for i in 1:N]
    dst = [vec(D[i][2:end]) for i in 1:N]
    return idx,dst
end

function L_gamma_subset(S::Vector{Int}, idx_all, dst_all; γ::Float64=1.0)
    present = fill(false, length(idx_all))
    @inbounds for s in S; present[s] = true; end
    total = 0.0
    @inbounds for i in S
        neigh = idx_all[i]; dists = dst_all[i]
        for (j, dij) in zip(neigh, dists)
            if present[j]; total += dij^γ; end
        end
    end
    return total / 2.0
end

ls_fit(xs, ys) = begin
    x̄ = mean(xs); ȳ = mean(ys)
    num = sum((x - x̄)*(y - ȳ) for (x,y) in zip(xs,ys))
    den = sum((x - x̄)^2 for x in xs) + eps()
    a = num/den
    b = ȳ - a*x̄
    a,b
end

function id_entropy_global(X; k::Int=10, γ::Float64=0.5, α::Float64=0.5,
                           plist::Vector{Int}=Int[], boots::Int=16, rng=Random.GLOBAL_RNG)
    d,N = size(X)
    plist = isempty(plist) ? collect(round.(Int, range(max(32, 2k), N; length=8))) : plist
    idx,dst = knn_distances(X; k=k)
    logp = Float64[]; logL = Float64[]
    for p in plist
        μ = 0.0
        trials = min(boots, max(1, cld(N, p)))
        for _ in 1:trials
            S = sample(rng, 1:N, p; replace=false) |> collect
            μ += L_gamma_subset(S, idx, dst; γ=γ)
        end
        μ /= trials
        push!(logp, log(p)); push!(logL, log(max(μ, eps())))
    end
    a,b = ls_fit(logp, logL)
    m_hat = γ / max(1e-9, (1.0 - a))
    H_hat = b / γ
    m_hat, H_hat, (a=a,b=b,γ=γ,α=α,k=k,plist=plist,boots=boots)
end

function id_entropy_local(X; k::Int=10, r::Int=64, γ::Float64=0.5, α::Float64=0.5, boots::Int=8, rng=Random.GLOBAL_RNG)
    d,N = size(X)
    idx,dst = knn_distances(X; k=max(k, r))
    m̂ = fill(Float64(NaN), N); Ĥ = fill(Float64(NaN), N)
    for i in 1:N
        order = sortperm(dst[i])[1:min(r, length(dst[i]))]
        Sfull = vcat(i, idx[i][order]) |> unique |> collect
        P = length(Sfull)
        if P < max(24, 2k); continue; end
        plist = round.(Int, clamp.(range(ceil(Int, 0.3P), P; length=6), 8, P))
        logp=Float64[]; logL=Float64[]
        for p in plist
            μ=0.0; trials=min(boots, max(1, cld(P, p)))
            for _ in 1:trials
                S = sample(rng, Sfull, p; replace=false) |> collect
                μ += L_gamma_subset(S, idx, dst; γ=γ)
            end
            μ/=trials; push!(logp,log(p)); push!(logL,log(max(μ,eps())))
        end
        a,b = ls_fit(logp, logL)
        m̂[i] = γ / max(1e-9, (1.0 - a))
        Ĥ[i] = b / γ
    end
    m̂, Ĥ, (γ=γ, α=α, k=k, r=r, boots=boots)
end

end # CostaHeroID

##############################
# (B) QVNM build/preview
##############################
module QVNM
using LinearAlgebra, Statistics, Graphs, SparseArrays, SimpleWeightedGraphs

export quantum_fidelity, geodesic_dist, blend_weights

quantum_fidelity(V::AbstractMatrix{<:Real}) = begin
    # V: d×N normalized cols
    S = transpose(V) * V
    S .^ 2
end

function geodesic_dist(nei::Vector{Vector{Int}}, w::Vector{Vector{Float64}})
    N = length(nei)
    g = SimpleWeightedGraph(N)
    for i in 1:N
        for (j,wij) in zip(nei[i], w[i])
            add_edge!(g, i, j, wij)
            add_edge!(g, j, i, wij)
        end
    end
    D = fill(Inf, N, N)
    for s in 1:N
        dists = dijkstra_shortest_paths(g, s).dists
        D[s,:] = dists
    end
    D
end

function blend_weights(V::AbstractMatrix{<:Real}, Dg::AbstractMatrix{<:Real},
                       m_hat::AbstractVector{<:Real}, H_hat::AbstractVector{<:Real};
                       k::Int=10, lam_m::Float64=0.3, lam_h::Float64=0.3)
    N = size(V,2)
    sortD = mapslices(sort, Dg; dims=2)[:,1:min(k+1,N)]
    σ = [median(sortD[i,2:end]) for i in 1:N]
    m̄ = mean(m_hat[isfinite.(m_hat)]); H̄ = mean(H_hat[isfinite.(H_hat)])
    for i in 1:N
        σ[i] *= exp(lam_h*((H_hat[i]-H̄)) - lam_m*((m_hat[i]-m̄)))
    end
    # normalize V cols
    Vn = Matrix{Float64}(V)
    for j in 1:N
        n = sqrt(sum(Vn[:,j].^2)); Vn[:,j] ./= (n>1e-12 ? n : 1.0)
    end
    F = quantum_fidelity(Vn)
    S = @. exp(-(Dg^2) / (σ*σ'))
    W = S .* F
    for i in 1:N; W[i,i] = 0.0; end
    Matrix{Float32}(W)
end

end # QVNM

##############################
# (C) QVNM Preview
##############################
module QVNMPreview
using LinearAlgebra, Statistics, JSON3
export preview_from_W

function histo(v::AbstractVector{<:Real}, bins::Int)
    lo,hi = extrema(v)
    if !isfinite(lo) || !isfinite(hi) || hi ≤ lo
        return Dict("bins"=>[], "edges"=>[], "min"=>lo, "max"=>hi, "mean"=>mean(v), "std"=>std(v))
    end
    edges = collect(range(lo, hi; length=bins+1))
    counts = zeros(Int, bins)
    for x in v
        if !isfinite(x); continue; end
        b = clamp(searchsortedlast(edges, x), 1, bins)
        counts[b] += 1
    end
    Dict("bins"=>counts, "edges"=>edges, "min"=>lo, "max"=>hi, "mean"=>mean(v), "std"=>std(v))
end

function preview_from_W(W::AbstractMatrix{<:Real},
                        m_hat::AbstractVector{<:Real},
                        H_hat::AbstractVector{<:Real};
                        r::Int=2, k_eval::Int=10, bins::Int=20)
    N = size(W,1)
    d = vec(sum(W; dims=2))
    deg = Dict("min"=>minimum(d), "max"=>maximum(d), "mean"=>mean(d), "std"=>std(d))
    # normalized kernel S
    ϵ = 1e-12
    Dm12 = 1.0 ./ sqrt.(d .+ ϵ)
    S = (Dm12 .* W) .* transpose(Dm12)
    # eigens (dense fallback)
    ev = eigen(Symmetric(Matrix(S)))
    evals = reverse(ev.values)
    evecs = reverse(ev.vectors, dims=2)
    if length(evals) > k_eval
        evals = evals[1:k_eval]; evecs = evecs[:,1:k_eval]
    end
    gap = length(evals) ≥ 2 ? (evals[1] - evals[2]) : NaN
    rd = min(r, size(evecs,2))
    coords = rd > 0 ? evecs[:,1:rd] : zeros(Float64, N, 0)
    Dict(
        "n"=>N, "edges"=>Int(sum(W .> 0.0) ÷ 2), "degree"=>deg,
        "spectrum"=>Dict("evals"=>evals, "gap"=>gap),
        "eigenmaps"=>Dict("r"=>rd, "coords"=>vec(coords')),
        "histograms"=>Dict("m_hat"=>histo(m_hat, bins), "H_hat"=>histo(H_hat, bins))
    )
end

end # QVNMPreview

##############################
# (D) CPLearn codebook + projector
##############################
module CPLearnCodes
using LinearAlgebra, Statistics, Random
export CodeProj, make_codebook, project_codes, code_affinity, code_hist

struct CodeProj
    W :: Matrix{Float32} # f×c
    τ :: Float32
end

function make_codebook(f::Int, c::Int; seed::Int=2214)
    rng = Random.MersenneTwister(seed)
    W = rand(rng, (-1f0, +1f0), f, c)
    for j in 1:c
        nj = norm(@view W[:,j]); if nj>0; @views W[:,j] .= (sqrt(f)/nj).*W[:,j]; end
    end
    W
end

function row_softmax!(S::Matrix{Float32})
    N,c = size(S)
    @inbounds for i in 1:N
        m = maximum(@view S[i,:]); s=0.0f0
        for j in 1:c; S[i,j] = exp(S[i,j]-m); s+=S[i,j]; end
        invs = 1.0f0/max(s,1e-12f0)
        for j in 1:c; S[i,j]*=invs; end
    end
    S
end

function project_codes(H::AbstractMatrix{<:Real}, cp::CodeProj)
    N,f = size(H); fW,c = size(cp.W); @assert f==fW
    logits = Matrix{Float32}(H) * cp.W
    logits ./= max(cp.τ, 1e-6f0)
    P = row_softmax!(logits)
    hard = Vector{Int}(undef, N)
    @inbounds for i in 1:N
        best=1; bv=P[i,1]
        for j in 2:c
            v=P[i,j]; if v>bv; best=j; bv=v; end
        end
        hard[i]=best
    end
    P,hard
end

function code_affinity(P::AbstractMatrix{<:Real}, hard_idx::Vector{Int}; hard::Bool=false)
    N,c = size(P)
    if hard
        A = zeros(Float32, N,N)
        @inbounds for i in 1:N
            ci=hard_idx[i]; A[i,i]=1.0f0
            for j in i+1:N
                v = (ci==hard_idx[j]) ? 1.0f0 : 0.0f0
                A[i,j]=v; A[j,i]=v
            end
        end
        return A
    else
        A = Matrix{Float32}(P) * Matrix{Float32}(P')
        mx = maximum(A); if mx>0; A./=mx; end
        return A
    end
end

function code_hist(hard_idx::Vector{Int}, c::Int)
    cnt = zeros(Int, c); @inbounds for x in hard_idx; if 1≤x≤c; cnt[x]+=1; end; end
    cnt
end

end # CPLearnCodes

# active code projector
global CPL_CP = CPLearnCodes.CodeProj(CPLearnCodes.make_codebook(256, 4096), 0.07f0)

function qvnm_with_codes(W::AbstractMatrix{<:Real}, H::AbstractMatrix{<:Real},
                         cp::CPLearnCodes.CodeProj; lambda_code::Float64=0.25, hard::Bool=false)
    P, hard_idx = CPLearnCodes.project_codes(H, cp)
    A = CPLearnCodes.code_affinity(P, hard_idx; hard=hard)
    Wf = (1.0 - lambda_code) .* Matrix{Float32}(W) .+ lambda_code .* Matrix{Float32}(A)
    hist = CPLearnCodes.code_hist(hard_idx, size(cp.W,2))
    Wf, P, hard_idx, hist
end

##############################
# (E) DP (compact) — optional
##############################
module DPCollapse
export dp_diffusion_frictionless, dp_sigma_eq_smallbeta, dp_beta_critical
function dp_diffusion_frictionless(m,R0; G,ħ)
    (G*ħ*m^2) / (3*sqrt(pi)*R0^3)
end
function dp_sigma_eq_smallbeta(m,R0,ω,β; ħ)
    (4/(ω*β)) * (1 + (3*ħ^2*β)/(4*m*R0^2))
end
function dp_beta_critical(m,R0,ω; ħ)
    denom = (ħ*ω - 3*ħ^2/(m*R0^2))
    4/denom
end
end

##############################
# Handlers
##############################
router = HTTP.Router()

# Health
HTTP.register!(router, "GET", "/health", r->HTTP.Response(200, "ok"))

# (1) Estimate ID/entropy
using .CostaHeroID
function handle_estimate_id(req::HTTP.Request)
    o = JSON3.read(String(req.body))
    d = Int(o["d"]); N = Int(o["N"])
    V = reshape(Vector{Float32}(o["V"]), d, N) |> Array{Float64}
    k = Int(get(o, "k", 10)); γ = Float64(get(o,"gamma",0.5)); α = Float64(get(o,"alpha",0.5));
    boots = Int(get(o,"boots",8)); mode = String(get(o,"mode","local"))
    if mode=="global"
        m,H,diag = CostaHeroID.id_entropy_global(V; k=k, γ=γ, α=α, boots=boots)
        return HTTP.Response(200, JSON3.write(Dict("mode"=>"global","m_hat"=>m,"H_hat"=>H,"diag"=>Dict(diag))))
    else
        r = Int(get(o,"r",64))
        m,H,diag = CostaHeroID.id_entropy_local(V; k=k, γ=γ, α=α, boots=boots, r=r)
        return HTTP.Response(200, JSON3.write(Dict("mode"=>"local","m_hat"=>m,"H_hat"=>H,"diag"=>Dict(diag))))
    end
end
HTTP.register!(router, "POST", "/qvnm/estimate_id", handle_estimate_id)

# (2) Build QVNM
using .QVNM
function qvnm_build(o)
    # expects: V(d×N col-major), neighbors, weights, m_hat, H_hat, lambda_m, lambda_h
    d = Int(o["d"]); N = Int(o["N"])
    V = reshape(Vector{Float32}(o["V"]), d, N) |> Array{Float64}
    nei = [Vector{Int}(x) for x in o["neighbors"]]
    wts = [Vector{Float64}(x) for x in o["weights"]]
    Dg = QVNM.geodesic_dist(nei, wts)
    m = [Float64(x) for x in o["m_hat"]]; H = [Float64(x) for x in o["H_hat"]]
    lam_m = Float64(get(o,"lambda_m",0.3)); lam_h = Float64(get(o,"lambda_h",0.3))
    W = QVNM.blend_weights(V, Dg, m, H; k=min(10, size(Dg,2)-1), lam_m=lam_m, lam_h=lam_h)
    Dict("W"=>W)
end

function handle_build(req::HTTP.Request)
    o = JSON3.read(String(req.body))
    out = qvnm_build(o)
    HTTP.Response(200, JSON3.write(out))
end
HTTP.register!(router, "POST", "/qvnm/build", handle_build)

# (3) Preview
using .QVNMPreview
function handle_preview(req::HTTP.Request)
    o = JSON3.read(String(req.body))
    mode = String(get(o,"mode","W"))
    r    = Int(get(o,"r",2)); k_eval = Int(get(o,"k_eval",10)); bins = Int(get(o,"bins",20))
    if mode=="build"
        out = qvnm_build(o)
        W = Array{Float64}(out["W"])
        m = [Float64(x) for x in o["m_hat"]]; H = [Float64(x) for x in o["H_hat"]]
        prev = QVNMPreview.preview_from_W(W, m, H; r=r, k_eval=k_eval, bins=bins)
        return HTTP.Response(200, JSON3.write(prev))
    else
        N = Int(o["N"]); Wvec = Vector{Float64}(o["W"])
        W = reshape(permutedims(reshape(Wvec,(N,N))), (N,N))
        m = [Float64(x) for x in o["m_hat"]]; H = [Float64(x) for x in o["H_hat"]]
        prev = QVNMPreview.preview_from_W(W, m, H; r=r, k_eval=k_eval, bins=bins)
        return HTTP.Response(200, JSON3.write(prev))
    end
end
HTTP.register!(router, "POST", "/qvnm/preview", handle_preview)

# (4) Query (diffusion walk with optional prior)
function row_stochastic(W)
    N = size(W,1)
    P = Matrix{Float64}(W); rs = sum(P; dims=2)
    for i in 1:N
        s = rs[i]
        if s>0; P[i,:] ./= s; end
    end
    P
end
function diffusion_walk(P, psi0; alpha=0.85, steps=10, theta=0.0, prior=nothing)
    N = size(P,1)
    ψ = collect(Float64, psi0); s0=sum(abs.(ψ)); ψ ./= (s0>0 ? s0 : 1.0)
    blend = copy(ψ)
    if prior !== nothing
        ρ = collect(Float64, prior); ρ .-= minimum(ρ); sρ=sum(ρ); ρ./=(sρ>0 ? sρ : 1.0/N)
        @. blend = (1-theta)*ψ + theta*ρ
    end
    for _ in 1:max(1,steps)
        ψ = alpha * (P * ψ) .+ (1-alpha)*blend
    end
    s = sum(abs.(ψ)); ψ./=(s>0 ? s : 1.0)
    ψ
end
function topk(x::AbstractVector{<:Real}, k::Int)
    N=length(x); k=min(max(k,1),N)
    idx = partialsortperm(x, rev=true, 1:k)
    [(i, x[i]) for i in idx]
end

function handle_query(req::HTTP.Request)
    o = JSON3.read(String(req.body))
    mode = String(get(o,"mode","W"))
    topK = Int(get(o,"topk",5)); steps=Int(get(o,"steps",10)); alpha=Float64(get(o,"alpha",0.85))
    theta=Float64(get(o,"theta",0.0))
    ids = get(o,"ids", nothing)

    W = nothing
    if mode=="build"
        out = qvnm_build(o); W = Array{Float64}(out["W"])
    else
        N = Int(o["N"]); Wvec = Vector{Float64}(o["W"])
        W = reshape(permutedims(reshape(Wvec,(N,N))), (N,N))
    end

    N = size(W,1); Pmat = row_stochastic(W)
    ψ0 = fill(1.0/N, N)
    if haskey(o,"seed_id") && ids !== nothing && o["seed_id"] !== nothing
        idlist = [String(x) for x in ids]; seed = String(o["seed_id"])
        pos = findfirst(==(seed), idlist)
        if pos !== nothing; ψ0 .= 0.0; ψ0[pos]=1.0; end
    end

    prior = haskey(o,"prior") ? [Float64(x) for x in o["prior"]] : nothing
    ψT = diffusion_walk(Pmat, ψ0; alpha=alpha, steps=steps, theta=theta, prior=prior)
    winners = topk(ψT, topK)
    results = Any[]
    if ids !== nothing
        idlist = [String(x) for x in ids]
        for (i,s) in winners; push!(results, Dict("id"=>idlist[i], "idx"=>i, "score"=>s)); end
    else
        for (i,s) in winners; push!(results, Dict("idx"=>i, "score"=>s)); end
    end
    HTTP.Response(200, JSON3.write(Dict("top"=>results,"meta"=>Dict("N"=>N,"steps"=>steps,"alpha"=>alpha,"theta"=>theta,"mode"=>mode))))
end
HTTP.register!(router, "POST", "/qvnm/query", handle_query)

# (4b) Query trajectory: return ψ for each step (length = steps+1)
function handle_query_traj(req::HTTP.Request)
    o = JSON3.read(String(req.body))
    mode   = String(get(o, "mode", "build"))
    steps  = Int(get(o, "steps", 10))
    alpha  = Float64(get(o, "alpha", 0.85))
    theta  = Float64(get(o, "theta", 0.0))
    ids    = get(o, "ids", nothing)

    # Build/accept W
    W = if mode == "build"
        Array{Float64}(qvnm_build(o)["W"])
    else
        N = Int(o["N"])
        Wvec = Vector{Float64}(o["W"])
        reshape(permutedims(reshape(Wvec, (N, N))), (N, N))
    end
    N = size(W, 1)
    Pmat = row_stochastic(W)

    # Seed ψ0
    ψ0 = fill(1.0 / N, N)
    if haskey(o, "seed_id") && ids !== nothing && o["seed_id"] !== nothing
        idlist = [String(x) for x in ids]; seed = String(o["seed_id"])
        pos = findfirst(==(seed), idlist)
        if pos !== nothing
            ψ0 .= 0.0; ψ0[pos] = 1.0
        end
    end
    prior = haskey(o, "prior") ? [Float64(x) for x in o["prior"]] : nothing

    # Roll out trajectory
    traj = Vector{Vector{Float64}}(undef, steps + 1)
    ψ = copy(ψ0); traj[1] = copy(ψ)
    for t in 1:steps
        ψ = diffusion_walk(Pmat, ψ; alpha = alpha, steps = 1, theta = theta, prior = prior)
        traj[t + 1] = copy(ψ)
    end

    return HTTP.Response(200, JSON3.write(Dict(
        "N" => N, "steps" => steps, "alpha" => alpha, "theta" => theta, "traj" => traj
    )))
end
HTTP.register!(router, "POST", "/qvnm/query_traj", handle_query_traj)


# (5) Build+codes
function handle_build_codes(req::HTTP.Request)
    o = JSON3.read(String(req.body))
    λc = Float64(get(o,"lambda_code",0.25)); hard = Bool(get(o,"hard",false))
    out = qvnm_build(o); W = Array{Float32}(out["W"])
    d = Int(o["d"]); N = Int(o["N"])
    V = reshape(Vector{Float32}(o["V"]), d, N) # d×N
    H = tanh.(Matrix{Float32}(V'))            # N×d
    Wf,P,hard_idx,hist = qvnm_with_codes(W, H, CPL_CP; lambda_code=λc, hard=hard)
    HTTP.Response(200, JSON3.write(Dict("W"=>Wf,"codes"=>Dict("hist"=>hist,"hard_idx"=>hard_idx,"f"=>size(CPL_CP.W,1),"c"=>size(CPL_CP.W,2),"tau"=>CPL_CP.τ))))
end
HTTP.register!(router, "POST", "/qvnm/build_codes", handle_build_codes)

# (6) Codes init
function handle_cpl_init(req::HTTP.Request)
    o = JSON3.read(String(req.body))
    f = Int(get(o,"f",256)); c = Int(get(o,"c",4096)); τ = Float32(get(o,"tau",0.07)); seed=Int(get(o,"seed",2214))
    global CPL_CP = CPLearnCodes.CodeProj(CPLearnCodes.make_codebook(f,c;seed=seed), τ)
    HTTP.Response(200, JSON3.write(Dict("ok"=>true,"f"=>f,"c"=>c,"tau"=>τ)))
end
HTTP.register!(router, "POST", "/cpl/init", handle_cpl_init)

# (7) DP summary
using .DPCollapse
function handle_dp_summary(req::HTTP.Request)
    o = JSON3.read(String(req.body))
    m=Float64(get(o,"m",1.0)); R0=Float64(get(o,"R0",1e-9)); ω=Float64(get(o,"omega",1.0)); β=Float64(get(o,"beta",1e-5))
    G=Float64(get(o,"G",6.67430e-11)); ħ=Float64(get(o,"hbar",1.054571817e-34))
    Df = DPCollapse.dp_diffusion_frictionless(m,R0; G=G,ħ=ħ)
    σeq2 = DPCollapse.dp_sigma_eq_smallbeta(m,R0,ω,β; ħ=ħ)
    βc = DPCollapse.dp_beta_critical(m,R0,ω; ħ=ħ)
    HTTP.Response(200, JSON3.write(Dict("frictionless_D"=>Df,"sigma_eq2"=>σeq2,"beta_c"=>βc)))
end
HTTP.register!(router, "POST", "/dp/summary", handle_dp_summary)

# Serve
port = parse(Int, get(ENV,"PORT","9000"))
println("PolyServe+QVNM on :$port")
HTTP.serve(router, ip"0.0.0.0", port)

# Julia deps (install from REPL):
# using Pkg; Pkg.add(["HTTP","JSON3","NearestNeighbors","Statistics","StatsBase","LinearAlgebra","Graphs","SparseArrays"])
#!/usr/bin/env python3

from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, FileResponse

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, FileResponse


from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import tempfile
import aiofiles
import json
import csv
import io
import time


> main
import numpy as np
import httpx
import uuid
from typing import List, Dict, Any

app = FastAPI()
app.mount(
    "/ui",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "..", "static")),
    name="ui",
)

JULIA_BASE = os.environ.get("JULIA_BASE", "http://localhost:9000")

# Session memory: sid -> { V (d×N), ids (N), d, N, neighbors, weights, m_hat, H_hat }
QSESS: Dict[str, Dict[str, Any]] = {}


@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    return '<meta http-equiv="refresh" content="0;url=/ui/qvnm.html" />'


# ---------- helpers ----------
def _ensure_float32_col_unit(V: np.ndarray) -> np.ndarray:
    # V can be d×N or N×d; we expect d×N
    if V.ndim != 2:
        raise ValueError("Vectors must be a 2D array")
    d0, d1 = V.shape
    # if looks like N×d (more rows than cols), assume transpose
    if d0 > d1:
        V = V.T
    V = V.astype(np.float32, copy=False)
    norms = np.linalg.norm(V, axis=0, keepdims=True) + 1e-12
    V = V / norms
    return V


def _knn_graph_from_V(V: np.ndarray, k: int = 10) -> tuple[list[list[int]], list[list[float]]]:
    # V is d×N unit-norm
    d, N = V.shape
    # cosine sim
    S = (V.T @ V).astype(np.float64)
    # ensure numerical cleanliness
    np.clip(S, -1.0, 1.0, out=S)
    neighbors: list[list[int]] = []
    weights: list[list[float]] = []
    # distance on sphere: chordal distance sqrt(2-2cos)
    for i in range(N):
        sims = S[i]
        # exclude self by setting to -inf
        sims_i = sims.copy()
        sims_i[i] = -np.inf
        # top-k by similarity
        nn_idx = np.argpartition(-sims_i, kth=min(k, N - 1) - 1)[: min(k, N - 1)]
        # sort by similarity desc
        nn_idx = nn_idx[np.argsort(-sims_i[nn_idx])]
        # convert to 1-based for Julia
        nbs = (nn_idx + 1).tolist()
        # weights are distances (cost)
        dist = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * sims_i[nn_idx])).tolist()
        neighbors.append(nbs)
        weights.append(dist)
    return neighbors, weights


async def _post_json(url: str, payload: dict) -> dict:
    async with httpx.AsyncClient(timeout=120.0) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        return r.json()

# ---------- text helpers ----------
def _texts_for_ids(sess: Dict[str, Any], id_list: list[str]) -> list[dict[str, str]]:
    ids: list[str] = sess.get("ids", []) or []
    texts: list[str] = sess.get("texts", []) or []
    pos: dict[str, int] = {ids[i]: i for i in range(len(ids))}
    out: list[dict[str, str]] = []
    for _id in id_list:
        i = pos.get(_id)
        if i is not None and i < len(texts):
            out.append({"id": _id, "text": texts[i]})
    return out


# ---------- text helpers ----------
def _texts_for_ids(sess: Dict[str, Any], id_list: list[str]) -> list[dict[str, str]]:
    ids: list[str] = sess.get("ids", []) or []
    texts: list[str] = sess.get("texts", []) or []
    pos: dict[str, int] = {ids[i]: i for i in range(len(ids))}
    out: list[dict[str, str]] = []
    for _id in id_list:
        i = pos.get(_id)
        if i is not None and i < len(texts):
            out.append({"id": _id, "text": texts[i]})
    return out


# ---------- upload vectors (.jsonl or .npz) ----------
@app.post("/qvnm/upload_vectors")
async def qvnm_upload_vectors(file: UploadFile = File(...)) -> JSONResponse:
    fd, tmp = tempfile.mkstemp(suffix=os.path.splitext(file.filename or "")[1])
    os.close(fd)
    try:
        async with aiofiles.open(tmp, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                await f.write(chunk)
        ids: list[str] = []
        V: np.ndarray | None = None
        if tmp.endswith(".jsonl"):
            vecs: list[np.ndarray] = []
            texts: list[str] = []
            async with aiofiles.open(tmp, "r", encoding="utf-8") as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    ids.append(str(rec.get("id", f"id{len(ids)}")))
                    vecs.append(np.asarray(rec["vector"], dtype=np.float32))
                    texts.append(str(rec.get("text", "")))
            M = np.stack(vecs, axis=0)
            M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
            V = M.T.astype(np.float32, copy=False)  # d×N


                    texts.append(str(rec.get("text", "")))
            M = np.stack(vecs, axis=0)
            M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
            V = M.T.astype(np.float32, copy=False)  # d×N
> main
            TEXTS = texts
        elif tmp.endswith(".npz"):
            dat = np.load(tmp, allow_pickle=False)
            if "V" in dat:
                V = _ensure_float32_col_unit(np.array(dat["V"]))
                d, N = V.shape
                if "ids" in dat:
                    ids = list(map(str, dat["ids"]))
                else:
                    ids = [f"id{i}" for i in range(N)]
            elif "vectors" in dat:
                M = np.array(dat["vectors"]).astype(np.float32, copy=False)  # N×d
                M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
                V = M.T
                if "ids" in dat:
                    ids = list(map(str, dat["ids"]))
                else:
                    ids = [f"id{i}" for i in range(M.shape[0])]
            elif "X" in dat:
                M = np.array(dat["X"]).astype(np.float32, copy=False)
                M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
                V = M.T
                if "ids" in dat:
                    ids = list(map(str, dat["ids"]))
                else:
                    ids = [f"id{i}" for i in range(M.shape[0])]
            else:
                raise ValueError(".npz must contain 'V' (d×N) or 'vectors'/'X' (N×d)")
        else:
            raise ValueError("Unsupported file type. Use .jsonl or .npz")

        assert V is not None
        d, N = int(V.shape[0]), int(V.shape[1])
        # truncate/extend ids
        if len(ids) < N:
            ids += [f"id{i}" for i in range(len(ids), N)]
        elif len(ids) > N:
            ids = ids[:N]
        sid = uuid.uuid4().hex
        QSESS[sid] = {
            "V": V,
            "ids": ids,

            "texts": locals().get("TEXTS", []),

            "texts": locals().get("TEXTS", [])
> main
            "d": d,
            "N": N,
            "neighbors": None,
            "weights": None,
            "m_hat": None,
            "H_hat": None,
        }
        return JSONResponse({"sid": sid, "d": d, "N": N, "ids_head": ids[:5]})
    finally:
        try:
            os.remove(tmp)
        except Exception:
            pass


# ---------- proxies ----------
@app.post("/qvnm/estimate_id")
async def proxy_estimate_id(payload: Dict[str, Any]) -> JSONResponse:
    sid = payload.get("sid")
    if not sid or sid not in QSESS:
        return JSONResponse({"error": "missing or invalid sid"}, status_code=400)
    sess = QSESS[sid]
    V = sess["V"]
    d, N = int(V.shape[0]), int(V.shape[1])
    req = {
        "d": d,
        "N": N,
        "V": V.astype(np.float32, copy=False).ravel(order="F").tolist(),
        "k": int(payload.get("k", 10)),
        "gamma": float(payload.get("gamma", 0.5)),
        "alpha": float(payload.get("alpha", 0.5)),
        "boots": int(payload.get("boots", 8)),
        "mode": payload.get("mode", "local"),
    }
    if req["mode"] == "local":
        req["r"] = int(payload.get("r", 64))
    out = await _post_json(f"{JULIA_BASE}/qvnm/estimate_id", req)

    mode = out.get("mode", "local")
    if mode == "global":
        # expand scalars to vectors
        m_hat = float(out.get("m_hat", 0.0))
        H_hat = float(out.get("H_hat", 0.0))
        sess["m_hat"] = [m_hat] * N
        sess["H_hat"] = [H_hat] * N
    else:
        # ensure numeric lists and fill NaNs with means
        m_hat = np.array(out.get("m_hat", [0.0] * N), dtype=np.float64)
        H_hat = np.array(out.get("H_hat", [0.0] * N), dtype=np.float64)
        # replace non-finite
        for arr in (m_hat, H_hat):
            mask = ~np.isfinite(arr)
            if mask.any():
                arr[mask] = float(np.nanmean(arr[~mask])) if (~mask).any() else 0.0
        sess["m_hat"] = m_hat.astype(float).tolist()
        sess["H_hat"] = H_hat.astype(float).tolist()
    return JSONResponse({"sid": sid, **out})


@app.post("/qvnm/build_preview")
async def proxy_build_preview(payload: Dict[str, Any]) -> JSONResponse:
    sid = payload.get("sid")
    if not sid or sid not in QSESS:
        return JSONResponse({"error": "missing or invalid sid"}, status_code=400)
    sess = QSESS[sid]
    V = sess["V"]
    d, N = int(V.shape[0]), int(V.shape[1])
    # ensure estimates
    if not sess.get("m_hat") or not sess.get("H_hat"):
        return JSONResponse({"error": "run estimate_id first"}, status_code=400)
    # ensure knn graph
    k_graph = int(payload.get("knn_k", 10))
    if sess.get("neighbors") is None or sess.get("weights") is None or sess.get("_k_graph") != k_graph:
        nei, wts = _knn_graph_from_V(V, k=k_graph)
        sess["neighbors"], sess["weights"], sess["_k_graph"] = nei, wts, k_graph
    req = {
        "mode": "build",
        "d": d,
        "N": N,
        "V": V.astype(np.float32, copy=False).ravel(order="F").tolist(),
        "neighbors": sess["neighbors"],
        "weights": sess["weights"],
        "m_hat": sess["m_hat"],
        "H_hat": sess["H_hat"],
        "lambda_m": float(payload.get("lambda_m", 0.3)),
        "lambda_h": float(payload.get("lambda_h", 0.3)),
        "r": int(payload.get("r", 2)),
        "k_eval": int(payload.get("k_eval", 10)),
        "bins": int(payload.get("bins", 20)),
    }
    out = await _post_json(f"{JULIA_BASE}/qvnm/preview", req)
    # cache preview for exports
    sess["preview"] = out
    return JSONResponse({"sid": sid, **out})

    # cache preview for exports
    sess["preview"] = out
    return JSONResponse({"sid": sid, **out})


# ---------- Qwen optimize: retrieve context and propose diffs ----------
async def run_qwen(prompt: str, model_id: str | None = None, max_new_tokens: int = 800) -> str:
    """
    - If MODEL_ENDPOINT is set, POST there as a simple JSON API {prompt, max_new_tokens}
    - Else try local transformers using QWEN_MODEL or model_id.
    """
    endpoint = os.environ.get("MODEL_ENDPOINT")
    if endpoint:
        async with httpx.AsyncClient(timeout=300.0) as cx:
            r = await cx.post(endpoint, json={"prompt": prompt, "max_new_tokens": max_new_tokens})
            r.raise_for_status()
            try:
                j = r.json()
                return j.get("text") or j.get("output") or r.text
            except Exception:
                return r.text

    # Local transformers fallback
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    mid = model_id or os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        mid,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    msgs = [
        {"role": "system", "content": "You are a senior engineer. Be concise; return unified diffs when changing code."},
        {"role": "user", "content": prompt},
    ]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0], skip_special_tokens=True)


def build_prompt(goal: str, seed_id: str, ctx_blobs: list[dict]) -> str:
    header = (
        "You are given code/docs context. Task: propose concrete performance, safety, or readability optimizations relevant to the goal. Return:\n"
        "1) a brief rationale; 2) prioritized checklist; 3) unified diffs for files you modify; 4) tests if applicable.\n\n"
        f"GOAL:\n{goal}\n\n"
        f"SEED: {seed_id}\n\n"
        "CONTEXT (IDs + excerpts):\n"
    )
    parts: list[str] = []
    for blob in ctx_blobs:
        txt = str(blob.get("text", ""))[:3000]
        parts.append(f"--- {blob.get('id','')} ---\n{txt}\n")
    return header + "\n".join(parts)


@app.post("/pilot/optimize")
async def pilot_optimize(
    session: str,
    seed_id: str,
    goal: str = Body(..., embed=True),
    topk: int = 8,
    steps: int = 10,
    alpha: float = 0.85,
    theta: float = 0.0,
    model_id: str | None = None,
):
    sess = QSESS.get(session)
    if not sess:
        return JSONResponse({"error": "bad session"}, status_code=400)
    V = sess["V"]
    d, N = int(V.shape[0]), int(V.shape[1])
    # ensure graph and estimates
    if sess.get("neighbors") is None or sess.get("weights") is None:
        nei, wts = _knn_graph_from_V(V, k=10)
        sess["neighbors"], sess["weights"] = nei, wts
    if not sess.get("m_hat") or not sess.get("H_hat"):
        sess["m_hat"], sess["H_hat"] = [0.0] * N, [0.0] * N

    req = {
        "mode": "build",
        "d": d,
        "N": N,
        "V": V.astype(np.float32, copy=False).ravel(order="F").tolist(),
        "neighbors": sess["neighbors"],
        "weights": sess["weights"],
        "m_hat": sess["m_hat"],
        "H_hat": sess["H_hat"],
        "ids": sess.get("ids"),
        "seed_id": seed_id,
        "topk": int(topk),
        "steps": int(steps),
        "alpha": float(alpha),
        "theta": float(theta),
    }
    ans = await _post_json(f"{JULIA_BASE}/qvnm/query", req)

    top_ids = [x.get("id") for x in (ans.get("top") or []) if x.get("id")]
    ctx = _texts_for_ids(sess, top_ids)
    prompt = build_prompt(goal, seed_id, ctx)
    llm_out = await run_qwen(prompt, model_id=model_id, max_new_tokens=1200)
    return JSONResponse({
        "seed": seed_id,
        "goal": goal,
        "top": ans.get("top"),
        "qwen": llm_out,
    })

# ---------- Qwen optimize: retrieve context and propose diffs ----------
async def run_qwen(prompt: str, model_id: str | None = None, max_new_tokens: int = 800) -> str:
    """
    - If MODEL_ENDPOINT is set, POST there as a simple JSON API {prompt, max_new_tokens}
    - Else try local transformers using QWEN_MODEL or model_id.
    """
    endpoint = os.environ.get("MODEL_ENDPOINT")
    if endpoint:
        async with httpx.AsyncClient(timeout=300.0) as cx:
            r = await cx.post(endpoint, json={"prompt": prompt, "max_new_tokens": max_new_tokens})
            r.raise_for_status()
            try:
                j = r.json()
                return j.get("text") or j.get("output") or r.text
            except Exception:
                return r.text

    # Local transformers fallback
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    mid = model_id or os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-Coder-7B-Instruct")
    tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        mid,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    msgs = [
        {"role": "system", "content": "You are a senior engineer. Be concise; return unified diffs when changing code."},
        {"role": "user", "content": prompt},
    ]
    inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
    out = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0], skip_special_tokens=True)


def build_prompt(goal: str, seed_id: str, ctx_blobs: list[dict]) -> str:
    header = (
        "You are given code/docs context. Task: propose concrete performance, safety, or readability optimizations relevant to the goal. Return:\n"
        "1) a brief rationale; 2) prioritized checklist; 3) unified diffs for files you modify; 4) tests if applicable.\n\n"
        f"GOAL:\n{goal}\n\n"
        f"SEED: {seed_id}\n\n"
        "CONTEXT (IDs + excerpts):\n"
    )
    parts: list[str] = []
    for blob in ctx_blobs:
        txt = str(blob.get("text", ""))[:3000]
        parts.append(f"--- {blob.get('id','')} ---\n{txt}\n")
    return header + "\n".join(parts)


@app.post("/pilot/optimize")
async def pilot_optimize(
    session: str,
    seed_id: str,
    goal: str = Body(..., embed=True),
    topk: int = 8,
    steps: int = 10,
    alpha: float = 0.85,
    theta: float = 0.0,
    model_id: str | None = None,
):
    sess = QSESS.get(session)
    if not sess:
        return JSONResponse({"error": "bad session"}, status_code=400)
    V = sess["V"]
    d, N = int(V.shape[0]), int(V.shape[1])
    # ensure graph and estimates
    if sess.get("neighbors") is None or sess.get("weights") is None:
        nei, wts = _knn_graph_from_V(V, k=10)
        sess["neighbors"], sess["weights"] = nei, wts
    if not sess.get("m_hat") or not sess.get("H_hat"):
        sess["m_hat"], sess["H_hat"] = [0.0] * N, [0.0] * N

    req = {
        "mode": "build",
        "d": d,
        "N": N,
        "V": V.astype(np.float32, copy=False).ravel(order="F").tolist(),
        "neighbors": sess["neighbors"],
        "weights": sess["weights"],
        "m_hat": sess["m_hat"],
        "H_hat": sess["H_hat"],
        "ids": sess.get("ids"),
        "seed_id": seed_id,
        "topk": int(topk),
        "steps": int(steps),
        "alpha": float(alpha),
        "theta": float(theta),
    }
    ans = await _post_json(f"{JULIA_BASE}/qvnm/query", req)

    top_ids = [x.get("id") for x in (ans.get("top") or []) if x.get("id")]
    ctx = _texts_for_ids(sess, top_ids)
    prompt = build_prompt(goal, seed_id, ctx)
    llm_out = await run_qwen(prompt, model_id=model_id, max_new_tokens=1200)
    return JSONResponse({
        "seed": seed_id,
        "goal": goal,
        "top": ans.get("top"),
        "qwen": llm_out,
    })

@app.post("/qvnm/query")
async def proxy_query(payload: Dict[str, Any]) -> JSONResponse:
    sid = payload.get("sid")
    if not sid or sid not in QSESS:
        return JSONResponse({"error": "missing or invalid sid"}, status_code=400)
    sess = QSESS[sid]
    V = sess["V"]
    d, N = int(V.shape[0]), int(V.shape[1])
    if sess.get("neighbors") is None or sess.get("weights") is None:
        nei, wts = _knn_graph_from_V(V, k=int(payload.get("knn_k", 10)))
        sess["neighbors"], sess["weights"] = nei, wts
    if not sess.get("m_hat") or not sess.get("H_hat"):
        # default to zeros if not estimated, to allow demo
        sess["m_hat"] = [0.0] * N
        sess["H_hat"] = [0.0] * N
    req = {
        "mode": "build",
        "d": d,
        "N": N,
        "V": V.astype(np.float32, copy=False).ravel(order="F").tolist(),
        "neighbors": sess["neighbors"],
        "weights": sess["weights"],
        "m_hat": sess["m_hat"],
        "H_hat": sess["H_hat"],
        "ids": sess["ids"],
        "seed_id": payload.get("seed_id"),
        "topk": int(payload.get("topk", 5)),
        "steps": int(payload.get("steps", 10)),
        "alpha": float(payload.get("alpha", 0.85)),
        "theta": float(payload.get("theta", 0.0)),
    }
    if "prior" in payload and payload["prior"] is not None:
        req["prior"] = payload["prior"]
    out = await _post_json(f"{JULIA_BASE}/qvnm/query", req)
    return JSONResponse({"sid": sid, **out})

> main
@app.post("/qvnm/query_traj")
async def proxy_query_traj(payload: Dict[str, Any]) -> JSONResponse:
    sid = payload.get("sid")
    if not sid or sid not in QSESS:
        return JSONResponse({"error": "missing or invalid sid"}, status_code=400)
    sess = QSESS[sid]
    V = sess["V"]
    d, N = int(V.shape[0]), int(V.shape[1])
    if sess.get("neighbors") is None or sess.get("weights") is None:
        nei, wts = _knn_graph_from_V(V, k=int(payload.get("knn_k", 10)))
        sess["neighbors"], sess["weights"] = nei, wts
    if not sess.get("m_hat") or not sess.get("H_hat"):
        sess["m_hat"] = [0.0] * N
        sess["H_hat"] = [0.0] * N
    req = {
        "mode": "build",
        "d": d,
        "N": N,
        "V": V.astype(np.float32, copy=False).ravel(order="F").tolist(),
        "neighbors": sess["neighbors"],
        "weights": sess["weights"],
        "m_hat": sess["m_hat"],
        "H_hat": sess["H_hat"],
        "ids": sess["ids"],
        "seed_id": payload.get("seed_id"),
        "steps": int(payload.get("steps", 20)),
        "alpha": float(payload.get("alpha", 0.85)),
        "theta": float(payload.get("theta", 0.0)),
    }
    if "prior" in payload and payload["prior"] is not None:
        req["prior"] = payload["prior"]
    out = await _post_json(f"{JULIA_BASE}/qvnm/query_traj", req)
    return JSONResponse({"sid": sid, **out})
> main
@app.post("/qvnm/build_codes")
async def proxy_build_codes(payload: Dict[str, Any]) -> JSONResponse:
    sid = payload.get("sid")
    if not sid or sid not in QSESS:
        return JSONResponse({"error": "missing or invalid sid"}, status_code=400)
    sess = QSESS[sid]
    V = sess["V"]
    d, N = int(V.shape[0]), int(V.shape[1])
    if sess.get("neighbors") is None or sess.get("weights") is None:
        nei, wts = _knn_graph_from_V(V, k=int(payload.get("knn_k", 10)))
        sess["neighbors"], sess["weights"] = nei, wts
    if not sess.get("m_hat") or not sess.get("H_hat"):
        sess["m_hat"] = [0.0] * N
        sess["H_hat"] = [0.0] * N
    req = {
        "d": d,
        "N": N,
        "V": V.astype(np.float32, copy=False).ravel(order="F").tolist(),
        "neighbors": sess["neighbors"],
        "weights": sess["weights"],
        "m_hat": sess["m_hat"],
        "H_hat": sess["H_hat"],
        "lambda_code": float(payload.get("lambda_code", 0.25)),
        "hard": bool(payload.get("hard", False)),
    }
    out = await _post_json(f"{JULIA_BASE}/qvnm/build_codes", req)
    # cache codes summary for exports
    sess["codes"] = out.get("codes")
    return JSONResponse({"sid": sid, **out})

    # cache codes summary for exports
    sess["codes"] = out.get("codes")
    return JSONResponse({"sid": sid, **out})


# --- helpers to rebuild W and get coords from last preview ---
def _last_coords(sess: Dict[str, Any]) -> tuple[list[float], int]:
    prev = sess.get("preview") or {}
    em = prev.get("eigenmaps") or {}
    coords = em.get("coords") or []
    r = int(em.get("r", 0))
    return coords, r


async def _rebuild_W(sess: Dict[str, Any], k: int = 10, lambda_m: float = 0.3, lambda_h: float = 0.3) -> list[list[float]]:
    V = sess["V"]
    d, N = int(V.shape[0]), int(V.shape[1])
    nei, wts = _knn_graph_from_V(V, k=k)
    req = {
        "d": d,
        "N": N,
        "V": V.astype(np.float32, copy=False).ravel(order="F").tolist(),
        "neighbors": nei,
        "weights": wts,
        "m_hat": sess.get("m_hat") or [0.0] * N,
        "H_hat": sess.get("H_hat") or [0.0] * N,
        "lambda_m": float(lambda_m),
        "lambda_h": float(lambda_h),
    }
    out = await _post_json(f"{JULIA_BASE}/qvnm/build", req)
    return out["W"]


# --- export as CSV/JSON files served by FastAPI ---
@app.get("/qvnm/export")
async def qvnm_export(session: str, kind: str = "coords_csv", k: int = 10, lambda_m: float = 0.3, lambda_h: float = 0.3, threshold: float = 0.0):
    """
    kind ∈ {coords_csv, codes_csv, W_json, edges_csv}
    - coords_csv: N rows: node_id, x, y  (if r>=2; else returns 1D y=0)
    - codes_csv:  counts per code index (if available from last code-blend)
    - W_json:     dense adjacency as JSON (float32)
    - edges_csv:  i,j,weight (only weights >= threshold)
    """
    sid = session
    if sid not in QSESS:
        return JSONResponse({"error": "bad session"}, status_code=400)
    sess = QSESS[sid]
    ids = sess.get("ids") or [str(i) for i in range(sess["V"].shape[1])]

    if kind == "coords_csv":
        coords, r = _last_coords(sess)
        N = len(ids)
        xs = [0.0] * N
        ys = [0.0] * N
        if r >= 1 and coords and len(coords) >= r * N:
            # coords is vec(coords') in Julia ⇒ order: for i in 0..N-1, for d in 0..r-1: coords[i*r + d]
            for i in range(N):
                xs[i] = float(coords[i * r + 0])
                ys[i] = float(coords[i * r + 1]) if r >= 2 else 0.0
        # build CSV
        sio = io.StringIO()
        w = csv.writer(sio)
        w.writerow(["node_id", "x", "y"])
        for nid, x, y in zip(ids, xs, ys):
            w.writerow([nid, f"{x:.6g}", f"{y:.6g}"])
        return PlainTextResponse(content=sio.getvalue(), media_type="text/csv")

    if kind == "codes_csv":
        codes = sess.get("codes") or {}
        hist = codes.get("hist") if isinstance(codes, dict) else None
        if hist is None:
            return JSONResponse({"error": "no codes available; run build_codes first"}, status_code=400)
        sio = io.StringIO()
        w = csv.writer(sio)
        w.writerow(["code", "count"])
        for idx, cnt in enumerate(hist, start=1):
            w.writerow([idx, int(cnt)])
        return PlainTextResponse(content=sio.getvalue(), media_type="text/csv")

    if kind == "W_json":
        W = await _rebuild_W(sess, k=int(k), lambda_m=float(lambda_m), lambda_h=float(lambda_h))
        return JSONResponse({"W": W})

    if kind == "edges_csv":
        W = await _rebuild_W(sess, k=int(k), lambda_m=float(lambda_m), lambda_h=float(lambda_h))
        thr = float(threshold)
        N = len(W)
        sio = io.StringIO()
        w = csv.writer(sio)
        w.writerow(["i", "j", "weight"])  # 1-based indices as in Julia
        for i in range(N):
            row = W[i]
            for j in range(i + 1, N):  # undirected edges once
                wij = float(row[j])
                if wij >= thr:
                    w.writerow([i + 1, j + 1, f"{wij:.6g}"])
        return PlainTextResponse(content=sio.getvalue(), media_type="text/csv")

    return JSONResponse({"error": "unknown kind"}, status_code=400)


# --- helpers to rebuild W and get coords from last preview ---
def _last_coords(sess: Dict[str, Any]) -> tuple[list[float], int]:
    prev = sess.get("preview") or {}
    em = prev.get("eigenmaps") or {}
    coords = em.get("coords") or []
    r = int(em.get("r", 0))
    return coords, r


async def _rebuild_W(sess: Dict[str, Any], k: int = 10, lambda_m: float = 0.3, lambda_h: float = 0.3) -> list[list[float]]:
    V = sess["V"]
    d, N = int(V.shape[0]), int(V.shape[1])
    nei, wts = _knn_graph_from_V(V, k=k)
    req = {
        "d": d,
        "N": N,
        "V": V.astype(np.float32, copy=False).ravel(order="F").tolist(),
        "neighbors": nei,
        "weights": wts,
        "m_hat": sess.get("m_hat") or [0.0] * N,
        "H_hat": sess.get("H_hat") or [0.0] * N,
        "lambda_m": float(lambda_m),
        "lambda_h": float(lambda_h),
    }
    out = await _post_json(f"{JULIA_BASE}/qvnm/build", req)
    return out["W"]


# --- export as CSV/JSON files served by FastAPI ---
@app.get("/qvnm/export")
async def qvnm_export(session: str, kind: str = "coords_csv", k: int = 10, lambda_m: float = 0.3, lambda_h: float = 0.3, threshold: float = 0.0):
    """
    kind ∈ {coords_csv, codes_csv, W_json, edges_csv}
    - coords_csv: N rows: node_id, x, y  (if r>=2; else returns 1D y=0)
    - codes_csv:  counts per code index (if available from last code-blend)
    - W_json:     dense adjacency as JSON (float32)
    - edges_csv:  i,j,weight (only weights >= threshold)
    """
    sid = session
    if sid not in QSESS:
        return JSONResponse({"error": "bad session"}, status_code=400)
    sess = QSESS[sid]
    ids = sess.get("ids") or [str(i) for i in range(sess["V"].shape[1])]

    if kind == "coords_csv":
        coords, r = _last_coords(sess)
        N = len(ids)
        xs = [0.0] * N
        ys = [0.0] * N
        if r >= 1 and coords and len(coords) >= r * N:
            # coords is vec(coords') in Julia ⇒ order: for i in 0..N-1, for d in 0..r-1: coords[i*r + d]
            for i in range(N):
                xs[i] = float(coords[i * r + 0])
                ys[i] = float(coords[i * r + 1]) if r >= 2 else 0.0
        # build CSV
        sio = io.StringIO()
        w = csv.writer(sio)
        w.writerow(["node_id", "x", "y"])
        for nid, x, y in zip(ids, xs, ys):
            w.writerow([nid, f"{x:.6g}", f"{y:.6g}"])
        return PlainTextResponse(content=sio.getvalue(), media_type="text/csv")

    if kind == "codes_csv":
        codes = sess.get("codes") or {}
        hist = codes.get("hist") if isinstance(codes, dict) else None
        if hist is None:
            return JSONResponse({"error": "no codes available; run build_codes first"}, status_code=400)
        sio = io.StringIO()
        w = csv.writer(sio)
        w.writerow(["code", "count"])
        for idx, cnt in enumerate(hist, start=1):
            w.writerow([idx, int(cnt)])
        return PlainTextResponse(content=sio.getvalue(), media_type="text/csv")

    if kind == "W_json":
        W = await _rebuild_W(sess, k=int(k), lambda_m=float(lambda_m), lambda_h=float(lambda_h))
        return JSONResponse({"W": W})

    if kind == "edges_csv":
        W = await _rebuild_W(sess, k=int(k), lambda_m=float(lambda_m), lambda_h=float(lambda_h))
        thr = float(threshold)
        N = len(W)
        sio = io.StringIO()
        w = csv.writer(sio)
        w.writerow(["i", "j", "weight"])  # 1-based indices as in Julia
        for i in range(N):
            row = W[i]
            for j in range(i + 1, N):  # undirected edges once
                wij = float(row[j])
                if wij >= thr:
                    w.writerow([i + 1, j + 1, f"{wij:.6g}"])
        return PlainTextResponse(content=sio.getvalue(), media_type="text/csv")

    return JSONResponse({"error": "unknown kind"}, status_code=400)


@app.post("/cpl/init")
async def proxy_cpl_init(payload: Dict[str, Any]) -> JSONResponse:
    req = {
        "f": int(payload.get("f", 256)),
        "c": int(payload.get("c", 4096)),
        "tau": float(payload.get("tau", 0.07)),
        "seed": int(payload.get("seed", 2214)),
    }
    out = await _post_json(f"{JULIA_BASE}/cpl/init", req)
    return JSONResponse(out)


@app.get("/qvnm/session/{sid}")
async def get_session_meta(sid: str) -> JSONResponse:
    if sid not in QSESS:
        return JSONResponse({"error": "not found"}, status_code=404)
    sess = QSESS[sid]
    return JSONResponse({
        "sid": sid,
        "d": sess["d"],
        "N": sess["N"],
        "ids_head": sess["ids"][:5],
        "has_graph": sess.get("neighbors") is not None,
        "has_estimates": sess.get("m_hat") is not None,
    })


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), reload=False)
#!/usr/bin/env python3
# dual_llm_wavecaster_enhanced.py
# SPDX-License-Identifier: MIT
"""
Enhanced Dual LLM WaveCaster with Neuro-Symbolic Adaptive Reflective Engine
----------------------------------------------------------------------------
Two-LLM orchestration (local final inference + remote resource-only summaries) → framed bits
→ modulated waveform (BFSK/BPSK/QPSK/16QAM/AFSK/OFDM) → WAV/IQ files (+ optional audio out)
with visualization, simple FEC, encryption, watermarking, and metadata.

Deps (minimum):
  pip install numpy scipy requests

Optional:
  pip install matplotlib sounddevice pycryptodome

Quick start:
  python dual_llm_wavecaster_enhanced.py modulate --text "hello airwaves" --scheme qpsk --wav --iq
  python dual_llm_wavecaster_enhanced.py cast --prompt "2-paragraph plan" \
      --resource-file notes.txt --local-url http://127.0.0.1:8080 --local-mode llama-cpp \
      --remote-url https://api.openai.com --remote-key $OPENAI_API_KEY --scheme bfsk --wav
"""

from __future__ import annotations
import argparse, base64, binascii, hashlib, json, logging, math, os, struct, sys, time, warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Callable
from enum import Enum, auto
from datetime import datetime

# ---------- Hard requirements ----------
try:
    import numpy as np
    from scipy import signal as sp_signal
    from scipy.fft import rfft, rfftfreq
except Exception as e:
    raise SystemExit("numpy and scipy are required: pip install numpy scipy") from e

# ---------- Optional dependencies ----------
try:
    import requests
except Exception:
    requests = None  # HTTP backends disabled if missing

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False

try:
    import sounddevice as sd
    HAS_AUDIO = True
except Exception:
    HAS_AUDIO = False

try:
    from Crypto.Cipher import AES
    from Crypto.Random import get_random_bytes
    from Crypto.Protocol.KDF import PBKDF2
    HAS_CRYPTO = True
except Exception:
    HAS_CRYPTO = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("wavecaster")

# =========================================================
# Enums / Config
# =========================================================

class ModulationScheme(Enum):
    BFSK = auto()
    BPSK = auto()
    QPSK = auto()
    QAM16 = auto()
    AFSK = auto()
    OFDM = auto()
    DSSS_BPSK = auto()

class FEC(Enum):
    NONE = auto()
    HAMMING74 = auto()
    REED_SOLOMON = auto()   # stub
    LDPC = auto()           # stub
    TURBO = auto()          # stub

@dataclass
class HTTPConfig:
    base_url: str
    api_key: Optional[str] = None
    model: Optional[str] = None
    timeout: int = 60
    mode: str = "openai-chat"  # ["openai-chat","openai-completions","llama-cpp","textgen-webui"]
    verify_ssl: bool = True
    max_retries: int = 2
    retry_delay: float = 0.8

@dataclass
class OrchestratorSettings:
    temperature: float = 0.7
    max_tokens: int = 512
    style: str = "concise"
    max_context_chars: int = 8000

@dataclass
class ModConfig:
    sample_rate: int = 48000
    symbol_rate: int = 1200
    amplitude: float = 0.7
    f0: float = 1200.0     # BFSK 0
    f1: float = 2200.0     # BFSK 1
    fc: float = 1800.0     # PSK/QAM audio carrier (for WAV)
    clip: bool = True
    # OFDM (toy)
    ofdm_subc: int = 64
    cp_len: int = 16
    # DSSS
    dsss_chip_rate: int = 4800

@dataclass
class FrameConfig:
    use_crc32: bool = True
    use_crc16: bool = False
    preamble: bytes = b"\x55" * 8  # 01010101 * 8
    version: int = 1

# =========================================================
# Neuro-Symbolic Adaptive Reflective Engine
# =========================================================

class EntropyAnalyzer:
    def measure(self, data: Any) -> float:
        s = str(data)
        if not s:
            return 0.0
        counts: Dict[str, int] = {}
        for c in s:
            counts[c] = counts.get(c, 0) + 1
        n = len(s)
        ent = 0.0
        for cnt in counts.values():
            p = cnt / n
            if p > 0:
                ent -= p * math.log2(p)
        return ent

class DianneReflector:
    def reflect(self, data: Any) -> Dict[str, Any]:
        patterns = self._detect_patterns(data)
        head = str(data)[:40].replace("\n", " ")
        if "high_repetition" in patterns:
            insight = f"Cyclical resonance detected in Reflecting essence of: {head}..."
        elif "hierarchical_structure" in patterns:
            insight = f"Nested reality layers within Reflecting essence of: {head}..."
        else:
            insight = f"Linear transformation potential in Reflecting essence of: {head}..."
        return {"insight": insight, "patterns": patterns, "symbolic_depth": self._depth(data)}
    
    def _detect_patterns(self, data: Any) -> List[str]:
        s = str(data)
        patterns = []
        if len(s) > 100 and len(set(s)) < 20:
            patterns.append("high_repetition")
        if s.count('\n') > 5 and any(c in s for c in ['{', '[', '(', '<']):
            patterns.append("hierarchical_structure")
        return patterns
    
    def _depth(self, data: Any) -> int:
        s = str(data)
        return min(10, len(s) // 100)

class MatrixTransformer:
    def project(self, data: Any) -> Dict[str, Any]:
        dims = self._analyze(data)
        h = hash(str(data)) & 0xFFFFFFFF
        rank = int(dims["rank"])
        eivals = [math.sin(h * 0.001 * i) for i in range(max(1, min(3, rank)))]
        return {
            "projected_rank": dims["rank"],
            "structure": dims["structure"],
            "eigenvalues": eivals,
            "determinant": math.cos(h * 0.0001),
            "trace": (math.tan(h * 0.00001) if (h % 100) else 0.0),
        }
    
    def _analyze(self, data: Any) -> Dict[str, Any]:
        s = str(data)
        return {
            "rank": min(10, len(s) // 50),
            "structure": "sparse" if len(set(s)) < 20 else "dense"
        }

class JuliaSymbolEngine:
    def analyze(self, data: Any) -> Dict[str, Any]:
        coeffs = self._coeffs(data)
        return {
            "chebyshev_polynomial": self._poly(coeffs),
            "coefficients": coeffs,
            "derivatives": self._derivs(coeffs),
            "critical_points": self._crit(coeffs),
        }
    
    def _coeffs(self, data: Any) -> List[float]:
        s = str(data)
        return [math.sin(hash(s[i:i+4]) % 100) if i < len(s) else 0.0 for i in range(5)]
    
    def _poly(self, coeffs: List[float]) -> str:
        return f"{coeffs[0]:.3f} + {coeffs[1]:.3f}x + {coeffs[2]:.3f}x²"
    
    def _derivs(self, coeffs: List[float]) -> List[float]:
        return [coeffs[1], 2*coeffs[2], 0.0, 0.0, 0.0]
    
    def _crit(self, coeffs: List[float]) -> List[float]:
        if abs(coeffs[2]) > 1e-6:
            return [-coeffs[1]/(2*coeffs[2])]
        return []

class ChoppyProcessor:
    def chunk(self, data: Any, chunk_size: int = 64, overlap: int = 16) -> Dict[str, Any]:
        s = str(data)
        step = max(1, chunk_size - overlap)
        std = [s[i:i + chunk_size] for i in range(0, len(s), step)]
        words = s.split(); wsize = max(1, chunk_size // 5)
        wchunks = [" ".join(words[i:i + wsize]) for i in range(0, len(words), wsize)]
        return {
            "standard": std,
            "semantic": wchunks,
            "fibonacci": self._fib(s),
            "statistics": {"total_length": len(s), "chunk_count": len(std), "average_chunk_size": len(s) / max(1, len(std))},
        }
    
    def _fib(self, s: str) -> List[str]:
        fib = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        chunks = []
        pos = 0
        for f in fib:
            if pos >= len(s):
                break
            chunks.append(s[pos:pos+f])
            pos += f
        return chunks

class EndpointCaster:
    def generate(self, data: Any) -> Dict[str, Any]:
        sig = hashlib.sha256(json.dumps(data, default=str, sort_keys=True).encode()).hexdigest()[:12]
        base = uuid.uuid4().hex[:6]
        return {
            "primary_endpoint": f"/api/v1/cast/{base}",
            "versioned_endpoints": [
                f"/api/v1/cast/{base}/reflect",
                f"/api/v1/cast/{base}/transform",
                f"/api/v1/cast/{base}/metadata",
                f"/api/v2/mirror/{sig}",
            ],
            "artifact_id": f"art-{uuid.uuid4().hex[:8]}",
            "metadata": {"content_type": self._ctype(data), "estimated_size": len(str(data)), "complexity": self._cpx(data)},
        }
    
    def _ctype(self, data: Any) -> str:
        s = str(data)
        if len(s) < 100:
            return "text/plain"
        if any(c in s for c in ['{', '[', '(']):
            return "application/json"
        return "text/plain"
    
    def _cpx(self, data: Any) -> float:
        s = str(data)
        return min(1.0, len(set(s)) / max(1, len(s)))

class CarryOnManager:
    def __init__(self, max_history: int = 200):
        self.memory: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.max_history = max_history
        self.access: Dict[str, int] = {}
    
    def store(self, key: str, value: Any) -> None:
        self.memory[key] = value
        self.access[key] = time.time()
        self.history.append({"key": key, "value": str(value)[:100], "time": time.time()})
        if len(self.history) > self.max_history:
            self.history.pop(0)

class SemanticMapper:
    def __init__(self):
        self.semantic_networks = {
            "reflection": ["mirror", "echo", "reverberation", "contemplation", "introspection"],
            "transformation": ["metamorphosis", "mutation", "evolution", "adaptation", "transmutation"],
            "analysis": ["examination", "scrutiny", "dissection", "investigation", "exploration"],
            "synthesis": ["combination", "fusion", "amalgamation", "integration", "unification"],
        }
    
    def map(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        scores = {}
        for category, words in self.semantic_networks.items():
            score = sum(1 for word in words if word in text_lower)
            scores[category] = score / len(words)
        return scores

class LoveReflector:
    def infuse(self, data: Any) -> Dict[str, Any]:
        text = str(data)
        return {"poetic": self._poem(text), "emotional_resonance": self._emo(text), "love_quotient": self._lq(text), "harmony_index": self._hi(text)}
    
    def _poem(self, text: str) -> str:
        words = text.split()
        if len(words) < 3:
            return text
        return f"{words[0]} {words[1]} {words[-1]}"
    
    def _emo(self, text: str) -> float:
        emotional_words = ['love', 'hate', 'joy', 'sad', 'happy', 'angry', 'peace', 'war']
        return sum(1 for word in emotional_words if word in text.lower()) / len(emotional_words)
    
    def _lq(self, text: str) -> float:
        return min(1.0, text.count('love') / max(1, len(text.split())))
    
    def _hi(self, text: str) -> float:
        return 0.7  # Placeholder

class FractalResonator:
    def __init__(self, max_depth: int = 8):
        self.max_depth = max_depth
    
    def cascade(self, data: Any) -> Dict[str, Any]:
        s = str(data)
        layers = []
        for depth in range(1, min(self.max_depth + 1, len(s) // 10 + 1)):
            chunk = s[:depth * 10]
            entropy = EntropyAnalyzer().measure(chunk)
            layers.append({"depth": depth, "entropy": entropy, "content": chunk})
        return {"layers": layers, "max_depth_reached": len(layers)}

class FeatureExtractor:
    def __init__(self, dim: int = 64, ngram: int = 3):
        self.dim = dim
        self.ngram = ngram
    
    def extract(self, text: str) -> List[float]:
        s = text.lower()
        features = [0.0] * self.dim
        for i in range(len(s) - self.ngram + 1):
            ngram = s[i:i+self.ngram]
            idx = hash(ngram) % self.dim
            features[idx] += 1.0
        # Normalize
        total = sum(features)
        if total > 0:
            features = [f / total for f in features]
        return features

class NeuroSymbolicFusion:
    def __init__(self):
        self.w_neuro = 0.55
        self.w_symbol = 0.45
    
    def fuse(self, neuro_features: List[float], symbolic_metrics: Dict[str, float]) -> Dict[str, Any]:
        neuro_score = sum(neuro_features) / len(neuro_features) if neuro_features else 0.0
        symbol_score = sum(symbolic_metrics.values()) / len(symbolic_metrics) if symbolic_metrics else 0.0
        fused = self.w_neuro * neuro_score + self.w_symbol * symbol_score
        return {
            "neuro_score": neuro_score,
            "symbol_score": symbol_score,
            "fused_score": fused,
            "decision": "transmit" if fused > 0.5 else "hold"
        }

class DecisionLogger:
    def __init__(self):
        self.events: List[Dict[str, Any]] = []
    
    def log(self, event: Dict[str, Any]) -> None:
        self.events.append({**event, "timestamp": time.time()})

class ReflectiveDB:
    def __init__(self, path: str = "reflective_db.json"):
        self.path = path
        self._data: List[Dict[str, Any]] = []
        self._load()
    
    def _load(self) -> None:
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    self._data = json.load(f)
            except:
                self._data = []
    
    def save(self) -> None:
        with open(self.path, 'w') as f:
            json.dump(self._data, f, indent=2)
    
    def add_record(self, record: Dict[str, Any]) -> None:
        self._data.append(record)
        self.save()

class RLAgent:
    def __init__(self, actions: List[str] = None, eps: float = 0.1):
        self.actions = actions or ["bpsk", "qpsk", "ofdm"]
        self.eps = eps
        self.q: Dict[Tuple[int, int, int], Dict[str, Dict[str, float]]] = {}
    
    def choose_action(self, state: Tuple[int, int, int]) -> str:
        if np.random.random() < self.eps or state not in self.q:
            return np.random.choice(self.actions)
        action_values = {a: self.q[state][a]["q"] for a in self.actions if a in self.q[state]}
        return max(action_values.items(), key=lambda x: x[1])[0] if action_values else np.random.choice(self.actions)
    
    def update(self, state: Tuple[int, int, int], action: str, reward: float) -> None:
        if state not in self.q:
            self.q[state] = {a: {"q": 0.0, "n": 0} for a in self.actions}
        self.q[state][action]["n"] += 1
        n = self.q[state][action]["n"]
        old_q = self.q[state][action]["q"]
        self.q[state][action]["q"] = old_q + (reward - old_q) / n

class MirrorCastEngine:
    def __init__(self):
        self.entropy = EntropyAnalyzer()
        self.reflector = DianneReflector()
        self.matrix = MatrixTransformer()
        self.symbols = JuliaSymbolEngine()
        self.choppy = ChoppyProcessor()
        self.endpoints = EndpointCaster()
        self.memory = CarryOnManager()
        self.semantic = SemanticMapper()
        self.love = LoveReflector()
        self.fractal = FractalResonator()
    
    def cast(self, data: Any) -> Dict[str, Any]:
        return {
            "entropy": self.entropy.measure(data),
            "reflection": self.reflector.reflect(data),
            "matrix": self.matrix.project(data),
            "symbolic": self.symbols.analyze(data),
            "chunks": self.choppy.chunk(data),
            "endpoints": self.endpoints.generate(data),
            "semantic": self.semantic.map(str(data)),
            "love": self.love.infuse(data),
            "fractal": self.fractal.cascade(data),
            "timestamp": time.time()
        }

class AdaptiveLinkPlanner:
    def __init__(self, db_path: str = "reflective_db.json"):
        self.extractor = FeatureExtractor()
        self.fusion = NeuroSymbolicFusion()
        self.agent = RLAgent(actions=["bpsk", "qpsk", "ofdm"], eps=0.1)
        self.db = ReflectiveDB(db_path)
        self.log = DecisionLogger()
    
    def plan(self, text: str, current_config: ModConfig, analysis: Dict[str, Any], snr_db: float) -> Tuple[ModConfig, str]:
        # Extract features
        features = self.extractor.extract(text)
        
        # Create symbolic metrics from analysis
        symbolic_metrics = {
            "entropy": analysis.get("entropy", 0.0),
            "complexity": analysis.get("endpoints", {}).get("metadata", {}).get("complexity", 0.5),
            "semantic_density": sum(analysis.get("semantic", {}).values()) / max(1, len(analysis.get("semantic", {})))
        }
        
        # Fuse neuro-symbolic
        fusion_result = self.fusion.fuse(features, symbolic_metrics)
        
        # Create state representation
        entropy_bin = min(9, int(analysis.get("entropy", 0.0) * 10))
        complexity_bin = min(9, int(symbolic_metrics["complexity"] * 10))
        snr_bin = min(9, int(snr_db / 10))
        state = (entropy_bin, complexity_bin, snr_bin)
        
        # Choose action
        action = self.agent.choose_action(state)
        
        # Update config based on action
        new_config = ModConfig(
            sample_rate=current_config.sample_rate,
            symbol_rate=current_config.symbol_rate,
            amplitude=current_config.amplitude,
            f0=current_config.f0,
            f1=current_config.f1,
            fc=current_config.fc,
            clip=current_config.clip,
            ofdm_subc=current_config.ofdm_subc,
            cp_len=current_config.cp_len,
            dsss_chip_rate=current_config.dsss_chip_rate
        )
        
        # Adjust based on action
        if action == "bpsk":
            new_config.symbol_rate = 1200
        elif action == "qpsk":
            new_config.symbol_rate = 2400
        elif action == "ofdm":
            new_config.symbol_rate = 4800
        
        explanation = f"Neuro-symbolic score: {fusion_result['fused_score']:.3f}, chose {action.upper()} for state {state}"
        
        self.log.log({
            "text_hash": hashlib.sha256(text.encode()).hexdigest()[:8],
            "state": state,
            "action": action,
            "fusion_result": fusion_result,
            "explanation": explanation
        })
        
        return new_config, explanation
    
    def reward_and_record(self, text: str, config: ModConfig, explanation: str, success: bool, snr_db: float, harmony: float, tx_len: int, rx_len: int) -> None:
        # Simple reward: +1 for success, -1 for failure
        reward = 1.0 if success else -1.0
        # Adjust based on harmony (higher is better)
        reward *= harmony
        
        # Get current state
        analysis = MirrorCastEngine().cast(text)
        entropy_bin = min(9, int(analysis.get("entropy", 0.0) * 10))
        complexity_bin = min(9, int(analysis.get("endpoints", {}).get("metadata", {}).get("complexity", 0.5) * 10))
        snr_bin = min(9, int(snr_db / 10))
        state = (entropy_bin, complexity_bin, snr_bin)
        
        # Determine action from config
        if config.symbol_rate <= 1200:
            action = "bpsk"
        elif config.symbol_rate <= 2400:
            action = "qpsk"
        else:
            action = "ofdm"
        
        # Update Q-values
        self.agent.update(state, action, reward)
        
        # Record to DB
        self.db.add_record({
            "timestamp": time.time(),
            "text_hash": hashlib.sha256(text.encode()).hexdigest()[:8],
            "state": state,
            "action": action,
            "reward": reward,
            "success": success,
            "snr_db": snr_db,
            "harmony": harmony,
            "tx_len": tx_len,
            "rx_len": rx_len,
            "explanation": explanation
        })

# =========================================================
# Utilities
# =========================================================

def now_ms() -> int:
    return int(time.time() * 1000)

def crc32_bytes(data: bytes) -> bytes:
    return binascii.crc32(data).to_bytes(4, "big")

def crc16_ccitt(data: bytes) -> bytes:
    poly, crc = 0x1021, 0xFFFF
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFFFF if (crc & 0x8000) else ((crc << 1) & 0xFFFF)
    return crc.to_bytes(2, "big")

def to_bits(data: bytes) -> List[int]:
    return [(byte >> i) & 1 for byte in data for i in range(7, -1, -1)]

def from_bits(bits: Sequence[int]) -> bytes:
    if len(bits) % 8 != 0:
        bits = list(bits) + [0] * (8 - len(bits) % 8)
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for b in bits[i:i+8]:
            byte = (byte << 1) | (1 if b else 0)
        out.append(byte)
    return bytes(out)

def chunk_bits(bits: Sequence[int], n: int) -> List[List[int]]:
    return [list(bits[i:i+n]) for i in range(0, len(bits), n)]

def safe_json(obj: Any) -> str:
    def enc(x):
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
        if isinstance(x, complex):
            return {"real": float(x.real), "imag": float(x.imag)}
        if isinstance(x, datetime):
            return x.isoformat()
        return str(x)
    return json.dumps(obj, ensure_ascii=False, indent=2, default=enc)

def write_wav_mono(path: Path, signal: np.ndarray, sample_rate: int):
    import wave
    sig = np.clip(signal, -1.0, 1.0)
    pcm = (sig * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())

def write_iq_f32(path: Path, iq: np.ndarray):
    if iq.ndim != 1 or not np.iscomplexobj(iq):
        raise ValueError("iq must be 1-D complex array")
    interleaved = np.empty(iq.size * 2, dtype=np.float32)
    interleaved[0::2] = iq.real.astype(np.float32)
    interleaved[1::2] = iq.imag.astype(np.float32)
    path.write_bytes(interleaved.tobytes())

def plot_wave_and_spectrum(path_png: Path, x: np.ndarray, sr: int, title: str):
    if not HAS_MPL: 
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,5))
    t = np.arange(len(x))/sr
    ax1.plot(t[:min(len(t), 0.05*sr)], x[:min(len(x), int(0.05*sr))])
    ax1.set_title(f"{title} (first 50ms)")
    ax1.set_xlabel("s"); ax1.set_ylabel("amplitude")
    spec = np.abs(rfft(x)) + 1e-12
    freqs = rfftfreq(len(x), 1.0/sr)
    ax2.semilogy(freqs, spec/spec.max())
    ax2.set_xlim(0, min(8000, sr//2)); ax2.set_xlabel("Hz"); ax2.set_ylabel("norm |X(f)|")
    plt.tight_layout(); fig.savefig(path_png); plt.close(fig)

def play_audio(x: np.ndarray, sr: int):
    if not HAS_AUDIO:
        log.warning("sounddevice not installed; cannot play audio")
        return
    sd.play(x, sr); sd.wait()

# =========================================================
# FEC (simple Hamming 7,4; heavy codes are stubs)
# =========================================================

def hamming74_encode(data_bits: List[int]) -> List[int]:
    if len(data_bits) % 4 != 0:
        data_bits = data_bits + [0] * (4 - len(data_bits) % 4)
    out = []
    for i in range(0, len(data_bits), 4):
        d0, d1, d2, d3 = data_bits[i:i+4]
        p1 = d0 ^ d1 ^ d3
        p2 = d0 ^ d2 ^ d3
        p3 = d1 ^ d2 ^ d3
        out += [p1, p2, d0, p3, d1, d2, d3]
    return out

def fec_encode(bits: List[int], scheme: FEC) -> List[int]:
    if scheme == FEC.NONE:
        return list(bits)
    if scheme == FEC.HAMMING74:
        return hamming74_encode(bits)
    if scheme in (FEC.REED_SOLOMON, FEC.LDPC, FEC.TURBO):
        raise NotImplementedError(f"{scheme.name} encoding not implemented in this minimal build")
    raise ValueError("Unknown FEC")

# =========================================================
# Framing / Security / Watermark
# =========================================================

@dataclass
class SecurityConfig:
    password: Optional[str] = None           # AES-GCM if provided
    watermark: Optional[str] = None          # prepended SHA256[0:8]
    hmac_key: Optional[str] = None           # HMAC-SHA256 appended

def aes_gcm_encrypt(plaintext: bytes, password: str) -> bytes:
    if not HAS_CRYPTO:
        raise RuntimeError("pycryptodome required for encryption")
    salt = get_random_bytes(16)
    key = PBKDF2(password, salt, dkLen=32, count=200_000)
    nonce = get_random_bytes(12)
    cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
    ct, tag = cipher.encrypt_and_digest(plaintext)
    return b"AGCM" + salt + nonce + tag + ct

def apply_hmac(data: bytes, hkey: str) -> bytes:
    import hmac
    key = hashlib.sha256(hkey.encode("utf-8")).digest()
    mac = hmac.new(key, data, hashlib.sha256).digest()
    return data + b"HMAC" + mac

def add_watermark(data: bytes, wm: str) -> bytes:
    return hashlib.sha256(wm.encode("utf-8")).digest()[:8] + data

def frame_payload(payload: bytes, fcfg: FrameConfig) -> bytes:
    header = struct.pack(">BBI", 0xA5, fcfg.version, now_ms() & 0xFFFFFFFF)
    core = header + payload
    tail = b""
    if fcfg.use_crc32:
        tail += crc32_bytes(core)
    if fcfg.use_crc16:
        tail += crc16_ccitt(core)
    return fcfg.preamble + core + tail

def encode_text(
    text: str,
    fcfg: FrameConfig,
    sec: SecurityConfig,
    fec_scheme: FEC,
) -> List[int]:
    data = text.encode("utf-8")
    if sec.watermark:
        data = add_watermark(data, sec.watermark)
    if sec.password:
        data = aes_gcm_encrypt(data, sec.password)
    framed = frame_payload(data, fcfg)
    if sec.hmac_key:
        framed = apply_hmac(framed, sec.hmac_key)
    bits = to_bits(framed)
    bits = fec_encode(bits, fec_scheme)
    return bits

# =========================================================
# Modulators (audio & IQ)
# =========================================================

class Modulators:
    @staticmethod
    def bfsK(bits: Sequence[int], cfg: ModConfig) -> np.ndarray:
        sr, rb = cfg.sample_rate, cfg.symbol_rate
        spb = int(sr / rb)
        t = np.arange(spb) / sr
        s = []
        a = cfg.amplitude
        for b in bits:
            f = cfg.f1 if b else cfg.f0
            s.append(a * np.sin(2*np.pi*f*t))
        y = np.concatenate(s) if s else np.zeros(0, dtype=np.float64)
        return np.clip(y, -1, 1).astype(np.float32) if cfg.clip else y.astype(np.float32)

    @staticmethod
    def bpsK(bits: Sequence[int], cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        sr, rb, fc = cfg.sample_rate, cfg.symbol_rate, cfg.fc
        spb = int(sr / rb)
        t = np.arange(spb) / sr
        a = cfg.amplitude
        audio_blocks, iq_blocks = [], []
        for b in bits:
            phase = 0.0 if b else np.pi
            audio_blocks.append(a * np.sin(2*np.pi*fc*t + phase))
            iq_blocks.append(a * (np.cos(phase) + 1j*np.sin(phase)) * np.ones_like(t, dtype=np.complex64))
        audio = np.concatenate(audio_blocks) if audio_blocks else np.zeros(0, dtype=np.float64)
        iq = np.concatenate(iq_blocks) if iq_blocks else np.zeros(0, dtype=np.complex64)
        if cfg.clip: audio = np.clip(audio, -1, 1)
        return audio.astype(np.float32), iq

    @staticmethod
    def qpsK(bits: Sequence[int], cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        # Gray map: 00->(1+1j), 01->(-1+1j), 11->(-1-1j), 10->(1-1j)
        pairs = chunk_bits(bits, 2)
        syms = []
        for p in pairs:
            b0, b1 = (p + [0,0])[:2]
            if (b0, b1) == (0,0): s = 1 + 1j
            elif (b0, b1) == (0,1): s = -1 + 1j
            elif (b0, b1) == (1,1): s = -1 - 1j
            else: s = 1 - 1j
            syms.append(s / math.sqrt(2))  # unit energy
        return Modulators._psk_qam_to_audio_iq(np.array(syms, dtype=np.complex64), cfg)

    @staticmethod
    def qam16(bits: Sequence[int], cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        quads = chunk_bits(bits, 4)
        def map2(b0,b1):
            # Gray 2-bit to {-3,-1,1,3}
            val = (b0<<1) | b1
            return [-3,-1,1,3][val]
        syms = []
        for q in quads:
            b0,b1,b2,b3 = (q+[0,0,0,0])[:4]
            I = map2(b0,b1); Q = map2(b2,b3)
            syms.append((I + 1j*Q)/math.sqrt(10)) # unit average power
        return Modulators._psk_qam_to_audio_iq(np.array(syms, dtype=np.complex64), cfg)

    @staticmethod
    def _psk_qam_to_audio_iq(syms: np.ndarray, cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        sr, rb, fc = cfg.sample_rate, cfg.symbol_rate, cfg.fc
        spb = int(sr / rb)
        a = cfg.amplitude
        # Upsample each symbol to 'spb' samples (rectangular pulse)
        i = np.repeat(syms.real.astype(np.float32), spb)
        q = np.repeat(syms.imag.astype(np.float32), spb)
        t = np.arange(len(i)) / sr
        audio = a * (i*np.cos(2*np.pi*fc*t) - q*np.sin(2*np.pi*fc*t))
        iq = (a * i) + 1j*(a * q)
        if cfg.clip: audio = np.clip(audio, -1, 1)
        return audio.ast(np.float32), iq.astype(np.complex64)

    @staticmethod
    def afsK(bits: Sequence[int], cfg: ModConfig) -> np.ndarray:
        return Modulators.bfsK(bits, cfg)

    @staticmethod
    def dsss_bpsK(bits: Sequence[int], cfg: ModConfig) -> np.ndarray:
        # Very simple DSSS: chip with PN sequence at cfg.dsss_chip_rate
        pn = np.array([1, -1, 1, 1, -1, 1, -1, -1], dtype=np.float32)  # toy PN8
        sr = cfg.sample_rate
        chips_per_symbol = max(1, int(cfg.dsss_chip_rate / cfg.symbol_rate))
        spb = int(sr / (cfg.dsss_chip_rate))
        base = []
        for b in bits:
            bit_val = 1.0 if b else -1.0
            ch = bit_val * pn
            ch = np.repeat(ch, spb)
            base.append(ch)
        baseband = np.concatenate(base) if base else np.zeros(0, dtype=np.float32)
        # Upconvert to audio carrier
        t = np.arange(len(baseband))/sr
        audio = cfg.amplitude * baseband * np.sin(2*np.pi*cfg.fc*t)
        if cfg.clip: audio = np.clip(audio, -1, 1)
        return audio.astype(np.float32)

    @staticmethod
    def ofdm(bits: Sequence[int], cfg: ModConfig) -> Tuple[np.ndarray, np.ndarray]:
        # Toy OFDM: QPSK mapping across N subcarriers, IFFT, add cyclic prefix
        N = cfg.ofdm_subc
        spb_sym = int(cfg.sample_rate / cfg.symbol_rate)  # samples per OFDM symbol (approx shaping)
        chunks = chunk_bits(bits, 2*N)
        a = cfg.amplitude
        wave = []
        iq = []
        for ch in chunks:
            # map 2 bits -> QPSK symbol
            qsyms = []
            pairs = chunk_bits(ch, 2)
            for p in pairs:
                b0,b1 = (p+[0,0])[:2]
                if (b0,b1)==(0,0): s = 1+1j
                elif (b0,b1)==(0,1): s = -1+1j
                elif (b0,b1)==(1,1): s = -1-1j
                else: s = 1-1j
                qsyms.append(s/math.sqrt(2))
            # pad to N
            if len(qsyms) < N:
                qsyms += [0j]*(N-len(qsyms))
            Xk = np.array(qsyms, dtype=np.complex64)
            xt = np.fft.ifft(Xk)  # time domain symbol (complex)
            # cyclic prefix
            cp = xt[-cfg.cp_len:]
            sym = np.concatenate([cp, xt])
            # stretch to samples-per-symbol for audio mixing
            reps = max(1, int(spb_sym/len(sym)))
            sym_up = np.repeat(sym, reps)
            # audio upconvert
            t = np.arange(len(sym_up))/cfg.sample_rate
            audio = a*(sym_up.real*np.cos(2*np.pi*cfg.fc*t) - sym_up.imag*np.sin(2*np.pi*cfg.fc*t))
            wave.append(audio.astype(np.float32))
            iq.append((a*sym_up).astype(np.complex64))
        audio = np.concatenate(wave) if wave else np.zeros(0, dtype=np.float32)
        iqc = np.concatenate(iq) if iq else np.zeros(0, dtype=np.complex64)
        if cfg.clip: audio = np.clip(audio, -1, 1)
        return audio, iqc

# =========================================================
# LLM backends (Local final inference; Remote resource-only)
# =========================================================

class BaseLLM:
    def generate(self, prompt: str, **kwargs) -> str: raise NotImplementedError

class LocalLLM(BaseLLM):
    def __init__(self, configs: List[HTTPConfig]):
        if requests is None:
            raise RuntimeError("LocalLLM requires 'requests' (pip install requests)")
        self.configs = configs
        self.idx = 0

    def generate(self, prompt: str, **kwargs) -> str:
        last = None
        for _ in range(len(self.configs)):
            cfg = self.configs[self.idx]
            try:
                out = self._call(cfg, prompt, **kwargs)
                return out
            except Exception as e:
                last = e
                self.idx = (self.idx + 1) % len(self.configs)
        raise last or RuntimeError("All local LLM configs failed")

    def _post(self, cfg: HTTPConfig, url: str, headers: dict, body: dict) -> dict:
        s = requests.Session()
        for attempt in range(cfg.max_retries):
            try:
                r = s.post(url, headers=headers, json=body, timeout=cfg.timeout, verify=cfg.verify_ssl)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                if attempt < cfg.max_retries-1:
                    time.sleep(cfg.retry_delay*(2**attempt))
                else:
                    raise

    def _call(self, cfg: HTTPConfig, prompt: str, **kwargs) -> str:
        mode = cfg.mode
        if mode == "openai-chat":
            url = f"{cfg.base_url.rstrip('/')}/v1/chat/completions"
            headers = {"Content-Type": "application/json"}
            if cfg.api_key: headers["Authorization"] = f"Bearer {cfg.api_key}"
            body = {
                "model": cfg.model or "gpt-4o-mini",
                "messages": [{"role":"user","content":prompt}],
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 512),
            }
            data = self._post(cfg, url, headers, body)
            return data["choices"][0]["message"]["content"]
        if mode == "openai-completions":
            url = f"{cfg.base_url.rstrip('/')}/v1/completions"
            headers = {"Content-Type": "application/json"}
            if cfg.api_key: headers["Authorization"] = f"Bearer {cfg.api_key}"
            body = {
                "model": cfg.model or "gpt-3.5-turbo-instruct",
                "prompt": prompt,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 512),
            }
            data = self._post(cfg, url, headers, body)
            return data["choices"][0]["text"]
        if mode == "llama-cpp":
            url = f"{cfg.base_url.rstrip('/')}/completion"
            body = {"prompt": prompt, "temperature": kwargs.get("temperature",0.7), "n_predict": kwargs.get("max_tokens",512)}
            data = self._post(cfg, url, {}, body)
            if "content" in data: return data["content"]
            if "choices" in data and data["choices"]: return data["choices"][0].get("text","")
            return data.get("text","")
        if mode == "textgen-webui":
            url = f"{cfg.base_url.rstrip('/')}/api/v1/generate"
            body = {"prompt": prompt, "max_new_tokens": kwargs.get("max_tokens",512), "temperature": kwargs.get("temperature",0.7)}
            data = self._post(cfg, url, {}, body)
            return data.get("results",[{}])[0].get("text","")
        raise ValueError(f"Unsupported mode: {mode}")

class ResourceLLM(BaseLLM):
    def __init__(self, cfg: Optional[HTTPConfig] = None):
        self.cfg = cfg

    def generate(self, prompt: str, **kwargs) -> str:
        # Constrained to resources-only summarization
        if self.cfg is None or requests is None:
            return LocalSummarizer().summarize(prompt)
        url = f"{self.cfg.base_url.rstrip('/')}/v1/chat/completions"
        headers = {"Content-Type":"application/json"}
        if self.cfg.api_key: headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        system = ("You are a constrained assistant. ONLY summarize/structure the provided INPUT RESOURCES. "
                  "Do not add external knowledge.")
        body = {
            "model": self.cfg.model or "gpt-4o-mini",
            "messages":[{"role":"system","content":system},{"role":"user","content":prompt}],
            "temperature": kwargs.get("temperature",0.2),
            "max_tokens": kwargs.get("max_tokens",512),
        }
        s = requests.Session()
        r = s.post(url, headers=headers, json=body, timeout=self.cfg.timeout, verify=self.cfg.verify_ssl)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

class LocalSummarizer:
    def __init__(self):
        self.stop = {
            "the","a","an","and","or","but","in","on","at","to","for","of","with","by","is","are",
            "was","were","be","been","being","have","has","had","do","does","did","will","would",
            "could","should","from","that","this","it","as"
        }
    def summarize(self, text: str) -> str:
        txt = " ".join(text.split())
        if not txt: return "No content to summarize."
        sents = [s.strip() for s in txt.replace("?",".").replace("!",".").split(".") if s.strip()]
        if not sents: return txt[:300] + ("..." if len(txt)>300 else "")
        # score sentences by length + term frequency (very light heuristic)
        words = [w.lower().strip(",;:()[]") for w in txt.split()]
        freq: Dict[str,int] = {}
        for w in words:
            if w and w not in self.stop: freq[w] = freq.get(w,0)+1
        scored = []
        for s in sents:
            sw = [w.lower().strip(",;:()[]") for w in s.split()]
            score = len(s) * 0.1 + sum(freq.get(w,0) for w in sw)
            scored.append((s, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        keep = [s for s,_ in scored[: min(6,len(scored))]]
        keep.sort(key=lambda k: sents.index(k))
        out = " ".join(keep)
        return out[:800] + ("..." if len(out)>800 else "")

# =========================================================
# Orchestrator
# =========================================================

class DualLLMOrchestrator:
    def __init__(self, local: LocalLLM, resource: ResourceLLM, settings: OrchestratorSettings):
        self.local, self.resource, self.set = local, resource, settings

    def _load_resources(self, paths: List[str], inline: List[str]) -> str:
        parts = []
        for p in paths:
            pa = Path(p)
            if pa.exists() and pa.is_file():
                try:
                    parts.append(pa.read_text(encoding="utf-8", errors="ignore"))
                except Exception:
                    parts.append(f"[[UNREADABLE_FILE:{pa.name}]]")
            else:
                parts.append(f"[[MISSING_FILE:{pa}]]")
        parts += [str(x) for x in inline]
        blob = "\n\n".join(parts)
        return blob[: self.set.max_context_chars]

    def compose(self, user_prompt: str, resource_paths: List[str], inline_resources: List[str]) -> Tuple[str,str]:
        res_text = self._load_resources(resource_paths, inline_resources)
        res_summary = self.resource.generate(
            f"INPUT RESOURCES:\n{res_text}\n\nTASK: Summarize/structure ONLY the content above.",
            temperature=0.2, max_tokens=self.set.max_tokens
        )
        final_prompt = (
            "You are a LOCAL expert system. Use ONLY the structured summary below; do not invent facts.\n\n"
            f"=== STRUCTURED SUMMARY ===\n{res_summary}\n\n"
            f"=== USER PROMPT ===\n{user_prompt}\n\n"
            f"STYLE: {self.set.style}. Be clear and directly actionable."
        )
        return final_prompt, res_summary

    def run(self, user_prompt: str, resource_paths: List[str], inline_resources: List[str]) -> Dict[str,str]:
        fp, summary = self.compose(user_prompt, resource_paths, inline_resources)
        ans = self.local.generate(fp, temperature=self.set.temperature, max_tokens=self.set.max_tokens)
        return {"summary": summary, "final": ans, "prompt": fp}

# =========================================================
# End-to-end casting
# =========================================================

@dataclass
class OutputPaths:
    wav: Optional[Path] = None
    iq: Optional[Path] = None
    meta: Optional[Path] = None
    png: Optional[Path] = None

def bits_to_signals(bits: List[int], scheme: ModulationScheme, mcfg: ModConfig) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if scheme == ModulationScheme.BFSK:
        return Modulators.bfsK(bits, mcfg), None
    if scheme == ModulationScheme.AFSK:
        return Modulators.afsK(bits, mcfg), None
    if scheme == ModulationScheme.BPSK:
        return Modulators.bpsK(bits, mcfg)
    if scheme == ModulationScheme.QPSK:
        return Modulators.qpsK(bits, mcfg)
    if scheme == ModulationScheme.QAM16:
        return Modulators.qam16(bits, mcfg)
    if scheme == ModulationScheme.OFDM:
        return Modulators.ofdm(bits, mcfg)
    if scheme == ModulationScheme.DSSS_BPSK:
        return Modulators.dsss_bpsK(bits, mcfg), None
    raise ValueError("Unknown modulation scheme")

def cast_to_files(
    text: str,
    outdir: Path,
    scheme: ModulationScheme,
    mcfg: ModConfig,
    want_wav: bool,
    want_iq: bool,
    title: str = "WaveCaster"
) -> OutputPaths:
    outdir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    base = outdir / f"cast_{scheme.name.lower()}_{ts}"
    # Minimal frame (no FEC/security here; caller handles)
    fcfg = FrameConfig()
    bits = to_bits(frame_payload(text.encode("utf-8"), fcfg))
    audio, iq = bits_to_signals(bits, scheme, mcfg)
    paths = OutputPaths()
    if want_wav and audio is not None and len(audio)>0:
        paths.wav = base.with_suffix(".wav"); write_wav_mono(paths.wav, audio, mcfg.sample_rate)
    if want_iq:
        if iq is None and audio is not None:
            # make a naive hilbert to IQ for convenience
            try:
                q = np.imag(sp_signal.hilbert(audio))
                iq = audio.astype(np.float32) + 1j*q.astype(np.float32)
            except Exception:
                iq = (audio.astype(np.float32) + 1j*np.zeros_like(audio, dtype=np.float32))
        if iq is not None:
            paths.iq = base.with_suffix(".iqf32"); write_iq_f32(paths.iq, iq)
    # Visualization
    if audio is not None and len(audio)>0 and HAS_MPL:
        paths.png = base.with_suffix(".png"); plot_wave_and_spectrum(paths.png, audio, mcfg.sample_rate, title)
    # Meta
    meta = {
        "timestamp": ts, "scheme": scheme.name, "sample_rate": mcfg.sample_rate,
        "symbol_rate": mcfg.symbol_rate, "framesec": len(audio)/mcfg.sample_rate if audio is not None else 0,
    }
    paths.meta = base.with_suffix(".json")
    paths.meta.write_text(safe_json(meta), encoding="utf-8")
    return paths

def full_cast_and_save(
    text: str,
    outdir: Path,
    scheme: ModulationScheme,
    mcfg: ModConfig,
    fcfg: FrameConfig,
    sec: SecurityConfig,
    fec_scheme: FEC,
    want_wav: bool,
    want_iq: bool,
    title: str = "WaveCaster"
) -> OutputPaths:
    outdir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    base = outdir / f"cast_{scheme.name.lower()}_{ts}"
    bits = encode_text(text, fcfg, sec, fec_scheme)
    audio, iq = bits_to_signals(bits, scheme, mcfg)
    paths = OutputPaths()
    if want_wav and audio is not None and len(audio)>0:
        paths.wav = base.with_suffix(".wav"); write_wav_mono(paths.wav, audio, mcfg.sample_rate)
    if want_iq:
        if iq is None and audio is not None:
            try:
                q = np.imag(sp_signal.hilbert(audio))
                iq = audio.astype(np.float32) + 1j*q.astype(np.float32)
            except Exception:
                iq = (audio.astype(np.float32) + 1j*np.zeros_like(audio, dtype=np.float32))
        if iq is not None:
            paths.iq = base.with_suffix(".iqf32"); write_iq_f32(paths.iq, iq)
    if audio is not None and len(audio)>0 and HAS_MPL:
        paths.png = base.with_suffix(".png"); plot_wave_and_spectrum(paths.png, audio, mcfg.sample_rate, title)
    meta = {
        "timestamp": ts, "scheme": scheme.name, "sample_rate": mcfg.sample_rate,
        "symbol_rate": mcfg.symbol_rate, "framesec": len(audio)/mcfg.sample_rate if audio is not None else 0,
        "fec": fec_scheme.name, "encrypted": bool(sec.password), "watermark": bool(sec.watermark),
        "hmac": bool(sec.hmac_key),
    }
    paths.meta = base.with_suffix(".json"); paths.meta.write_text(safe_json(meta), encoding="utf-8")
    return paths

# =========================================================
# CLI
# =========================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="dual_llm_wavecaster_enhanced", description="Two-LLM orchestration → modulated waveform")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_mod_args(sp):
        sp.add_argument("--scheme", choices=[s.name.lower() for s in ModulationScheme], default="bfsk")
        sp.add_argument("--sample-rate", type=int, default=48000)
        sp.add_argument("--symbol-rate", type=int, default=1200)
        sp.add_argument("--amplitude", type=float, default=0.7)
        sp.add_argument("--f0", type=float, default=1200.0)
        sp.add_argument("--f1", type=float, default=2200.0)
        sp.add_argument("--fc", type=float, default=1800.0)
        sp.add_argument("--no-clip", action="store_true")
        sp.add_argument("--outdir", type=str, default="casts")
        sp.add_argument("--wav", action="store_true")
        sp.add_argument("--iq", action="store_true")
        sp.add_argument("--play", action="store_true", help="Play audio to soundcard (if available)")

        # OFDM / DSSS
        sp.add_argument("--ofdm-subc", type=int, default=64)
        sp.add_argument("--cp-len", type=int, default=16)
        sp.add_argument("--dsss-chip-rate", type=int, default=4800)

    # cast: 2-LLM orchestration then modulate
    sp_cast = sub.add_parser("cast", help="Compose via dual LLMs then modulate")
    sp_cast.add_argument("--prompt", type=str, required=True)
    sp_cast.add_argument("--resource-file", nargs="*", default=[])
    sp_cast.add_argument("--resource-text", nargs="*", default=[])
    # Local LLM
    sp_cast.add_argument("--local-url", type=str, default="http://127.0.0.1:8080")
    sp_cast.add_argument("--local-mode", choices=["openai-chat","openai-completions","llama-cpp","textgen-webui"], default="llama-cpp")
    sp_cast.add_argument("--local-model", type=str, default="local-gguf")
    sp_cast.add_argument("--local-key", type=str, default=None)
    # Remote Resource LLM
    sp_cast.add_argument("--remote-url", type=str, default=None)
    sp_cast.add_argument("--remote-model", type=str, default="gpt-4o-mini")
    sp_cast.add_argument("--remote-key", type=str, default=None)
    # Orchestration params
    sp_cast.add_argument("--style", type=str, default="concise")
    sp_cast.add_argument("--max-tokens", type=int, default=512)
    sp_cast.add_argument("--temperature", type=float, default=0.7)
    # Security / FEC
    sp_cast.add_argument("--password", type=str, default=None)
    sp_cast.add_argument("--watermark", type=str, default=None)
    sp_cast.add_argument("--hmac-key", type=str, default=None)
    sp_cast.add_argument("--fec", choices=[f.name.lower() for f in FEC], default="hamming74")
    # Neuro-symbolic adaptive
    sp_cast.add_argument("--adaptive", action="store_true", help="Use neuro-symbolic adaptive link planning")
    sp_cast.add_argument("--db-path", type=str, default="reflective_db.json", help="Path for reflective database")
    sp_cast.add_argument("--snr-db", type=float, default=30.0, help="Estimated SNR in dB for adaptive planning")
    add_mod_args(sp_cast)

    # modulate: direct text to waveform
    sp_mod = sub.add_parser("modulate", help="Modulate provided text directly")
    sp_mod.add_argument("--text", type=str, required=True)
    sp_mod.add_argument("--password", type=str, default=None)
    sp_mod.add_argument("--watermark", type=str, default=None)
    sp_mod.add_argument("--hmac-key", type=str, default=None)
    sp_mod.add_argument("--fec", choices=[f.name.lower() for f in FEC], default="none")
    add_mod_args(sp_mod)

    # visualize existing WAV
    sp_vis = sub.add_parser("visualize", help="Plot waveform + spectrum from WAV")
    sp_vis.add_argument("--wav", type=str, required=True)
    sp_vis.add_argument("--out", type=str, default=None)

    # analyze: print basic metrics
    sp_an = sub.add_parser("analyze", help="Basic audio metrics of WAV")
    sp_an.add_argument("--wav", type=str, required=True)

    # neuro-symbolic analysis
    sp_ns = sub.add_parser("neuro-symbolic", help="Neuro-symbolic analysis of text")
    sp_ns.add_argument("--text", type=str, required=True)
    sp_ns.add_argument("--plot", action="store_true", help="Generate plots")

    # adaptive learning
    sp_learn = sub.add_parser("learn", help="Run adaptive learning episodes")
    sp_learn.add_argument("--text", type=str, required=True)
    sp_learn.add_argument("--episodes", type=int, default=10)
    sp_learn.add_argument("--snr-db", type=float, default=30.0)
    sp_learn.add_argument("--db-path", type=str, default="reflective_db.json")

    return p

def parse_scheme(s: str) -> ModulationScheme:
    return ModulationScheme[s.upper()]

def parse_fec(s: str) -> FEC:
    return FEC[s.upper()]

def make_modcfg(args: argparse.Namespace) -> ModConfig:
    return ModConfig(
        sample_rate=args.sample_rate, symbol_rate=args.symbol_rate, amplitude=args.amplitude,
        f0=args.f0, f1=args.f1, fc=args.fc, clip=not args.no_clip,
        ofdm_subc=getattr(args, "ofdm_subc", 64), cp_len=getattr(args,"cp_len",16),
        dsss_chip_rate=getattr(args,"dsss_chip_rate",4800),
    )

def cmd_cast(args: argparse.Namespace) -> int:
    # Build LLMs
    local = LocalLLM([HTTPConfig(
        base_url=args.local_url, model=args.local_model, mode=args.local_mode, api_key=args.local_key
    )])
    rcfg = HTTPConfig(base_url=args.remote_url, model=args.remote_model, api_key=args.remote_key) if args.remote_url else None
    resource = ResourceLLM(rcfg)
    orch = DualLLMOrchestrator(local, resource, OrchestratorSettings(
        temperature=args.temperature, max_tokens=args.max_tokens, style=args.style
    ))
    result = orch.run(args.prompt, args.resource_file, args.resource_text)
    
    # Neuro-symbolic analysis
    engine = MirrorCastEngine()
    analysis = engine.cast(result["final"])
    
    # Build pipeline
    mcfg = make_modcfg(args)
    fcfg = FrameConfig()
    sec = SecurityConfig(password=args.password, watermark=args.watermark, hmac_key=args.hmac_key)
    scheme = parse_scheme(args.scheme)
    fec_s = parse_fec(args.fec)
    
    # Adaptive link planning
    if args.adaptive:
        planner = AdaptiveLinkPlanner(db_path=args.db_path)
        mcfg, explanation = planner.plan(result["final"], mcfg, analysis, args.snr_db)
        result["adaptive_explanation"] = explanation
        result["adaptive_config"] = {
            "symbol_rate": mcfg.symbol_rate,
            "scheme": scheme.name
        }
    
    paths = full_cast_and_save(
        text=result["final"], outdir=Path(args.outdir), scheme=scheme, mcfg=mcfg, fcfg=fcfg,
        sec=sec, fec_scheme=fec_s, want_wav=args.wav or (not args.iq), want_iq=args.iq,
        title=f"{scheme.name} | DualLLM Wave"
    )
    
    if args.play and paths.wav and HAS_AUDIO:
        import soundfile as sf
        try:
            data, sr = sf.read(str(paths.wav), dtype="float32")
            play_audio(data, sr)
        except Exception:
            log.warning("Install soundfile for playback of saved WAV, or use --play with 'modulate'")
    
    result["files"] = {
        "wav": str(paths.wav) if paths.wav else None,
        "iq": str(paths.iq) if paths.iq else None,
        "meta": str(paths.meta) if paths.meta else None,
        "png": str(paths.png) if paths.png else None
    }
    
    print(safe_json(result))
    return 0

def cmd_modulate(args: argparse.Namespace) -> int:
    mcfg = make_modcfg(args)
    fcfg = FrameConfig()
    sec = SecurityConfig(password=args.password, watermark=args.watermark, hmac_key=args.hmac_key)
    scheme = parse_scheme(args.scheme)
    fec_s = parse_fec(args.fec)
    paths = full_cast_and_save(
        text=args.text, outdir=Path(args.outdir), scheme=scheme, mcfg=mcfg, fcfg=fcfg,
        sec=sec, fec_scheme=fec_s, want_wav=args.wav or (not args.iq), want_iq=args.iq,
        title=f"{scheme.name} | Direct Mod"
    )
    if args.play and paths.wav:
        try:
            import soundfile as sf
            data, sr = sf.read(str(paths.wav), dtype="float32"); play_audio(data, sr)
        except Exception:
            log.warning("Install soundfile for playback of saved WAV")
    print(safe_json({"files": {"wav": str(paths.wav) if paths.wav else None,
                               "iq": str(paths.iq) if paths.iq else None,
                               "meta": str(paths.meta) if paths.meta else None,
                               "png": str(paths.png) if paths.png else None}}))
    return 0

def cmd_visualize(args: argparse.Namespace) -> int:
    if not HAS_MPL:
        print("matplotlib is not installed.")
        return 1
    import wave
    with wave.open(args.wav, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        s = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)/32767.0
    out = Path(args.out or (Path(args.wav).with_suffix(".png")))
    plot_wave_and_spectrum(out, s, sr, f"Visualize: {Path(args.wav).name}")
    print(safe_json({"png": str(out), "sample_rate": sr, "seconds": len(s)/sr}))
    return 0

def cmd_analyze(args: argparse.Namespace) -> int:
    import wave
    with wave.open(args.wav, "rb") as w:
        sr = w.getframerate(); n = w.getnframes()
        s = np.frombuffer(w.readframes(n), dtype=np.int16).astype(np.float32)/32767.0
    dur = len(s)/sr
    rms = float(np.sqrt(np.mean(s**2)))
    peak = float(np.max(np.abs(s)))
    spec = np.abs(rfft(s)); spec /= (spec.max()+1e-12)
    # simple SNR estimate
    snr = 10*np.log10(np.mean(s**2) / (np.var(s - np.mean(s)) + 1e-12))
    print(safe_json({"sample_rate": sr, "seconds": dur, "rms": rms, "peak": peak, "snr_db": float(snr)}))
    return 0

def cmd_neuro_symbolic(args: argparse.Namespace) -> int:
    engine = MirrorCastEngine()
    analysis = engine.cast(args.text)
    print(safe_json(analysis))
    
    if args.plot and HAS_MPL:
        # Plot fractal layers
        if "fractal" in analysis:
            layers = analysis["fractal"]["layers"]
            depths = [l["depth"] for l in layers]
            entropies = [l["entropy"] for l in layers]
            plt.figure(figsize=(10, 6))
            plt.subplot(2, 2, 1)
            plt.plot(depths, entropies, 'o-')
            plt.title("Fractal Entropy vs Depth")
            plt.xlabel("Depth")
            plt.ylabel("Entropy")
            plt.grid(True)
        
        # Plot semantic mapping
        if "semantic" in analysis:
            plt.subplot(2, 2, 2)
            categories = list(analysis["semantic"].keys())
            scores = list(analysis["semantic"].values())
            plt.bar(categories, scores)
            plt.title("Semantic Mapping")
            plt.xticks(rotation=45)
            plt.grid(True)
        
        # Plot love infusion
        if "love" in analysis:
            plt.subplot(2, 2, 3)
            love_data = analysis["love"]
            metrics = ["emotional_resonance", "love_quotient", "harmony_index"]
            values = [love_data.get(m, 0) for m in metrics]
            plt.bar(metrics, values)
            plt.title("Love Infusion Metrics")
            plt.xticks(rotation=45)
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("neuro_symbolic_analysis.png")
        plt.show()
    
    return 0

def cmd_learn(args: argparse.Namespace) -> int:
    engine = MirrorCastEngine()
    planner = AdaptiveLinkPlanner(db_path=args.db_path)
    
    for ep in range(args.episodes):
        analysis = engine.cast(args.text)
        cfg, explanation = planner.plan(args.text, ModConfig(), analysis, args.snr_db)
        
        # Simulate transmission (placeholder - in real implementation, this would use actual modem)
        success = np.random.random() > 0.3  # 70% success rate for demo
        harmony = analysis.get("love", {}).get("harmony_index", 0.7)
        
        planner.reward_and_record(
            args.text, cfg, explanation, success, args.snr_db, 
            harmony, len(args.text), len(args.text) if success else 0
        )
        
        print(json.dumps({
            "episode": ep + 1,
            "config": {
                "symbol_rate": cfg.symbol_rate,
                "scheme": "bpsk" if cfg.symbol_rate <= 1200 else "qpsk" if cfg.symbol_rate <= 2400 else "ofdm"
            },
            "success": success,
            "explanation": explanation
        }, indent=2))
    
    print(f"Learning complete. DB saved to {args.db_path}")
    return 0

def main(argv: Optional[List[str]] = None) -> int:
    p = build_parser()
    args = p.parse_args(argv)
    if args.cmd == "cast": return cmd_cast(args)
    if args.cmd == "modulate": return cmd_modulate(args)
    if args.cmd == "visualize": return cmd_visualize(args)
    if args.cmd == "analyze": return cmd_analyze(args)
    if args.cmd == "neuro-symbolic": return cmd_neuro_symbolic(args)
    if args.cmd == "learn": return cmd_learn(args)
    p.print_help(); return 2

if __name__ == "__main__":
    raise SystemExit(main())
Thought for 27s

I'll look through the workspace to find the mixed script, then split and fix it into clean Julia and Python modules, remove duplicates/markers, and repair any typos so both servers run. Then I'll wire up simple start commands.
Listed items in current directory and thought for 5s

I'll scan the codebase to see if any existing files overlap with the pasted Julia/Python servers, then create or fix files accordingly.
Listed items in current directory and thought for 23s

I'll open the top-level Project.toml to verify Julia dependencies, and then create new files for the Julia server and Python API, cleaning duplicates and fixing typos, plus a small README and requirements.
Read Project.toml and requirements.txt and listed items in 2 directories and thought for 73s

I'll create a small task list and start by updating dependencies and adding the cleaned Julia server, Python API, static UI, and helper scripts.
Updated todo list and thought for 43s

I'm going to update the Julia dependencies, add the cleaned Julia server, create the Python API and static UI, add helper scripts, and update Python requirements.
Used apply_patch and updated todo list

I'll add the cleaned Julia server file under services/julia and ensure the code from your message is placed there (fixed where needed). Then I'll check lints.
Used apply_patch and updated todo list

I added required Julia deps and a cleaned src/qvnm_server.jl server. Next, I’ll add the Python FastAPI service and update requirements.
Used apply_patch 6 times and updated todo list 5 times and thought for 16s

I'll make the run scripts executable so you can start services easily. Then I'll quickly check for linter issues on the new files.
Ran: chmod +x /workspace/run_julia.sh /workspace/run_ap...

    I added missing Julia deps to Project.toml, created src/qvnm_server.jl with the cleaned PolyServe+QVNM server, built a cleaned FastAPI bridge at api.py, added a minimal UI at static/qvnm.html, included a stubbed dual_llm_wavecaster_enhanced.py, and provided run scripts run_julia.sh and run_api.sh (now executable).

You can start:

    Julia server: ./run_julia.sh (PORT defaults to 9000)
    Python API: ./run_api.sh (PORT defaults to 8000)

Then open /ui/qvnm.html on the Python API to try upload → estimate → preview.
