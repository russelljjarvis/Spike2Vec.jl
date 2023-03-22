struct RateSynapseParameter{Q<:Number}
    lr::Q # = 1e-3
end

#struct SpikingSynapse{T,S,Q} <: AbstractSpikingSynapse
abstract type AbstractRateSynapse end


struct RateSynapse{VI,VF} <: AbstractRateSynapse
    param::RateSynapseParameter # = RateSynapseParameter()
    N::Int
    g::VF  # postsynaptic conductance
    p::Float32
    o::Float32

    records::Dict# = Dict()
    
    function RateSynapse(param,N,g,o)
        rJ::Vector{Any} = ones(length(colptr)) .* param.lr
        new{typeof(colptr),typeof(rJ)}(param,N,g,o,Dict())
    end

    function RateSynapse(pre, post; σ = 0.0, p = 0.0)
        #w = σ / √(p * pre.N) * sprandn(post.N, pre.N, p)
        #rowptr, colptr, I, J, index, W = dsparse(w)
        g = post.g
        N = pre.N
        param = RateSynapseParameter(1e-3)
        RateSynapse(param,N,g,o,p)
    end
end

"""
[Rate Synapse](https://brian2.readthedocs.io/en/2.0b4/resources/tutorials/2-intro-to-brian-synapses.html)
"""
RateSynapse


function forward!(c::RateSynapse)# param::RateSynapseParameter)
    @unpack param, N, g, p, o = c
    lr = param.lr
    g .= 0.0
    #@fill!(g, zeros(eltype(g)))
    @inbounds for j in 1:(length(colptr) - 1)
        #rJj = rJ[j]
        s = colptr[j]:(colptr[j+1] - 1)
        g[I[s]] += rand()#W[s] * rJj
        #end
    end
end
#=
function plasticity!(c::RateSynapse, param::RateSynapseParameter, dt::Float32, t::Float32)
    @unpack colptr, I, W, rI, rJ, g = c
    @unpack lr = param
    @inbounds for j in 1:(length(colptr) - 1)
        s_row = colptr[j]:(colptr[j+1] - 1)
        rIW = zero(Float32)
        for s in s_row
            rIW += rI[I[s]] * W[s]
        end
        Δ = lr * (rJ[j] - rIW)
        for s in s_row
            W[s] += rI[I[s]] * Δ
        end
    end
end
=#