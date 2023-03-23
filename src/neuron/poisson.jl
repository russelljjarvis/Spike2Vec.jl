
abstract type AbstractPoisson end

mutable struct Poisson{X<:Array{Bool},Y<:Array{<:Any}}
    N::Int # ::Int32 = 100
    randcache::Y # ::VFT = rand(N)
    fire::X # ::VBT = zeros(Bool, N)
    records::Dict #::Dict = Dict()

    function Poisson(N,x)
        r = tanh.(x)
        randcache ::Vector{Float32} = rand(N)
        fire ::Vector{Bool} = zeros(Bool, N)
        dict = Dict()  
        new{typeof(fire),typeof(randcache)}(N,randcache,fire,dict)
    end
end

"""
[Poisson Neuron](https://www.cns.nyu.edu/~david/handouts/poisson.pdf)
"""
Poisson
#ntegrate!(::SpikingNeuralNetworks.SpikingSynapse{Vector{Float16}, Vector{Int32}, Vector{Bool}}, ::Float32)

function integrate!(p::Poisson, dt::Float32)
    @unpack N, randcache, fire = p
    rate = 1Hz
    prob = rate * dt
    rand(randcache)
    @inbounds for i = 1:N
        fire[i] = randcache[i] < 0.000125
    end
end
#=
function forward!(c::SpikingSynapse)
    @unpack colptr, I, W, fireI,fireJ, g = c
    forward!(colptr, I, W, fireI,fireJ, g)
end

function forward!(colptr::Vector{<:Real}, I, W, fireI::Vector{Bool},fireJ::Vector{Bool},g::Vector)

    #@inbounds for j in 1:length(g)
    #    g[j] = 0.0
    #end
    #g .= 0.0
    #@inbounds for j in colptr[fireJ]
    #    s = colptr[j]:(colptr[j+1] - 1)
    #    g[I[s]] += W[s]
    #end
    @inbounds for j in 1:(length(colptr) - 1)
        if fireJ[j]
            for s in colptr[j]:(colptr[j+1] - 1)
                g[I[s]] += W[s]
            end
        end
    end
    replace!(g, Inf=>0.0)
    replace!(g, NaN=>0.0)

    #g[:]
    #@show(sum(g))
end
=#