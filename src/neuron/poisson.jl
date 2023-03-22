#FT=Float32
abstract type AbstractPoisson end

struct Poisson{X<:Array{Bool},Y<:Array{<:Any}}
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

function integrate!(p::Poisson, dt::Float32)
    @unpack N, randcache, fire = p
    @unpack rate = 1Hz
    prob = rate * dt
    rand!(randcache)
    @inbounds for i = 1:N
        fire[i] = randcache[i] < prob
    end
end
