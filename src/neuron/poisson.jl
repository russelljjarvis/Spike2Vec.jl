FT=Float32

struct PoissonParameter
    rate::FT # = 1#Hz
    function PoissonParameter(FT::value_) 
        new(value_)
    end
end

struct Poisson{VFT=Vector{Float32},VBT=Vector{Bool}}
    param::PoissonParameter = PoissonParameter(1.0)
    N::Int32 = 100
    randcache::VFT = rand(N)
    fire::VBT = zeros(Bool, N)
    records::Dict = Dict()
end

"""
[Poisson Neuron](https://www.cns.nyu.edu/~david/handouts/poisson.pdf)
"""
Poisson

function integrate!(p::Poisson, param::PoissonParameter, dt::Float32)
    @unpack N, randcache, fire = p
    @unpack rate = param
    prob = rate * dt
    rand!(randcache)
    @inbounds for i = 1:N
        fire[i] = randcache[i] < prob
    end
end
