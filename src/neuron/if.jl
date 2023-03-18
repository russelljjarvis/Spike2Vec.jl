
struct IF <: AbstractIF
    param::IFParameter
    N::Int32
    v::Vector{Float32} 
    ge::Vector{Float32}
    gi::Vector{Float32}
    fire::Vector{Bool}
    I::Vector{Float32}
    records::Dict
    function IF(;N::Int64,param::SpikingNeuralNetworks.IFParameter,I::Vector{Float32})
        VFT = Vector{Float32}
        v::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
        ge::VFT = zeros(N)
        gi::VFT = zeros(N)
        fire::Vector{Bool} = zeros(Bool, N)
        I::VFT = zeros(N)
        records::Dict = Dict()
        new(param,N,v,ge,gi,fire,I,records)
    end
end

"""
    [Integrate-And-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
"""
IF

function integrate!(p::IF, param::IFParameter, dt::Float32)
    @unpack N, v, ge, gi, fire, I = p
    @unpack τm, τe, τi, Vt, Vr, El, gL = param
    vtemp = v
    @inbounds for i = 1:N
        v[i] += dt * (ge[i] + gi[i] - (v[i] - El) + I[i]) /gL* τm
        ge[i] += dt * -ge[i] / τe
        gi[i] += dt * -gi[i] / τi
        fire[i] = v[i] > Vt
    end
    @inbounds for i = 1:N
        v[i] = ifelse(fire[i], Vr, v[i])
    end
    
end
