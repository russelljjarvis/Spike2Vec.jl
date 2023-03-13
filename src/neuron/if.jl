
abstract type AbstractIFParameter end
struct IFParameter# <: AbstractIFParameter
    τm::Float32# = 20ms
    τe::Float32# = 5ms
    τi::Float32# = 10ms
    Vt::Float32# = -50mV
    Vr::Float32# = -60mV
    El::Float32# = Vr
    function IFParameter()
        new(20.0,5.0,10.0,-50.0,-60.0,-35.0)
    end

end

abstract type AbstractIF end

struct IF <: AbstractIF
    param::IFParameter
    N::Int32
    v::Vector{Float32} 
    ge::Vector{Float32}
    gi::Vector{Float32}
    fire::Vector{Bool}
    I::Vector{Float32}
    records::Dict
    function IF(;N::Int64,param::SpikingNeuralNetworks.IFParameter)
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
    @unpack τm, τe, τi, Vt, Vr, El = param
    @inbounds for i = 1:N
        v[i] += dt * (ge[i] + gi[i] - (v[i] - El) + I[i]) / τm
        ge[i] += dt * -ge[i] / τe
        gi[i] += dt * -gi[i] / τi
        fire[i] = v[i] > Vt
        v[i] = ifelse(fire[i], Vr, v[i])
    end
end