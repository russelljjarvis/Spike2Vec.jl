abstract type AbstractIFParameter end
@snn_kw struct IFParameter{FT=Float32} <: AbstractIFParameter
    τm::FT = 20ms
    τe::FT = 5ms
    τi::FT = 10ms
    Vt::FT = -50mV
    Vr::FT = -60mV
    El::FT = Vr
end

abstract type AbstractIF end

@snn_kw mutable struct IF{VFT=Vector{Float32},VBT=Vector{Bool}} <: AbstractIF
    param::IFParameter = IFParameter()
    N::Int32 = 100
    v::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr)
    ge::VFT = zeros(N)
    gi::VFT = zeros(N)
    fire::VBT = zeros(Bool, N)
    I::VFT = zeros(N)
    records::Dict = Dict()
end

"""
    [Integrate-And-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
"""
IF

function integrate!(p::IF, param::IFParameter, dt::Float32)
    @inbounds for i = 1:p.N
        p.v[i] += dt * (p.ge[i] + p.gi[i] - (p.v[i] - param.El) + p.I[i]) / param.τm
        p.ge[i] += dt * -p.ge[i] / param.τe
        p.gi[i] += dt * -p.gi[i] / param.τi
    #end
    #@inbounds for i = 1:p.N
        p.fire[i] = p.v[i] > param.Vt
        p.v[i] = ifelse(p.fire[i], param.Vr, p.v[i])
    end
end
