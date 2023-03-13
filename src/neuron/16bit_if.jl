
abstract type AbstractIF16Parameter end

struct IFParameter16 <: AbstractIF16Parameter #<: AbstractIFParameter
    τm::Float16 # = 20#ms
    τe::Float16 # = 5#ms
    τi::Float16 # = 10#ms
    Vt::Float16 # = -50#mV
    Vr::Float16 # = -60#mV
    El::Float16 # = -49mV
end

function IFParameter16()#;τm::Float32,τe::Float32,τi::Float32,Vt::Float32,Vr::Float32,El::Float32)#20.0,5.0,10.0,-50.0,-60.0,-65.0)
    return IFParameter16(20ms,5.0,10.0,-50.0,-60.0,-49mV)
end    
#function IFParameter16(;τm::Float16=20ms,τe::Float16=5.0,τi::Float16,Vt::Float16,Vr::Float16,El::Float16)
#    return IFParameter16(τm,τe,τi,Vt,Vr,El)
#end
abstract type AbstractIF end

@snn_kw mutable struct IF16{VFT=Vector{Float32},VBT=Vector{Bool}} <: AbstractIF

    #=
    Population container.
    =#

    ## N is the number of neurons.
    param::IFParameter16 = IFParameter16()
    N::Int32 = 100
    v::VFT = param.Vr .+ rand(N) .* (param.Vt - param.Vr) #./10.0
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

#nested task error: MethodError: no method matching integrate!(::SpikingNeuralNetworks.IF16{Vector{Float16}, Vector{Bool}}, ::SpikingNeuralNetworks.IFParameter, ::Float16)
#function integrate!(p::IF, param::IFParameter, dt::Float32)


function integrate!(p::IF16, param::IFParameter16, dt::Float16)
    @unpack N, v, ge, gi, fire, I = p

    @unpack τm, τe, τi, Vt, Vr, El = param
    @inbounds for i = 1:N
        v[i] += dt * (ge[i] + gi[i] - (v[i] - El) + I[i]) / τm
        ge[i] += dt * -ge[i] / τe
        gi[i] += dt * -gi[i] / τi
    end
    @inbounds for i = 1:N
        fire[i] = v[i] > Vt
        if fire[i]
        end
        v[i] = ifelse(fire[i], Vr, v[i])
    end
end
