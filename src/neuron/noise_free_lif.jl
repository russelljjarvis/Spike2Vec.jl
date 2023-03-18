#=
abstract type AbstractIFParameter end
struct IFParameter# <: AbstractIFParameter
    τm::Float32# = 20ms
    τe::Float32# = 5ms
    τi::Float32# = 10ms
    Vt::Float32# = -50mV yes
    Vr::Float32# = -60mV
    El::Float32# = Vr
    gl::Float32# = Vr

    #{'V_th': -55.0, 'V_reset': -75.0, 'tau_m': 10.0, 'g_L': 10.0, 'V_init': -75.0, 'E_L': -75.0, 'tref': 2.0, 'T': 400.0, 'dt': 0.1, 'range_t': array([0.000e+00, 1.000e-01, 2.000e-01, ..., 3.997e+02, 3.998e+02,
    #   3.999e+02])}
    function IFParameter()
        new(10.0,5.0,10.0,-50.0,-60.0,-75.0,10.0)
    end

end
=#


abstract type AbstractIFParameter end
struct IFParameter <: AbstractIFParameter
    τm::Float32# = 20ms
    τe::Float32# = 5ms
    τi::Float32# = 10ms
    Vt::Float32# = -50mV
    Vr::Float32# = -60mV
    El::Float32# = Vr
    gL::Float32# = Vr
    #τm = 100
    #vreset = 0.0
    #vth = 0.1
    #R = 1.75
    
    
    function IFParameter()
        new(100.0,5.0,10.0,0.1,0.0,-60.0,10.0)
    end

end

abstract type AbstractIFNF end

struct IFNF <: AbstractIFNF
    param::IFParameter
    N::Int32
    v::Vector{Float32} 
    ge::Vector{Float32}
    gi::Vector{Float32}
    fire::Vector{Bool}
    I::Vector{Float32}
    records::Dict
    function IFNF(;N::Int64,param::SpikingNeuralNetworks.IFParameter,I::Vector{Float32})
        VFT = Vector{Float32}
        v::VFT = param.Vr .* ones(N) 
        ge::VFT = zeros(N)
        gi::VFT = zeros(N)
        fire::Vector{Bool} = zeros(Bool, N)
        #I::VFT = zeros(N)
        records::Dict = Dict()
        new(param,N,v,ge,gi,fire,I,records)
    end
    function IFNF(;N::Int64,param::SpikingNeuralNetworks.IFParameter)
        VFT = Vector{Float32}
        v::VFT = param.Vr .* ones(N) #.* (param.Vt - param.Vr)
        ge::VFT = zeros(N)
        gi::VFT = zeros(N)
        fire::Vector{Bool} = zeros(Bool, N)
        I::VFT = zeros(N)
        records::Dict = Dict()
        new(param,N,v,ge,gi,fire,I,records)
    end
end

#IFNF(;N::Int64,param::SpikingNeuralNetworks.IFParameter)



Base.show(io::IO, ::MIME"text/plain", neuron::IFParameter) =
    print(io, """LIF:
                     voltage: $(neuron.v)
                     current: $(neuron.I)
                     τm:      $(neuron.τm)
                     Vr    :  $(neuron.Vr)
                     R:       $(neuron.R)""")
Base.show(io::IO, neuron::IFNF) =
    print(io, "LIF(voltage = $(neuron.v), current = $(neuron.I))")

"""
    LIF(τm, vreset, vth, R = 1.0)
Create a LIF neuron with zero initial voltage and empty current queue.
"""
#LIF(τm::Real, vreset::Real, R::Real = 1.0) = LIF{Float32, Int}(vreset, 0, 0, τm, vreset, R)

"""
    [Integrate-And-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
"""
IF

function integrate!(p::IFNF, dt::Float32)
    @unpack N, v, ge, gi, fire, I = p
    τm, τe, τi, Vt, Vr, El, gL = (100.0,5.0,10.0,0.2,0.0,-60.0,10.0)
    
    R = 1.75
    @inbounds for i = 1:N
        X = ge[i] + gi[i]
        u = I[i]
        v[i] = v[i] * exp(-dt /τm) + Vr +X+u        
        v[i] += (I[i]+X) * (R/ τm)
        ge[i] += dt * -ge[i] / τe
        gi[i] += dt * -gi[i] / τi
        fire[i] = v[i] > Vt
        v[i] = ifelse(fire[i], Vr, v[i])
        
    end
end
