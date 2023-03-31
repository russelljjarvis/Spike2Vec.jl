using KernelAbstractions
using KernelAbstractions: @atomic, @atomicswap, @atomicreplace
using Revise
abstract type AbstractIFNF end


mutable struct IFNF{C<:Integer,Q<:AbstractArray{<:Bool},L<:AbstractVecOrMat{<:Real}} <: AbstractIFNF
    N::C
    v::L 
    ge::L
    gi::L
    fire::Q 
    u::L
    tr::L
    records::Dict

    function IFNF(N,v,ge,gi,fire,u,tr,records)
        new{typeof(N),typeof(fire),typeof(ge)}(N,v,ge,gi,fire,u,tr,records)
    end

    function IFNF(N,fire,u,sim_type)
        v = typeof(sim_type)(ones(N).-55.) 
        g = typeof(sim_type)(zeros(N))
        ge = typeof(sim_type)(zeros(N))
        gi = typeof(sim_type)(zeros(N))       
        tr = zeros(typeof(N),N)
        records::Dict = Dict()
        IFNF(N,v,ge,gi,fire,u,tr,records)
    end 

    function IFNF(N,sim_type::CuArray,u)
        fire::CuArray{Bool} = zeros(Bool,N)
        IFNF(N,fire,u,sim_type)
    end 
 
    function IFNF(N,sim_type::Array,u)
        fire::Array{Bool} = zeros(Bool,N)
        IFNF(N,fire,u,sim_type)
    end 
 

    function IFNF(N,sim_type)
        u = typeof(sim_type)(zeros(N))
        IFNF(N,sim_type,u)
    end 
    

end    

"""
    [Integrate-And-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
"""
IFNF

function integrate!(p::IFNF, dt::Float32)
    @unpack N, v, ge, gi, fire, u, tr = p
    integrate!(N, v, dt, ge, gi, fire, u, tr)

end


function integrate!(N::Integer,v::CuArray,dt::Real,ge::CuVector,gi::CuVector,fire::CuArray{Bool},u::CuVector,g::CuVector)
    kernel = @cuda launch=false lif_kernel!(N, v, ge, gi, fire,u,dt,g,tr)
    config = launch_configuration(kernel.fun)
    xthreads = min(32, N)
    xblocks = min(config.blocks, cld(N, xthreads))
    kernel(N, v, ge, gi, fire, u,dt;threads=(xthreads), blocks=(xblocks<<2))

end

function integrate!(N::Integer,v::Vector,dt::Real,ge::Vector,gi::Vector,fire::Vector{Bool},u::Vector{<:Real},tr::Vector{<:Number})
    τe, τi = 5.0,10.0
    τ::Real = 8.         
    R::Real = 10.      
    θ::Real = -50.     
    vSS::Real =-55.
    v0::Real = -100. 
    tref = 10.0

    @inbounds for i = 1:N

        ge[i] += dt * -ge[i] / τe        
        gi[i] += dt * -gi[i] / τi
        g = ge[i] + gi[i]           
        v[i] = v[i] + (g+u[i]) * R / τ
        v[i] += (dt/τ) * (-v[i] + vSS)
        if tr[i] > 0  # check if in refractory period
            v[i] = vSS  # set voltage to reset
            tr[i] = tr[i] - 1 # reduce running counter of refractory period
        elseif v[i] >  θ
            fire[i] = v[i] >  θ
            tr[i] = Int(round(tref*dt))  # set refractory time
        end        
    end

end
function lif_kernel!(N, v, ge, gi, fire, u,dt,g,tr)
    τm, τe, τi, Vt, Vr, El, gL = (100.0,5.0,10.0,0.2,0.0,-60.0,10.0)
    R = 1.75
    i0 = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    stride = blockDim().x
    g = g + (ge + gi)
    tref = 10.0

    @inbounds for i=i0:stride:N
        v[i] = v[i] * exp(-dt /τm) + Vr +X+g[i]+u[i]        
        v[i] += (u[i]+g[i]) * (R/ τm)
        ge[i] += dt * -ge[i] / τe
        gi[i] += dt * -gi[i] / τi
        g[i] += dt * -g[i] / (τi+τe)
        if tr[i] > 0  # check if in refractory period
            v[i] = vSS  # set voltage to reset
            tr[i] = tr[i] - 1 # reduce running counter of refractory period
        elseif v[i] >  θ
            fire[i] = v[i] >  θ
            tr[i] = Int(round(tref*dt))  # set refractory time
        end
    end
    nothing
end



Base.show(io::IO, ::MIME"text/plain") =
    print(io, """LIF:
                     voltage: $(neuron.v)
                     current: $(neuron.u)
                     τm:      $(neuron.τm)
                     Vr    :  $(neuron.Vr)""")
Base.show(io::IO, neuron::IFNF) =
    print(io, "LIF(voltage = $(neuron.v), current = $(neuron.u))")

"""
    LIF(τm, vreset, vth, R = 1.0)
Create a LIF neuron with zero initial voltage and empty current queue.
"""
