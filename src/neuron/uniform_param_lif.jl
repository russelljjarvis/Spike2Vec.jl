using Revise
using CUDA
"""
# * Changes to the cells struct as follows:
# * Make the typing used dynamic and parametric, ie types can be either CuArray or regular Array, depending the input
# arguments that are passed to the constructor.
# TODO, I have good things about KernelAbstractions.jl it possible that kernel abstractions will remove my reliance on different method dispatch to choose CPU/GPU backend.
# * I added in refractory periods.
# * I made examples of CUDA kernel calls for cell Vm update integrate, note it is not faster yet, but I am working on it.
# * Stored the post synaptic targets of each pre-synaptic cell populations cell struct for the reason described below
## TODO, as needed: 
# * Since each cuda kernel has a cost, do as much work as possible per cuda kernel call.
# to do as much work as possible in a cuda kernel, use code macros as a way to update both: `V_{m} ` 
# and weights together in the same loop, to make the folding together of synaptic weight updates with Vm updates, store the post
# synaptic targets of each pre-synaptic cell populations cell struct.
Note the CUDA kernels execute and produce results, but they are not yet as fast, or faster than CPU. To make the CUDA kernels 
fast and merited I would explore 2D threads (using `.x`, and `.y`). Folding Vm update togethor with synaptic update, to maximise
work done in each cuda kernel call.
Example of a LIF update as a CUDA kernel
"""
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

"""
Example of the multidispatch integrate function that is chosen to execute when `fire` is of CuArray type. 
"""
function integrate!(N::Integer,v::CuArray,dt::Real,ge::CuVector,gi::CuVector,fire::CuArray{Bool},u::CuVector,g::CuVector)
    kernel = @cuda launch=false lif_kernel!(N, v, ge, gi, fire,u,dt,g,tr)
    config = launch_configuration(kernel.fun)
    xthreads = min(32, N)
    xblocks = min(config.blocks, cld(N, xthreads))
    kernel(N, v, ge, gi, fire, u,dt;threads=(xthreads), blocks=(xblocks<<2))

end


abstract type AbstractIFNF end
"""
A population of cells
"""

mutable struct IFNF{C<:Integer,Q<:AbstractArray{<:Bool},L<:AbstractVecOrMat{<:Real}} <: AbstractIFNF
    N::C
    v::L 
    ge::L
    gi::L
    fire::Q 
    u::L
    tr::L
    records::Dict
    post_synaptic_targets::Vector{Any} # SVector

    function IFNF(N,v,ge,gi,fire,u,tr,records,post_synaptic_targets)
        new{typeof(N),typeof(fire),typeof(ge)}(N,v,ge,gi,fire,u,tr,records,post_synaptic_targets)

    end

    function IFNF(N,fire,u,sim_type::Array)
        v = typeof(sim_type)(ones(N).-55.0) 
        g = typeof(sim_type)(zeros(N))
        ge = typeof(sim_type)(zeros(N))
        gi = typeof(sim_type)(zeros(N))       
        tr = zeros(typeof(N),N)
        
        post_synaptic_targets = Array{Array{UInt64}}(undef,N)
        for i in 1:N
            post_synaptic_targets[i] = Array{UInt64}([])
        end
        #post_synaptic_targets = SVector{N, Array{UInt32}}(post_synaptic_targets)
        pre_synaptic_weights = Vector{Float32}(zeros(N))
       
        records::Dict = Dict()
        IFNF(N,v,ge,gi,fire,u,tr,records,post_synaptic_targets)
    end 
    function IFNF(N,fire,u,post_synaptic_targets::Vector{Any})
        v = typeof(u)(ones(N).-55.) 
        g = typeof(u)(zeros(N))
        ge = typeof(u)(zeros(N))
        gi = typeof(u)(zeros(N))       
        tr = zeros(typeof(N),N)
        records::Dict = Dict()
        IFNF(N,v,ge,gi,fire,u,tr,records,post_synaptic_targets)
    end 
    function IFNF(N,sim_type::CuArray,post_synaptic_targets::Vector{Any})
        fire::CuArray{Bool} = zeros(Bool,N)
        u = typeof(sim_type)(zeros(N))
        IFNF(N,fire,u,post_synaptic_targets)
    end 
 
    function IFNF(N,sim_type::Array,post_synaptic_targets::Vector{Any})
        fire::Array{Bool} = zeros(Bool,N)
        u = typeof(sim_type)(zeros(N))
        IFNF(N,fire,u,post_synaptic_targets)
    end 



end    
"""
    [Uniform parameter: Integrate-And-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
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
    #dt dt*ms

    @inbounds for i = 1:N

        ge[i] += dt * -ge[i] / τe        
        gi[i] += dt * -gi[i] / τi
        g = ge[i] + gi[i]           
        v[i] = v[i] + (g+u[i]) * R / τ
        v[i] += (dt/τ) * (-v[i] + vSS)# *V
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

