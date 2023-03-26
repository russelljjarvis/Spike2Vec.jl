
#using CUDA
#CUDA.allowscalar(false)

using KernelAbstractions
using KernelAbstractions: @atomic, @atomicswap, @atomicreplace
#include(joinpath(dirname(pathof(KernelAbstractions)), "../examples/utils.jl")) # Load backend
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
        #g = typeof(sim_type)(zeros(N))       
        tr = zeros(typeof(N),N)
        records::Dict = Dict()
        IFNF(N,v,ge,gi,fire,u,tr,records)
    end 
    
    #new{typeof(N),typeof(fire),typeof(ge)}(N,v,ge,gi,fire,u,tr,records)
     #(::UInt64, ::Vector{Float64}, ::Vector{Float64}, ::Vector{Float64}, ::Vector{Bool}, ::Vector{Float64}, ::Vector{UInt64}, ::Dict{Any, Any})

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

#integrate!(::UInt32, ::Vector{Float16}, ::Float32, ::Vector{Float16}, ::Vector{Float16}, ::Vector{Bool}, ::Vector{Float16})

function integrate!(N::Integer,v::Vector,dt::Real,ge::Vector,gi::Vector,fire::Vector{Bool},u::Vector{<:Real},tr::Vector{<:Number})
    τe, τi = 5.0,10.0
    #,0.2,0.0,-60.0,10.0)    
    #{'V_th': -55.0, 'V_reset': -75.0, 'tau_m': 10.0, 'g_L': 10.0, 'V_init': -75.0, 'E_L': -75.0, 'tref': 2.0, 'T': 400.0, 'dt': 0.1, 'range_t': array([0.000e+00, 1.000e-01, 2.000e-01, ..., 3.997e+02, 3.998e+02,
    #3.999e+02])}
    τ::Real = 8.         
    R::Real = 10.      
    θ::Real = -50.     
    vSS::Real =-55.
    v0::Real = -100. 
    tref = 10.0
    println("from cell model")
    @show(gi)

    println("from cell model")
    @show(ge)
    @inbounds for i = 1:N
        #state = neuron.state + input_update * neuron.R / neuron.τ
        #@show(ge[i])
        #@show(gi[i])
        # Euler method update
        #state += 1000 * (dt/neuron.τ) * (-state + neuron.vSS)
        #if ge[i]>0 
        #    @show(ge[i])
        #end
        ge[i] += dt * -ge[i] / τe
        #end
        #if gi[i]>0 
        
        gi[i] += dt * -gi[i] / τi
        #end
        #if ge[i]>0 || gi[i] >0
        g = ge[i] + gi[i]           
        #end
        
        #@show(g[i])
        
        v[i] = v[i] + (g+u[i]) * R / τ
        # Euler method update
        #@show(v[i])

        v[i] += (dt/τ) * (-v[i] + vSS)
        if tr[i] > 0  # check if in refractory period
            v[i] = vSS  # set voltage to reset
            tr[i] = tr[i] - 1 # reduce running counter of refractory period
            #print("fire lif")
        elseif v[i] >  θ
            fire[i] = v[i] >  θ
            tr[i] = Int(round(tref*dt))  # set refractory time
        end


        
    end

end
function lif_kernel!(N, v, ge, gi, fire, u,dt,g)
    τm, τe, τi, Vt, Vr, El, gL = (100.0,5.0,10.0,0.2,0.0,-60.0,10.0)
    R = 1.75
    i0 = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    stride = blockDim().x
    g = g + (ge + gi)
    
    @inbounds for i=i0:stride:N
        #X = ge[i] + gi[i]
        #u = u[i]
        v[i] = v[i] * exp(-dt /τm) + Vr +X+g[i]+u[i]        
        v[i] += (u[i]+g[i]) * (R/ τm)
        ge[i] += dt * -ge[i] / τe
        gi[i] += dt * -gi[i] / τi
        g[i] += dt * -g[i] / (τi+τe)
        fire[i] = v[i] > Vt
        v[i] = ifelse(fire[i], Vr, v[i])
    end
    nothing
end



Base.show(io::IO, ::MIME"text/plain") =
    print(io, """LIF:
                     voltage: $(neuron.v)
                     current: $(neuron.u)
                     τm:      $(neuron.τm)
                     Vr    :  $(neuron.Vr)
                     R:       $(neuron.R)""")
Base.show(io::IO, neuron::IFNF) =
    print(io, "LIF(voltage = $(neuron.v), current = $(neuron.u))")

"""
    LIF(τm, vreset, vth, R = 1.0)
Create a LIF neuron with zero initial voltage and empty current queue.
"""
#LIF(τm::Real, vreset::Real, R::Real = 1.0) = LIF{Float32, Int}(vreset, 0, 0, τm, vreset, R)


    #=
    function IFNF(;N::Integer,u::Vector{Float32})
        N::Integer = N
        PT=Vector{Float32}
        v::PT = zeros(N) 
        ge::PT = zeros(N)
        gi::PT = zeros(N)
        fire::Vector{Bool} = zeros(Bool, N)
        records::Dict = Dict()
        pop_indexs::Integer = 1
        new(N,pop_indexs,v,ge,gi,fire,I,records)
    end 

    function IFNF(;N::Integer,I::CuArray{Float32})
        PT=CuArray{Float32}
        N::Integer = N
        v::PT = zeros(N) 
        ge::PT = zeros(N)
        gi::PT = zeros(N)
        fire::CuArray{Bool} = zeros(Bool, N)
        records::Dict = Dict()
        pop_indexs::Integer = 1
        new(N,pop_indexs,v,ge,gi,fire,I,records)
    end 
    =#
    
    



#IFNF(N::Integer) = IFNF{Vector{Float32}}(N,pop_indexs,v,ge,gi,fire,I,records)
#IFNF(N::Integer) = IFNF{CuArray{Float32}}(N,pop_indexs,v,ge,gi,fire,I,records)
#IFNF(;N::Integer,I::Vector{Float32}) = IFNF{Vector{Float32}}(N,pop_indexs,v,ge,gi,fire,I,records)
#IFNF(;N::Integer,I::CuArray{Float32}) = IFNF{CuArray{Float32}}(N,pop_indexs,v,ge,gi,fire,I,records)

#IFNF(;N::Integer,I::CuArray{Float32},pop_indexs::Integer) = IFNF{CuArray{Float32}}(N,pop_indexs,v,ge,gi,fire,I,records)


#=function IFNF{Vector{Float32}}(;N::Integer,PT<:Vector{Float32})
    v::PT = param.Vr .* ones(N) 
    ge::PT = zeros(N)
    gi::PT= zeros(N)
    fire::PT{Bool} = zeros(Bool, N)
    I::PT = zeros(N)
    records::Dict = Dict()
    new(N,v,ge,gi,fire,I,records)
end

function IFNF{CuArray}(;N::Integer,I<:CuArray,PT<:CuArray)
    v::PT = param.Vr .* ones(N) 
    ge::PT = zeros(N)
    gi::PT = zeros(N)
    fire::PT{Bool} = zeros(Bool, N)
    records::Dict = Dict()
    new(N,v,ge,gi,fire,I,records)
end
function IFNF{CuArray}(;N::Integer,PT<:CuArray)
    v::PT = param.Vr .* ones(N) 
    ge::PT = zeros(N)
    gi::PT= zeros(N)
    fire::PT{Bool} = zeros(Bool, N)
    I::PT = zeros(N)
    records::Dict = Dict()
    new(N,v,ge,gi,fire,I,records)
end
=#
#struct LIF{VT<:Real, IT<:Integer} <: AbstractCell
#LIF(τm::Real, vreset::Real, R::Real = 1.0) = LIF{Float32, Int}(vreset, 0, 0, τm, vreset, R)
        


#MyType(v::Real) = ...
#function MyType{T}(v::Vector{T})  # parametric type
#   ....
#end


#