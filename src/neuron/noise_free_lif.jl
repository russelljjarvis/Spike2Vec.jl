
abstract type AbstractIFNF end


AA=AbstractArray
struct IFNF <: AbstractIFNF
    N::Integer
    pop_indexs::Integer
    v::AA 
    ge::AbstractArray
    gi::AbstractArray
    fire::AbstractArray{Bool}
    I::AbstractArray
    records::Dict
    function IFNF(N::Integer)
        PT=Vector{Float32}
        N::Integer = N
        v::PT = zeros(N) 
        ge::PT = zeros(N)
        gi::PT = zeros(N)
        I::PT = zeros(N)
        fire::PT{Bool} = zeros(Bool, N)
        records::Dict = Dict()
        pop_indexs::Integer = 1
        new(N,pop_indexs,v,ge,gi,fire,I,records)
    end 
    
    
    function IFNF(N::Integer)
        PT=CuArray{Float32}
        N::Integer = N
        v::PT = zeros(N) 
        ge::PT = zeros(N)
        gi::PT = zeros(N)
        I::PT = zeros(N)
        fire::CuArray{Bool} = zeros(Bool, N)
        records::Dict = Dict()
        pop_indexs::Integer = 1
        new(N,pop_indexs,v,ge,gi,fire,I,records)
    end 

    function IFNF(;N::Integer,I::CuArray{Float32},pop_indexs::Integer=1)
        PT=CuArray{Float32}
        N::Integer = N
        v::PT = zeros(N) 
        ge::PT = zeros(N)
        gi::PT = zeros(N)
        fire::CuArray{Bool} = zeros(Bool, N)
        records::Dict = Dict()
        #pop_indexs::Integer = pop_indexs
        new(N,pop_indexs,v,ge,gi,fire,I,records)
    end 
    function IFNF(;N::Integer,I::Vector{Float32})
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
    
    
end    

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

IFNF(N::Integer) = IFNF{Vector{Float32}}(N,pop_indexs,v,ge,gi,fire,I,records)

IFNF(N::Integer) = IFNF{CuArray{Float32}}(N,pop_indexs,v,ge,gi,fire,I,records)


IFNF(;N::Integer,I::CuArray{Float32}) = IFNF{CuArray{Float32}}(N,pop_indexs,v,ge,gi,fire,I,records)


IFNF(;N::Integer,I::Vector{Float32}) = IFNF{Vector{Float32}}(N,pop_indexs,v,ge,gi,fire,I,records)
IFNF(;N::Integer,I::CuArray{Float32},pop_indexs::Integer) = IFNF{CuArray{Float32}}(N,pop_indexs,v,ge,gi,fire,I,records)


#struct LIF{VT<:Real, IT<:Integer} <: AbstractCell
#LIF(τm::Real, vreset::Real, R::Real = 1.0) = LIF{Float32, Int}(vreset, 0, 0, τm, vreset, R)
        


#MyType(v::Real) = ...
#function MyType{T}(v::Vector{T})  # parametric type
#   ....
#end


#IFNF(;N::Int64,param::SpikingNeuralNetworks.IFParameter)




"""
    [Integrate-And-Fire Neuron](https://neuronaldynamics.epfl.ch/online/Ch1.S3.html)
"""
IF

function integrate!(p::IFNF, dt::Float32)
    @unpack N, v, ge, gi, fire, I = p
    integrate!(N, v, dt, ge, gi, fire, I)

end

#integrate!(::Int64, ::CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}, ::CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}, ::CuArray{Float32, 1, CUDA.Mem.DeviceBuffer}, ::CuArray{Bool, 1, CUDA.Mem.DeviceBuffer}, ::CuArray{Float32, 1, CUDA.Mem.DeviceBuffer})

function integrate!(N::Int64,v::CuArray{Float32} ,dt::Float32,ge::CuVector{Float32},gi::CuVector{Float32},fire::CuArray{Bool},I::CuVector{Float32})
    kernel = @cuda launch=false lif_kernel(N, v, ge, gi, fire,I,dt)
    config = launch_configuration(kernel.fun)
    xthreads = min(32, N)
    xblocks = min(config.blocks, cld(N, xthreads))
    kernel(N, v, ge, gi, fire, I,dt;threads=(xthreads), blocks=(xblocks<<2))

end


function integrate!(N::Float32,v::Vector{Float32},dt::Float32,ge::Vector{Float32},gi::Vector{Float32},fire::Vector{Bool},I::Vector{Float32})

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

function lif_kernel(N, v, ge, gi, fire, I,dt)
    τm, τe, τi, Vt, Vr, El, gL = (100.0,5.0,10.0,0.2,0.0,-60.0,10.0)
    R = 1.75
    i0 = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    stride = blockDim().x
    @inbounds for i=i0:stride:N
        X = ge[i] + gi[i]
        u = I[i]
        v[i] = v[i] * exp(-dt /τm) + Vr +X+u        
        v[i] += (I[i]+X) * (R/ τm)
        ge[i] += dt * -ge[i] / τe
        gi[i] += dt * -gi[i] / τi
        fire[i] = v[i] > Vt
        v[i] = ifelse(fire[i], Vr, v[i])
    end
    nothing
end



Base.show(io::IO, ::MIME"text/plain") =
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
