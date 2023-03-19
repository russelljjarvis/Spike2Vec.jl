
import Plots.heatmap
using Plots
using CUDA
using SparseArrays
CUDA.allowscalar(false)
FT=Float32
struct SpikingSynapseParameter
    τpre::FT # = 20ms
    τpost::FT#  = 20ms
    Wmax::FT#  = 0.01
    ΔApre::FT#  = 0.01 * Wmax
    ΔApost::FT#  = -ΔApre * τpre / τpost * 1.05
    function SpikingSynapseParameter()
        τpre::FT  = 20ms
        τpost::FT  = 20ms
        Wmax::FT  = 0.01
        ΔApre::FT  = 0.01 * Wmax
        ΔApost::FT  = -ΔApre * τpre / τpost * 1.05
        new(20ms,20ms,Wmax,ΔApre,ΔApost)
    end
end


"""
[Spking Synapse](https://brian2.readthedocs.io/en/2.0b4/resources/tutorials/2-intro-to-brian-synapses.html)
"""
SpikingSynapse



VIT=Vector{Int32}
VFT=Vector{Float32}
VBT=Vector{Bool}
CUB=CuArray{Bool}
CUV=CuArray{Float32}
mutable struct SpikingSynapse
    param::SpikingSynapseParameter # = SpikingSynapseParameter()
    rowptr::CUV # row pointer of sparse W
    colptr::CUV # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    J::VIT      # presynaptic index of W
    index::VIT  # index mapping: W[index[i]] = Wt[i], Wt = sparse(dense(W)')
    W::VFT  # synaptic weight
    tpre::VFT # = zero(W) # presynaptic spiking time
    tpost::VFT # = zero(W) # postsynaptic spiking time
    Apre::VFT # = zero(W) # presynaptic trace
    Apost::VFT # = zero(W) # postsynaptic trace
    fireI::CUB # postsynaptic firing
    fireJ::CUB # presynaptic firing
    g::CUV # postsynaptic conductance
    records::Dict # = Dict()
    wx::SparseMatrixCSC # postsynaptic conductance

    #LiiSyn = SpikingSynapse(;@symdict(rowptr, colptr, I, J, index, W, fireI, fireJ, g)..., kwargs...)
    function SpikingSynapse(rowptr, colptr, I, J, index, W, fireI, fireJ, g)
        param::SpikingSynapseParameter = SpikingSynapseParameter()
        #rowptr, colptr, I, J, index, W = dsparse(w)
        tpre::VFT = zero(W) # presynaptic spiking time
        tpost::VFT = zero(W) # postsynaptic spiking time
        Apre::VFT = zero(W) # presynaptic trace
        Apost::VFT = zero(W) # postsynaptic trace

        #@show(size(w))
        #fireI, fireJ = post.fire, pre.fire
        #g = getfield(post, sym)
        records::Dict  = Dict()
        new(param,rowptr,colptr,I,J,index,W,tpre,tpost,Apre,Apost,fireI,fireJ,g,records)
    end

    #EE = SNN.SpikingSynapse(E, E, :ge; σ = 60*0.27/10, p = 0.02)
    function SpikingSynapse(rowptr, colptr, I, J, index, w)#, fireI, fireJ, g)#::Int64,param::SpikingNeuralNetworks.IFParameter)
        param::SpikingSynapseParameter = SpikingSynapseParameter()
        w[diagind(w)] .= 0.0

        rowptr, colptr, I, J, index, W = dsparse(w.nzval)

        tpre::VFT = zero(W) # presynaptic spiking time
        tpost::VFT = zero(W) # postsynaptic spiking time
        Apre::VFT = zero(W) # presynaptic trace
        Apost::VFT = zero(W) # postsynaptic trace

        fireI, fireJ = post.fire, pre.fire
        g = getfield(post, sym)
        records::Dict  = Dict()
        new(param,rowptr,colptr,I,J,index,W,tpre,tpost,Apre,Apost,fireI,fireJ,g,records)
    end




    #EE = SNN.SpikingSynapse(E, E, :ge; σ = 60*0.27/10, p = 0.02)

    #SpikingNeuralNetworks.SpikingSynapse(::Dict{Any, Any}, ::Dict{Any, Any}, ::Symbol; σ=16.200000000000003, p=0.02)
    function SpikingSynapse(pre::Dict{Any, Any}, post::Dict{Any, Any}, sym::Symbol; σ = 0.0, p = 0.0)#, kwargs...)
        #@show(typeof(post))
        w = σ * sprand(post.N, pre.N, p) 
        #w .- diag(w)
        w[diagind(w)] .= 0.0
        rowptr, colptr, I, J, index, W = dsparse(w)
        fireI, fireJ = post.fire, pre.fire


        
        g = getfield(post, sym)

        param::SpikingSynapseParameter = SpikingSynapseParameter()
        rowptr, colptr, I, J, index, W = dsparse(w)
        tpre::VFT = zero(W) # presynaptic spiking time
        tpost::VFT = zero(W) # postsynaptic spiking time
        Apre::VFT = zero(W) # presynaptic trace
        Apost::VFT = zero(W) # postsynaptic trace
        records::Dict  = Dict()
        new(param,rowptr,colptr,I,J,index,W,tpre,tpost,Apre,Apost,fireI,fireJ,g,records)

    end

    function SpikingSynapse(pre::SpikingNeuralNetworks.IFNF, post::SpikingNeuralNetworks.IFNF, sym_::Symbol; σ = 0.0, p = 0.0)#, kwargs...)
        w = σ * sprand(post.N, pre.N, p) 
        w[diagind(w)] .= 0.0
        wx = w
        rowptr, colptr, I, J, index, W = dsparse(w)
        fireI, fireJ = post.fire, pre.fire
        g::VFT = ones(pre.N)#w[:,1]
        param::SpikingSynapseParameter = SpikingSynapseParameter()
        rowptr, colptr, I, J, index, W = dsparse(w)
        tpre::VFT = zero(W) # presynaptic spiking time
        tpost::VFT = zero(W) # postsynaptic spiking time
        Apre::VFT = zero(W) # presynaptic trace
        Apost::VFT = zero(W) # postsynaptic trace
        records::Dict  = Dict()

        
        new(param,rowptr,colptr,I,J,index,W,tpre,tpost,Apre,Apost,fireI,fireJ,g,records,wx)

    end

end
#=

function forward!(colptr, I, W, fireI,fireJ, g,wx)
    @inbounds for j in 1:(length(colptr) - 1)
        if fireJ[j]
            @inbounds for s in colptr[j]:(colptr[j+1] - 1)
                g[I[s]] += W[s]

            end
        end
    end
end
function forward!(colptr, I, W, fireI::CuArray{Bool},fireJ::CuArray{Bool}, g::CuArray{Float32},wx)
    println("hit")
    @inbounds for j in 1:(length(colptr) - 1)
        if fireJ[j]
            @inbounds for s in colptr[j]:(colptr[j+1] - 1)
                g[I[s]] += W[s]

            end
        end
    end
end
=#
function forward!(c::SpikingSynapse)
    @unpack colptr, I, W, fireI,fireJ, g = c
    forward!(colptr, I, W, fireI,fireJ, g)
end

#=
function syn_kernel(colptr, I, W, fireJ, g)
    i0 = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    stride = blockDim().x
    #x = (blockIdx().x - 1) * blockDim().x + threadIdx().x 
    @inbounds for j=i0:stride:(length(colptr) - 1)
        if fireJ[j]
            @inbounds for s in colptr[j]:(colptr[j+1] - 1)
                g[I[s]] += W[s]
            end
        end
    end
    nothing
end
=#
#=
function syn_kernel!(colptr, I, W, fireJ, g)
    j = (blockIdx()) * (blockDim().x-1) + threadIdx().x
    @cuprintln(j)
    @cushow(fireJ)
    @inbounds if j <= length(colptr)
        if fireJ[j]
            @cushow(fireJ[j])
            s = colptr[j]:(colptr[j+1] - 1)
            g[I[s]] += W[s]
        end
    end
    return nothing
end
=#
function syn_kernel!(colptr, fireJ,g,I,W)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    for i = index:stride:(length(colptr)-1)
        if fireJ[i]
            for x in colptr[i]:(colptr[i+1]-1)
                g[I[x]] += W[x]

            end                
        end
    end
    return nothing
end


function forward!(colptr::CuArray{Float32}, I::Vector{Int32}, W::Vector{Float32}, fireI::CuArray{Bool},fireJ::CuArray{Bool}, g::CuArray{Float32})
    W = convert(CuArray{Float32},W)
    I = convert(CuArray{Int32},I)
    colptr = convert(CuArray{Int32},colptr)
    kernel = @cuda launch=false syn_kernel!(colptr, fireJ,g,I,W)
    config = launch_configuration(kernel.fun)
    xthreads = min(32, length(colptr))
    xblocks = min(config.blocks, cld(length(colptr), xthreads))
    kernel(colptr, fireJ,g,I,W;threads=(xthreads), blocks=(xblocks<<2))

end
#using Cthulhu
#@code_typed(err; interactive = true)


#Hint: catch this exception as `err` and call `code_typed(err; interactive = true)` to introspect the erronous code with Cthulhu.jl

function plasticity!(c::SpikingSynapse, param::SpikingSynapseParameter, dt::Float32, t::Float32)
    @unpack rowptr, colptr, I, J, index, W, tpre, tpost, Apre, Apost, fireI, fireJ, g = c
    @unpack τpre, τpost, Wmax, ΔApre, ΔApost = param
    @inbounds for j in 1:(length(colptr) - 1)
        if fireJ[j]
            for s in colptr[j]:(colptr[j+1] - 1)
                Apre[s] *= exp32(- (t - tpre[s]) / τpre)
                Apost[s] *= exp32(- (t - tpost[s]) / τpost)
                Apre[s] += ΔApre
                tpre[s] = t
                W[s] = clamp(W[s] + Apost[s], 0f0, Wmax)
            end
        end
    end
    @inbounds for i in 1:(length(rowptr) - 1)
        if fireI[i]
            for st in rowptr[i]:(rowptr[i+1] - 1)
                s = index[st]
                Apre[s] *= exp32(- (t - tpre[s]) / τpre)
                Apost[s] *= exp32(- (t - tpost[s]) / τpost)
                Apost[s] += ΔApost
                tpost[s] = t
                W[s] = clamp(W[s] + Apre[s], 0f0, Wmax)
            end
        end
    end
end
#plasticity16!(c::SpikingSynapse, param::SpikingSynapseParameter, dt::Float16, t::Float32) = plasticity!(c::SpikingSynapse, param::SpikingSynapseParameter, dt::Float32, t::Float32)

#=
struct EPSP{IT<:Integer, VT<:Real} <: AbstractSynapse
    spikes::CircularBuffer{VT}
    ϵ₀::VT
    τm::VT
    τs::VT
end

"""
    EPSP{IT, VT}(;ϵ₀::Real = 1, τm::Real = 1, τs::Real = 1, N = 100)
    EPSP(;ϵ₀::Real = 1, τm::Real = 1, τs::Real = 1, N = 100)
Create an EPSP synapse with amplitude `ϵ₀`, rise time `τs`, and fall time `τm`.
Specify `N` to adjust how many pre-synaptic spikes are remembered between post-synaptic spikes.
"""
EPSP{IT, VT}(;ϵ₀::Real = 1, τm::Real = 1, τs::Real = 1, N = 100) where {IT<:Integer, VT<:Real} =
    EPSP{IT, VT}(fill!(CircularBuffer{VT}(N), -Inf), ϵ₀, τm, τs)
EPSP(;ϵ₀::Real = 1, τm::Real = 1, τs::Real = 1, N = 100) = EPSP{Int, Float32}(ϵ₀ = ϵ₀, τm = τm, τs = τs, N = N)

"""
    excite!(synapse::EPSP, spike::Integer)
    excite!(synapses::AbstractArray{<:EPSP}, spike::Integer)
Excite `synapse` with a `spike` (`spike` == time step of spike).
"""
excite!(synapse::EPSP, spike::Integer) = (spike > 0) && push!(synapse.spikes, spike)
excite!(synapses::T, spike::Integer) where T<:AbstractArray{<:EPSP} = (spike > 0) && push!.(synapses.spikes, spike)
=#