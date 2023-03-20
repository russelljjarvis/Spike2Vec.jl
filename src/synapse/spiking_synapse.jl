
import Plots.heatmap
using Plots
using CUDA
using SparseArrays
CUDA.allowscalar(false)
FT=Float32



"""
[Spking Synapse](https://brian2.readthedocs.io/en/2.0b4/resources/tutorials/2-intro-to-brian-synapses.html)
"""
SpikingSynapse


struct SpikingSynapse{T<:AbstractArray{Float32},S<:AbstractArray{Int32},Q<:AbstractArray{Bool}}
    rowptr::T # row pointer of sparse W
    colptr::S  # column pointer of sparse W
    I::S      # postsynaptic index of W
    J::S    # presynaptic index of W
    index::S  # index mapping: W[index[i]] = Wt[i], Wt = sparse(dense(W)')
    W::T  # synaptic weight
    fireI::Q # postsynaptic firing
    fireJ::Q # presynaptic firing
    g::T # postsynaptic conductance
    records::Dict # = Dict()


    function SpikingSynapse(rowptr, colptr, I, J, index, w,g,pre,post)
        fireI, fireJ = post.fire, pre.fire
        records::Dict  = Dict()
        new{typeof(w),typeof(colptr),typeof(fireJ)}(rowptr,colptr,I,J,index,w,fireI,fireJ,g,records)
    end


    function SpikingSynapse(pre::SpikingNeuralNetworks.IFNF, post::SpikingNeuralNetworks.IFNF,sim_type::CuArray{Float32},rowptr, colptr, I, J, index, w)
        g = CuArray{Float32}(w[:]).*sign.(minimum(w[:,1]))   
        SpikingSynapse(rowptr,colptr,I,J,index,w,g,pre,post)
    end

    
    function SpikingSynapse(pre::SpikingNeuralNetworks.IFNF, post::SpikingNeuralNetworks.IFNF,sim_type::Vector{Float32},rowptr, colptr, I, J, index, w)
        g = Vector{Float32}(ones(pre.N))*sign.(minimum(w[:,1]))
        SpikingSynapse(rowptr,colptr,I,J,index,w,g,pre,post)
    end

    
    function SpikingSynapse(pre::SpikingNeuralNetworks.IFNF, post::SpikingNeuralNetworks.IFNF,sim_type::CuArray{Float32}; σ = 0.0, p = 0.0)
        w = σ * sprand(post.N, pre.N, p) 
        w[diagind(w)] .= 0.0
        rowptr, colptr, I, J, index, w_ = dsparse(w)
        g = typeof(sim_type)(w[:]).*sign.(minimum(w[:,1]))   
        SpikingSynapse(rowptr,colptr,I,J,index,w_,g,pre,post)
    end
end


"""
Boost synaptic conductances according to weight values.

"""
function forward!(colptr::Vector{Int32}, I, W, fireI::Vector{Bool},fireJ::Vector{Bool},g::Vector{Float32})
    @inbounds for j in 1:(length(colptr) - 1)
        if fireJ[j]
            @inbounds for s in colptr[j]:(colptr[j+1] - 1)
                g[I[s]] += W[s]
            end
        end
    end
end

function forward!(c::SpikingSynapse)
    @unpack colptr, I, W, fireI,fireJ, g = c
    forward!(colptr, I, W, fireI,fireJ, g)
end


#function forward!(c::Array)
#    @unpack colptr, I, W, fireI,fireJ, g = c
#    forward!(colptr, I, W, fireI,fireJ, g)
#end


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

function forward_all_kernel!(N, v, ge, gi, fire, u,dt,colptr, fireJ,g,I,W)
    index = threadIdx().x    # this example only requires linear indexing, so just use `x`
    stride = blockDim().x
    τm, τe, τi, Vt, Vr, El, gL = (100.0,5.0,10.0,0.2,0.0,-60.0,10.0)
    R = 1.75
    i0 = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    stride = blockDim().x
    @inbounds for i=i0:stride:N
        X = ge[i] + gi[i]
        u = u[i]
        v[i] = v[i] * exp(-dt /τm) + Vr +X+u        
        v[i] += (I[i]+X) * (R/ τm)
        ge[i] += dt * -ge[i] / τe
        gi[i] += dt * -gi[i] / τi
        fireJ[i] = v[i] > Vt
        v[i] = ifelse(fire[i], Vr, v[i])
    end

    for i = index:stride:(length(colptr)-1)
        if fireJ[i]
            for x in colptr[i]:(colptr[i+1]-1)
                g[I[x]] += W[x]                

            end                
        end
    end

    return nothing
end


function forward!(colptr::CuArray{Int32}, I::CuArray{Int32}, W::CuArray{Float32}, fireI::CuArray{Bool},fireJ::CuArray{Bool}, g::CuArray{Float32})
    ###
    #CUDA.Const(fireJ::CuDeviceArray{Bool})#, CUDA.Mem.DeviceBuffer})
    #CUDA.Const(colptr::CuDeviceArray{Float32})#, CUDA.Mem.DeviceBuffer})
    ###
    kernel = @cuda launch=false syn_kernel!(colptr, fireJ,g,I,W)
    config = launch_configuration(kernel.fun)
    xthreads = min(32, length(colptr))
    xblocks = min(config.blocks, cld(length(colptr), xthreads))

    kernel(colptr, fireJ,g,I,W;threads=(xthreads), blocks=(xblocks<<2))

end
#using Cthulhu
#@code_typed(err; interactive = true)


#Hint: catch this exception as `err` and call `code_typed(err; interactive = true)` to introspect the erronous code with Cthulhu.jl


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