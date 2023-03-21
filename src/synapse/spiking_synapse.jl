
import Plots.heatmap
using Plots
using CUDA
using SparseArrays
CUDA.allowscalar(false)



"""
[Spking Synapse](https://brian2.readthedocs.io/en/2.0b4/resources/tutorials/2-intro-to-brian-synapses.html)
"""
SpikingSynapse


struct SpikingSynapse{T,S,Q}
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


    
    function SpikingSynapse(pre::SpikingNeuralNetworks.IFNF, post::SpikingNeuralNetworks.IFNF,sim_type::Array,rowptr, colptr, I, J, index, w)
        g = ones(eltype=sim_type,pre.N)*sign.(minimum(w[:,1]))
        SpikingSynapse(rowptr,colptr,I,J,index,w,g,pre,post)
    end

    
    function SpikingSynapse(pre::SpikingNeuralNetworks.IFNF, post::SpikingNeuralNetworks.IFNF,sim_type::Any; σ = 0.0, p = 0.0)
        w = σ * sprand(post.N, pre.N, p) 
        w[diagind(w)] .= 0.0
        rowptr, colptr, I, J, index,V = dsparse(w,sim_type)
        g::typeof(sim_type) = (w[:]).*sign.(minimum(w[:,1]))   
        V::typeof(sim_type) = convert(typeof(sim_type),V)
        SpikingSynapse(rowptr,colptr,I,J,index,V,g,pre,post)
    end
    #=
    function SpikingSynapse(pre::SpikingNeuralNetworks.IFNF, post::SpikingNeuralNetworks.IFNF,sim_type::Array; σ = 0.0, p = 0.0)
        w = σ * sprand(post.N, pre.N, p) 
        w[diagind(w)] .= 0.0
        rowptr, colptr, I, J, index,V = dsparse(w,sim_type)
        g::typeof(sim_type) = (w[:]).*sign.(minimum(w[:,1]))   
        V::typeof(sim_type) = convert(typeof(sim_type),V)
        SpikingSynapse(rowptr,colptr,I,J,index,V,g,pre,post)
    end
    =#

end


"""
Boost synaptic conductances according to weight values.

"""
#forward!(::Vector{Int32}, ::Vector{Int32}, ::Vector{Float16}, ::Vector{Bool}, ::Vector{Bool}, ::Vector{Float16})

function forward!(colptr::Vector{<:Real}, I, W, fireI::Vector{Bool},fireJ::Vector{Bool},g::Vector)
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
function forward!(colptr::CuArray, I::CuArray, W::CuArray, fireI::CuArray{Bool},fireJ::CuArray{Bool}, g::CuArray)

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
