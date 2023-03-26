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

struct PlasticSpikingSynapse{T<:AbstractArray{Float32},S<:AbstractArray{Int32},Q<:AbstractArray{Bool}}
    param::SpikingSynapseParameter # = SpikingSynapseParameter()
    rowptr::T # row pointer of sparse W
    colptr::S  # column pointer of sparse W
    I::S      # postsynaptic index of W
    J::S    # presynaptic index of W
    index::S  # index mapping: W[index[i]] = Wt[i], Wt = sparse(dense(W)')
    W::T  # synaptic weight
    tpre::T # = zero(W) # presynaptic spiking time
    tpost::T # = zero(W) # postsynaptic spiking time
    Apre::T# = zero(W) # presynaptic trace
    Apost::T# = zero(W) # postsynaptic trace
    fireI::Q # postsynaptic firing
    fireJ::Q # presynaptic firing
    g::T # postsynaptic conductance
    records::Dict # = Dict()

    function PlasticSpikingSynapse(pre::SpikingNeuralNetworks.IFNF, post::SpikingNeuralNetworks.IFNF,sim_type,rowptr, colptr, I, J, index, w)
        param::SpikingSynapseParameter = SpikingSynapseParameter()
        #w[diagind(w)] .= 0.0
        tpre::VFT = zero(W) # presynaptic spiking time
        tpost::VFT = zero(W) # postsynaptic spiking time
        Apre::VFT = zero(W) # presynaptic trace
        Apost::VFT = zero(W) # postsynaptic trace
        fireI, fireJ = post.fire, pre.fire
        records::Dict  = Dict()
        if sim_type == "CUDA"
            g = CuArray{Float32}(CUDA.ones(pre.N)*sign.(minimum(w[:,1])))    
            new{CuArray{Float32},CuArray{Int32},CuArray{Bool}}(param,rowptr,colptr,I,J,index,W,tpre,tpost,Apre,Apost,fireI,fireJ,g,records)
        elseif sim_type == "CPU"
            g = Vector{Float32}(ones(pre.N))*sign.(minimum(w[:,1]))
            new{Vector{Float32},Vector{Int32},Vector{Bool}}(param,rowptr,colptr,I,J,index,W,tpre,tpost,Apre,Apost,fireI,fireJ,g,records)
        end
        
        
    end

    function PlasticSpikingSynapse(pre::SpikingNeuralNetworks.IFNF, post::SpikingNeuralNetworks.IFNF,sim_type; σ = 0.0, p = 0.0)
        w = σ * sprand(post.N, pre.N, p) 
        #w[diagind(w)] .= 0.0
        rowptr, colptr, I, J, index, W = dsparse(w)
        fireI, fireJ = post.fire, pre.fire
        param::SpikingSynapseParameter = SpikingSynapseParameter()
        tpre::VFT = zero(W) # presynaptic spiking time
        tpost::VFT = zero(W) # postsynaptic spiking time
        Apre::VFT = zero(W) # presynaptic trace
        Apost::VFT = zero(W) # postsynaptic trace
        records::Dict  = Dict()
        if sim_type == "CUDA"
            g = CuArray{Float32}(CUDA.ones(pre.N)*sign.(minimum(w[:,1])))    
            SpikingSynapse{CuArray{Float32},CuArray{Int32},CuArray{Bool}}(param,rowptr,colptr,I,J,index,W,tpre,tpost,Apre,Apost,fireI,fireJ,g,records)
        elseif sim_type == "CPU"
            g = Vector{Float32}(ones(pre.N))*sign.(minimum(w[:,1]))
            SpikingSynapse{Vector{Float32},Vector{Int32},Vector{Bool}}(param,rowptr,colptr,I,J,index,W,tpre,tpost,Apre,Apost,fireI,fireJ,g,records)
        end
    end

end


function plasticity!(c::PlasticSpikingSynapse, param::SpikingSynapseParameter, dt::Float32, t::Float32)
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