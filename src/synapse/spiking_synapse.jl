
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

#@snn_kw mutable 

VIT=Vector{Int32}
VFT=Vector{Float32}
VBT=Vector{Bool}
struct SpikingSynapse
    param::SpikingSynapseParameter # = SpikingSynapseParameter()
    rowptr::VIT # row pointer of sparse W
    colptr::VIT # column pointer of sparse W
    I::VIT      # postsynaptic index of W
    J::VIT      # presynaptic index of W
    index::VIT  # index mapping: W[index[i]] = Wt[i], Wt = sparse(dense(W)')
    W::VFT  # synaptic weight
    tpre::VFT # = zero(W) # presynaptic spiking time
    tpost::VFT # = zero(W) # postsynaptic spiking time
    Apre::VFT # = zero(W) # presynaptic trace
    Apost::VFT # = zero(W) # postsynaptic trace
    fireI::VBT # postsynaptic firing
    fireJ::VBT # presynaptic firing
    g::VFT # postsynaptic conductance
    records::Dict # = Dict()
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
        rowptr, colptr, I, J, index, W = dsparse(w)
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

    function SpikingSynapse(pre, post, sym; σ = 0.0, p = 0.0)#, kwargs...)
        w = σ * sprand(post.N, pre.N, p)
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

end

"""
[Spking Synapse](https://brian2.readthedocs.io/en/2.0b4/resources/tutorials/2-intro-to-brian-synapses.html)
"""
SpikingSynapse


function SpikingSynapse(Lee::SparseMatrixCSC{Float32, Int64},Lei::SparseMatrixCSC{Float32, Int64},Lii::SparseMatrixCSC{Float32, Int64},Lie::SparseMatrixCSC{Float32, Int64}, kwargs...)
    inhib = Lii+Lie # total inhibitory population
    cnte=0
    exc = Lee+Lei # total excitatory population
    cnti=0
    # count neurons which are being pre-synapses to something

    for i in eachrow(inhib)
        if sum(i[:]) != 0.0
            cnti+=1
        end
    end


    # count neurons which are being pre-synapses to something
    for i in eachrow(exc)
        if sum(i[:]) != 0.0
            cnte+=1
        end
    end
    total = inhib+exc

    pop_size=length(total)
    cnt = length(total)
    Ipop = SNN.IFNF(;N = pop_size, param = SNN.IFParameter())
    Epop = SNN.IFNF(;N = pop_size, param = SNN.IFParameter())
    Noisey = SNN.IF(;N = cnte, param = SNN.IFParameter())

    σ, p = 60*0.27/40 , 0.005
    wnoise = σ * sprand(cnte, cnte, p)
    rowptr, colptr, I, J, index, W = dsparse(wnoise)
    fireI, fireJ = Noisey.fire, Epop.fire    
    g = getfield(Noisey, :ge)
    NoisyInputSyn = SpikingSynapse(rowptr, colptr, I, J, index, W, fireI, fireJ, g)

    
    σ, p = 60*0.27/40 , 0.005
    wnoise = σ * sprand(cnte, cnte, p)
    rowptr, colptr, I, J, index, W = dsparse(wnoise)
    fireI, fireJ = Noisey.fire, Ipop.fire    
    g = getfield(Noisey, :ge)
    NoisyInputSynInh = SpikingSynapse(rowptr, colptr, I, J, index, W, fireI, fireJ, g)


    @assert maximum(Lii) <= 0.0
    @assert maximum(Lei) >= 0.0
    @assert maximum(Lie) <= 0.0
    @assert maximum(Lee) >= 0.0

    rowptr, colptr, I, J, index, W = dsparse(Lee)
    fireI, fireJ = Epop.fire, Epop.fire
    g = getfield(Epop, :ge)
    LeeSyn = SpikingSynapse(rowptr, colptr, I, J, index, W, fireI, fireJ, g)#..., kwargs...)
    spy(LeeSyn)
    rowptr, colptr, I, J, index, W = dsparse(Lei)
    fireI, fireJ = Epop.fire, Ipop.fire
    g = getfield(Epop, :ge)    
    LeiSyn = SpikingSynapse(rowptr, colptr, I, J, index, W, fireI, fireJ, g)#..., kwargs...)
    #LeiSyn = SpikingSynapse(;@symdict(rowptr, colptr, I, J, index, W, fireI, fireJ, g)..., kwargs...)
    
    rowptr, colptr, I, J, index, W = dsparse(Lii)
    fireI, fireJ = Ipop.fire, Ipop.fire
    g = getfield(Ipop, :gi)
    LiiSyn = SpikingSynapse(rowptr, colptr, I, J, index, W, fireI, fireJ, g)

    #LiiSyn = SpikingSynapse(;@symdict(rowptr, colptr, I, J, index, W, fireI, fireJ, g)..., kwargs...)
    
    rowptr, colptr, I, J, index, W = dsparse(Lie)
    fireI, fireJ = Ipop.fire, Epop.fire
    g = getfield(Ipop, :gi)
    LieSyn = SpikingSynapse(rowptr, colptr, I, J, index, W, fireI, fireJ, g)
    (NoisyInputSynInh,NoisyInputSyn,LeeSyn,LeiSyn,LiiSyn,LieSyn,Epop,Ipop,Noisey)
end

function forward!(c::SpikingSynapse, param::SpikingSynapseParameter)
    @unpack colptr, I, W, fireJ, g = c
    @inbounds for j in 1:(length(colptr) - 1)
        if fireJ[j]
            for s in colptr[j]:(colptr[j+1] - 1)
                g[I[s]] += W[s]
            end
        end
    end
end

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
