using SparseArrays
using StaticArrays
using ProgressMeter
using LinearAlgebra

"""
This file contains a function stack that creates a network with Potjans and Diesmann wiring likeness in Julia using SpikingNeuralNetworks.jl to simulate
electrical neural network dynamics.
This code draws heavily on the PyNN OSB Potjans implementation code found here:
https://github.com/OpenSourceBrain/PotjansDiesmann2014/blob/master/PyNN/network_params.py#L139-L146
However the translation of this Python PyNN code into performant and scalable Julia was not trivial.
Hard coded Potjans parameters follow.
and then the function outputs adapted Potjans parameters.
"""
function potjans_params(ccu)
    # a cummulative cell count
    #cumulative = Dict{String, Vector{Int64}}()  
    layer_names = @SVector ["23E","23I","4E","4I","5E", "5I", "6E", "6I"] 
    # Probabilities for >=1 connection between neurons in the given populations. 
    # The first index is for the target population; the second for the source population
    #             2/3e      2/3i    4e      4i      5e      5i      6e      6i
    conn_probs = @SMatrix [0.1009  0.1689 0.0437 0.0818 0.0323 0.0     0.0076 0.    
                                    0.1346   0.1371 0.0316 0.0515 0.0755 0.     0.0042 0.    
                                    0.0077   0.0059 0.0497 0.135  0.0067 0.0003 0.0453 0.    
                                    0.0691   0.0029 0.0794 0.1597 0.0033 0.     0.1057 0.    
                                    0.1004   0.0622 0.0505 0.0057 0.0831 0.3726 0.0204 0.    
                                    0.0548   0.0269 0.0257 0.0022 0.06   0.3158 0.0086 0.    
                                    0.0156   0.0066 0.0211 0.0166 0.0572 0.0197 0.0396 0.2252
                                    0.0364   0.001  0.0034 0.0005 0.0277 0.008  0.0658 0.1443 ]

    # hard coded network wiring parameters are manipulated below:

    syn_pol = []
    for (i,syn) in enumerate(layer_names)
        if occursin("E",syn) 
            push!(syn_pol,true)
        else
            push!(syn_pol,false)
        end
    end
    syn_pol = syn_pol # synaptic polarity vector.
    return (conn_probs,syn_pol)
end


"""
Auxillary method, NB, this acts like the connectome constructor, so change function name to something more meaningful, like construct PotjanAndDiesmon
A mechanism for scaling cell population sizes to suite hardware constraints.
While Int64 might seem excessive when cell counts are between 1million to a billion Int64 is required.
Only dealing with positive count entities so Usigned is fine.
"""
function potjans_constructor(scale::Float64)
	ccu = Dict{String, UInt32}("23E"=>20683,
		    "4E"=>21915, 
		    "5E"=>4850, 
		    "6E"=>14395, 
		    "6I"=>2948, 
		    "23I"=>5834,
		    "5I"=>1065,
		    "4I"=>5479)
	ccu = Dict{String, UInt32}((k,ceil(Int64,v*scale)) for (k,v) in pairs(ccu))
    v_old=1
    K = length(keys(ccu))
    cum_array = []# Vector{Array{UInt32}}(undef,K)
    #for i in 1:K
    #    cum_array[i] = Array{UInt32}([])
    #end

    for (k,v) in pairs(ccu)
        ## update the cummulative cell count
        push!(cum_array,v_old:v+v_old)
        v_old=v+v_old

    end    

    cum_array = SVector{8,Array{UInt32}}(cum_array) # cumulative population counts array.
	Ncells = UInt64(sum([i for i in values(ccu)])+1)
	Ne = UInt64(sum([ccu["23E"],ccu["4E"],ccu["5E"],ccu["6E"]]))
    Ni = UInt64(Ncells - Ne)
    (Ncells, Ne, Ni, ccu, cum_array)
end
"""
The entry point to building the whole Potjans model in SNN.jl
Also some of the current density parameters needed to adjust synaptic gain initial values.
Some of the following calculations and parameters are borrowed from this repository:
https://github.com/SpikingNetwork/TrainSpikingNet.jl/blob/master/src/param.jl
"""
function potjans_layer(scale::Float64)
    (Ncells, Ne, Ni, ccu, cum_array)= potjans_constructor(scale)    
    (conn_probs,syn_pol) = potjans_params(ccu)    
 
    pree = 0.1
    K = round(Int, Ne*pree)
    sqrtK = sqrt(K)
    g = 1.0
    tau_meme = 10   # (ms)
    je = 2.0 / sqrtK * tau_meme * g
    ji = 2.0 / sqrtK * tau_meme * g 
    jee = 0.15je 
    jei = je 
    jie = -0.75ji 
    jii = -ji
    g_strengths = Vector{Float32}([jee,jie,jei,jii])
    Lxx = spzeros(Float32, (Ncells, Ncells))
    (jee,_,jei,_) = g_strengths 
    # Relative inhibitory synaptic weight
    wig = Float32(-20*4.5)
    build_matrix_prot!(jee,jei,wig,Lxx,cum_array,conn_probs,syn_pol,g_strengths)

    
end
export potjans_layer

#"""
#Build the matrix from the Potjans parameterpotjans_layers.
#"""
#function potjans_weights(args)
    #Ncells, g_strengths, ccu, scale = args
#   
#end


"""
This function contains synapse selection logic seperated from iteration logic for readability only.
Used inside the nested iterator inside build_matrix.
Ideally iteration could flatten to support the readability of subsequent code.
"""
#                      (jee,jei,wig,Lxx,cumvalues,conn_probs,UInt32(Ncells),syn_pol,g_strengths)
        #build_matrix_prot!(jee,jei,wig,Lxx,cumvalues,conn_probs,UInt32(Ncells),syn_pol,g_strengths)
function build_matrix_prot!(jee::Float32,jei::Float32,wig::Float32,Lxx::SparseMatrixCSC{Float32, Int64},cum_array::SVector{8, Array{UInt32}}, conn_probs::StaticArraysCore.SMatrix{8, 8, Float64, 64}, syn_pol, g_strengths::Vector{Float32})
    # excitatory weights.
    @inbounds @showprogress for (i,v) in enumerate(cum_array)

        @inbounds for (j,v1) in enumerate(cum_array)

            @inbounds for src in v
                @inbounds for tgt in v1

                    if src!=tgt
                        prob = conn_probs[i][j]


                        if rand()<prob

                            syn1 = syn_pol[j]
                            syn0 = syn_pol[i]
                
                            if syn0==true

                                if syn1==true
                                    #Lxx[src,tgt] = jee
                                    setindex!(Lxx,jee, src,tgt)
                                elseif syn1==true# meaning if the same as a logic: Inhibitory post synapse  is true                   
                                    setindex!(Lxx,jei, src,tgt)
                                    #Lxx[src,tgt] = jei

                                end
                            elseif syn0==false     
                                #Lxx[src,tgt] = jei
                                #println("gets here a")
                                #Lxx[src,tgt] = wig

                                #if syn1 

                                setindex!(Lxx,wig, src,tgt)
                                #elseif syn1
                                #    Lxx[src,tgt] = wig

                                    #setindex!(Lxx,wig, src,tgt)
                                #end
                            end 
                        end
                    end
                end
            end
            #display(Lxx)            
        end
    end
    #Lxx[diagind(Lxx)] .= 0.0

    ## Note to self function return annotations help.
    Lxx
end


#=
This function is now depreciated, and it was only aspirational at best.
function make_proj(xx,pop)
    rowptr, colptr, I, J, index, W = dsparse(xx)
    fireI, fireJ = pop.fire, pop.fire
    g = getfield(pop, :ge)
    SpikingSynapse(W,pre, post, sym)
    syn = SpikingSynapse(rowptr, colptr, I, J, index, W, fireI, fireJ, g)
    return syn
end
Similar to the methods above except that cells and synapses are instantiated in place to cut down on code.
Note this method is also asperational and depriciated.
function build_neurons_connections(Lee::SparseMatrixCSC{Float32, UInt64},Lei::SparseMatrixCSC{Float32, Int64},Lie::SparseMatrixCSC{Float32, Int64},Lii::SparseMatrixCSC{Float32, Int64},cumvalues, Ncells::Int32,syn_pol::StaticArraysCore.SVector{8, Int64})
    cntet=[]
    cntit=[]
    weights=[]
    cnte = 0
    cnti = 0
    
    @inbounds @showprogress for (i,v) in enumerate(cumvalues)
        @inbounds for (j,v1) in enumerate(cumvalues)
            @inbounds for src in v
                @inbounds for tgt in v1
                    if src!=tgt
                        prob = conn_probs[i,j]
                        
                        if rand()<prob
                            syn1 = syn_pol[j]
                            syn0 = syn_pol[i]
                            if syn0==1
			        cnte+=1 
                                if syn1==1
                                    push!(cntet,tgt)
                                    push!(weights,jee)
                                    setindex!(Lee,jee, src,tgt)
                                elseif syn1==0# meaning if the same as a logic: Inhibitory post synapse  is true                   
                                    push!(cntet,src)          
                                    setindex!(Lei,jei, src,tgt)
                                    push!(weights,jei)
                                end
                            elseif syn0==0         
			        cnti+=1 
				if syn1==1 
                                    push!(cntit,tgt)
                                    push!(weights,jie)
        
                                    setindex!(Lie,wig, src,tgt)
                                elseif syn1==0pop_size
                                    push!(cntit,tgt)
                                    push!(weights,jii)
                                    cnti+=1
                                    setindex!(Lii,wig, src,tgt)
                                end
                            end 
                        end
                    end
                end
            end
        end
        EE_ = Lee+Lei 
        II_ = Lii+Lie
        cntepopsize=0
        cntipopsize=0
        for (ind,row) in enumerate(eachcol(EE_'))
            if sum(row[:])!=0
                cntepopsize+=1
            end
        end
        for row in enumerate(eachcol(II_'))
            if sum(row[:])!=0
                cntipopsize+=1
            end            
        end
    end
    symtype = Vector{Float32}(zeros(cntepopsize))
    post_synaptic_targets = Array{Array{UInt64}}(undef,pop_size)
    for i in 1:pop_size
        post_synaptic_targets[i] = Array{UInt64}([])
    end
    E_ = SNN.IFNF(cntepopsize,sim_type,post_synaptic_targets,weights)
    I_ = SNN.IFNF(cntipopsize,sim_type,post_synaptic_targets,weights)
    (E_,I_)
end
=#
