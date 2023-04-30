using SparseArrays
using StaticArrays
using ProgressMeter
using LinearAlgebra
#using UnicodePlots

"""
This file contains a function stack that creates a network with Potjans and Diesmann wiring likeness in Julia using SpikingNeuralNetworks.jl to simulate
electrical neural network dynamics.
This code draws heavily on the PyNN OSB Potjans implementation code found here:
https://github.com/OpenSourceBrain/PotjansDiesmann2014/blob/master/PyNN/network_params.py#L139-L146
However the translation of this Python PyNN code into performant and scalable Julia was not trivial.

Hard coded Potjans parameters follow.
and then the function outputs adapted Potjans parameters.

TODO for self, considering PR differencing my old branch with this one, as I may have neglected to carry over other constructive code changes.
Old branch worth diffing.
https://github.com/RJsWorkatWSU/SpikingNeuralNetworks.jl/tree/potjans_model
"""
function potjans_params(ccu, scale=1.0::Float64)
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
    v_old=1
    #cum_array = Vector{Any}([])#Matrix{Any}(zeros((2,length(ccu))))

    K = length(ccu)
    cum_array = Vector{Array{UInt32}}(undef,K)
    for i in 1:K
        cum_array[i] = Array{UInt32}([])
    end
    for (k,v) in pairs(ccu)
        ## update the cummulative cell count
        #cumulative[k]=collect(v_old:v+v_old)
        #@show(length(collect(v_old:v+v_old)[:]))
        push!(cum_array,Vector{UInt32}(collect(v_old:v+v_old)[:]))
        v_old=v+v_old
    end    
    @show(typeof(cum_array))
    cum_array = SVector{16,Array{UInt32}}(cum_array)
    syn_pol = Vector{UInt32}(zeros(length(ccu)))
    for (i,(k,v)) in enumerate(pairs(ccu))
        if occursin("E",k) 
            syn_pol[i] = 1
        else
            syn_pol[i] = 0
        end
    end
    syn_pol = SVector{8,UInt32}(syn_pol)
    return (cum_array,ccu,layer_names,conn_probs,syn_pol)
end


"""
A mechanism for scaling cell population sizes to suite hardware constraints.
While Int64 might seem excessive when cell counts are between 1million to a billion Int64 is required.
Only dealing with positive count entities so Usigned is fine.
"""
function auxil_potjans_param(scale=1.0::Float32)

	ccu = Dict{String, UInt32}("23E"=>20683,
		    "4E"=>21915, 
		    "5E"=>4850, 
		    "6E"=>14395, 
		    "6I"=>2948, 
		    "23I"=>5834,
		    "5I"=>1065,
		    "4I"=>5479)
	ccu = Dict{String, UInt32}((k,ceil(Int64,v*scale)) for (k,v) in pairs(ccu))
	Ncells = UInt64(sum([i for i in values(ccu)])+1)
	Ne = UInt64(sum([ccu["23E"],ccu["4E"],ccu["5E"],ccu["6E"]]))
    Ni = UInt64(Ncells - Ne)
    Ncells, Ne, Ni, ccu
end

"""
The entry point to building the whole Potjans model in SNN.jl
Also some of the current density parameters needed to adjust synaptic gain initial values.
"""
function potjans_layer(scale)
    Ncells,Ne,Ni, ccu = auxil_potjans_param(scale)    
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
    genStaticWeights = (;Ncells,g_strengths,ccu,scale)
    potjans_weights(genStaticWeights),Ne,Ni,ccu
end
export potjans_layer
"""
This function contains synapse selection logic seperated from iteration logic for readability only.
Used inside the nested iterator inside build_matrix.
Ideally iteration could flatten to support the readability of subsequent code.
"""

#matching 
#build_matrix_prot!(::SparseMatrixCSC{Float32, Int64}, ::SparseMatrixCSC{Float32, Int64}, ::SparseMatrixCSC{Float32, Int64}, ::SparseMatrixCSC{Float32, Int64}, ::Vector{Any}, ::SMatrix{8, 8, Float64, 64}, ::Int32, ::SVector{8, UInt64}, ::Vector{Float64})
#Closest candidates are:
#build_matrix_prot!(::SparseMatrixCSC{Float32, Int64}, ::SparseMatrixCSC{Float32, Int64}, ::SparseMatrixCSC{Float32, Int64}, ::SparseMatrixCSC{Float32, Int64}, ::Any, ::SMatrix{8, 8, Float64, 64}, ::Int32, ::SVector{8, Int64}, ::Vector{Float64}) 
#build_matrix_prot!(::Float32, ::Float32, ::Float32, ::SparseMatrixCSC{Float32, Int64}, ::SVector{16, Array{UInt32}}, ::SMatrix{8, 8, Float64, 64}, ::UInt32, ::SVector{8, UInt32}, ::Vector{Float32})

function build_matrix_prot!(wig::Float32,jee::Float32,jei::Float32,Lxx::SparseMatrixCSC{Float32, Int64},cumvalues::SVector{16, Array{UInt32}}, conn_probs::StaticArraysCore.SMatrix{8, 8, Float64, 64}, Ncells::UInt32, syn_pol::StaticArraysCore.SVector{8, UInt32},g_strengths::Vector{Float32})
    # excitatory weights.
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
                                if syn1==1 
                                    setindex!(Lxx,jee, src,tgt)
                                elseif syn1==0# meaning if the same as a logic: Inhibitory post synapse  is true                   
                                    setindex!(Lxx,jei, src,tgt)
                                end
                            elseif syn0==0         
                                if syn1==1 
                                    setindex!(Lxx,wig, src,tgt)
                                elseif syn1==0
                                    setindex!(Lxx,wig, src,tgt)
                                end
                            end 
                        end
                    end
                end
            end            
        end
    end
    #Lxx = Lee+Lei+Lii+Lie
    Lxx[diagind(Lxx)] .= 0.0

    #display(Lxx)
    ## Note to self function return annotations help.
    #Lxx#,Lee,Lei,Lii,Lie
end

function make_proj(xx,pop)
    rowptr, colptr, I, J, index, W = dsparse(xx)
    fireI, fireJ = pop.fire, pop.fire
    g = getfield(pop, :ge)
    SpikingSynapse(W,pre, post, sym)
    syn = SpikingSynapse(rowptr, colptr, I, J, index, W, fireI, fireJ, g)
    return syn
end
"""
Similar to the methods above except that cells and synapses are instantiated in place to cut down on code
"""
function build_neurons_connections(Lee::SparseMatrixCSC{Float32, UInt64},Lei::SparseMatrixCSC{Float32, Int64},Lie::SparseMatrixCSC{Float32, Int64},Lii::SparseMatrixCSC{Float32, Int64},cumvalues, Ncells::Int32,syn_pol::StaticArraysCore.SVector{8, Int64})
    #cntees=[]
    #cnteis=[]
    #cnties=[]
    #cntiis=[]
    
    cntet=[]
    #cnteit=[]
    cntit=[]
    #cntiit=[]

    weights=[]
    #weightsi=[]
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
                                if syn1==1
                                    cnte+=1 
                                    #append!(cntees,src)
                                    push!(cntet,tgt)
                                    push!(weights,jee)

                                    setindex!(Lee,jee, src,tgt)
                                elseif syn1==0# meaning if the same as a logic: Inhibitory post synapse  is true                   
                                    push!(cntet,src)
                                    #append!(cnteit,tgt)
          
                                    cnte+=1 
                                    setindex!(Lei,jei, src,tgt)
                                    push!(weights,jei)

                                end
                            elseif syn0==0         
                                if syn1==1 
                                    cnti+=1 
                                    #append!(cnties,src)
                                    push!(cntit,tgt)
                                    push!(weights,jie)
        
                                    setindex!(Lie,wig, src,tgt)
                                elseif syn1==0pop_size
                                    #append!(cntiis,src)
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

    #u1 = Float32[10.0*abs(4.0*rand()) for i in 0:1ms:sim_duration]
    
   
    #E = SNN.IFNF(pop_size,sim_type,post_synaptic_targets)
    #I = SNN.IFNF(pop_size,sim_type,post_synaptic_targets)

    #(E,I)
    #potjans_pop = SNN.SpikingSynapse(E, E,sim_type; σ = 160*0.27/1, p = 0.025)
    (E_,I_)
end

"""
Build the matrix from the Potjans parameterpotjans_layers.
"""
function potjans_weights(args)
    Ncells, g_strengths, ccu, scale = args
    (cumvalues,_,_,conn_probs,syn_pol) = potjans_params(ccu,scale)    
    #cumvalues = values(cumulative)
    
    Lxx = spzeros(Float32, (Ncells, Ncells))
    #Lie = spzeros(Float32, (Ncells, Ncells))
    #Lei = spzeros(Float32, (Ncells, Ncells))
    #Lii = spzeros(Float32, (Ncells, Ncells))
    (jee,_,jei,_) = g_strengths 
    # Relative inhibitory synaptic weight
    wig = Float32(-20*4.5)

    #prealloc_size = sum(cumvalues)*sum(cumvalues)
    #dense_src = Vector{UInt32}(zeros(prealloc_size))
    #dense_tgt = Vector{UInt32}(zeros(prealloc_size))
    #dense_wgt = Vector{UInt32}(zeros(prealloc_size))
 
    return build_matrix_prot!(jee,jei,wig,Lxx,cumvalues,conn_probs,UInt32(Ncells),syn_pol,g_strengths)
    #(EE,EI,IE,II,synII,synIE,synEI,synEE) = build_neurons_connections(Lee,Lei,Lie,Lii,cumvalues, Ncells,syn_pol)
end




#=
Depreciated methods

function index_assignment!(item::NTuple{4, UInt64}, g_strengths::Vector{Float32}, lxx::SparseMatrixCSC{Float32, UInt64})#,lee::Vector{Vector{Tuple{Int64, UInt64}}},lie::Vector{Vector{Tuple{Int64, UInt64}}}, lii::Vector{Vector{Tuple{Int64, Int64}}}, lei::Vector{Vector{Tuple{Int64, Int64}}})
    # excitatory weights.
    (jee,_,jei,_) = g_strengths 
    # Relative inhibitory synaptic weight
    wig = -20*4.5
    (src,tgt,syn0,syn1) = item
    if syn0==1
        if syn1==1            
            setindex!(lxx,jee, src,tgt)
        elseif syn1==0# meaning if the same as a logic: Inhibitory post synapse  is true                   
            setindex!(lxx, jei, src,tgt)
        end
    elseif syn0==0# meaning if the same as a logic: Inhibitory post synapse  is true   
        if syn1==1
            setindex!(lxx, wig, src,tgt)
        elseif syn1==0# eaning meaning if the same as a logic: if occursin("I",k1)      is true               
            @assert syn1==0
            setindex!(lxx,wig, src,tgt)
            @assert syn1==0
        end
    end
end
function build_matrix(Ncells::Int32,just_iterator,g_strengths::Vector{Float64})
    Lee = Vector{Vector{Tuple{Int64, Int64}}}[]
    Lie = Vector{Vector{Tuple{Int64, Int64}}}[]
    Lii = Vector{Vector{Tuple{Int64, Int64}}}[]
    Lei = Vector{Vector{Tuple{Int64, Int64}}}[]
    @showprogress for i in just_iterator
        index_assignment!(i[:],g_strengths,Lxx)#,Lee,Lie,Lii,Lei)
    end
    Lee = Lxx[(i[1],i[2]) for i in Lee]
    Lie = Lxx[Lie[1,:],Lie[2,:]]
    Lii = Lxx[Lii[1,:],Lii[2,:]]
    Lei = Lxx[Lei[1,:],Lei[2,:]]
    Lexc = Lee+Lei
    Linh = Lie+Lii
    return Lee,Lie,Lei,Lii
end



"""
An optional container that is not yet utilized.
"""
#=
struct NetParameter 
    syn_pol::Vector{Float32}
    conn_probs::Matrix{Float32} 
    cumulative::Dict{String, Vector{Int64}}
    layer_names::Vector{String}
    columns_conn_probs::SubArray{Float32, 1, Matrix{Float32}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true}
end
=#


function build_matrix!(Lxx::SparseMatrixCSC{Float32, Int64},cumvalues, conn_probs::StaticArraysCore.SMatrix{8, 8, Float64, 64}, Ncells::Int32, syn_pol::StaticArraysCore.SVector{8, Int64},g_strengths::Vector{Float64})
    ##
    # use maybe threaded paradigm.
    ##
    #Threads.@threads for i = 1:10
    @inbounds for (i,(syn0,v)) in enumerate(zip(syn_pol,cumvalues))
        @inbounds for src in v
            @inbounds for (j,(syn1,v1)) in enumerate(zip(syn_pol,cumvalues))
                @inbounds for tgt in v1
                    if src!=tgt                        
                        prob = conn_probs[i,j]
                        if rand()<prob
                            item = src,tgt,syn0,syn1
                            index_assignment!(item,g_strengths,Lxx)#,Lee,Lie,Lii,Lei)
                        end
                    end
                end
            end
        end
    end
end
=#

#=

SGet in-degrees for each connection for the full-scale (1 mm^2) model
function get_indegrees()
    K = np.zeros([n_layers * n_pops_per_layer, n_layers * n_pops_per_layer])
    for target_layer in layers:
        for target_pop in pops:
            for source_layer in layers:
                for source_pop in pops:
                    target_index = structure[target_layer][target_pop]
                    source_index = structure[source_layer][source_pop]
                    n_target = N_full[target_layer][target_pop]
                    n_source = N_full[source_layer][source_pop]
                    K[target_index][source_index] = round(np.log(1. -
                        conn_probs[target_index][source_index]) / np.log(
                        (n_target * n_source - 1.) / (n_target * n_source))) / n_target
                end
            end
        end
    end
    return K
   
end
=#


#=
"""
Adjust synaptic weights and external drive to the in-degrees
to preserve mean and variance of inputs in the diffusion approximation
function adjust_w_and_ext_to_K(K_full, K_scaling, w, DC)
    K_ext_new = Dict()
    I_ext = Dict()
    for target_layer in layers:
        K_ext_new[target_layer] = {}
        I_ext[target_layer] = {}
        for target_pop in pops:
            target_index = structure[target_layer][target_pop]
            x1 = 0
            for source_layer in layers:
                for source_pop in pops:
                source_index = structure[source_layer][source_pop]
                x1 += w[target_index][source_index] * K_full[target_index][source_index] * \
                        full_mean_rates[source_layer][source_pop]
                end
            end

            if input_type == 'poisson'
                x1 += w_ext*K_ext[target_layer][target_pop]*bg_rate
                K_ext_new[target_layer][target_pop] = K_ext[target_layer][target_pop]*K_scaling
            end
            I_ext[target_layer][target_pop] = 0.001 * neuron_params['tau_syn_E'] * \
                (1. - np.sqrt(K_scaling)) * x1 + DC[target_layer][target_pop]
            w_new = w / np.sqrt(K_scaling)
            w_ext_new = w_ext / np.sqrt(K_scaling)

        end
    end
    return w_new, w_ext_new, K_ext_new, I_ext

end
"""
=#
#=

# Create cortical populations
self.pops = {}
layer_structures = {}
total_cells = 0 

x_dim_scaled = x_dimension * math.sqrt(N_scaling)
z_dim_scaled = z_dimension * math.sqrt(N_scaling)using SparseArrays
using StaticArrays
using ProgressMeter
#using UnicodePlots

"""
This file contains a function stack that creates a network with Potjans and Diesmann wiring likeness in Julia using SpikingNeuralNetworks.jl to simulate
electrical neural network dynamics.
This code draws heavily on the PyNN OSB Potjans implementation code found here:
https://github.com/OpenSourceBrain/PotjansDiesmann2014/blob/master/PyNN/network_params.py#L139-L146
However the translation of this Python PyNN code into performant and scalable Julia was not trivial.

Hard coded Potjans parameters follow.
and then the function outputs adapted Potjans parameters.

TODO for self, considering PR differencing my old branch with this one, as I may have neglected to carry over other constructive code changes.
Old branch worth diffing.
https://github.com/RJsWorkatWSU/SpikingNeuralNetworks.jl/tree/potjans_model
"""
function potjans_params(ccu, scale=1.0::Float64)
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
    v_old=1
    #cum_array = Vector{Any}([])#Matrix{Any}(zeros((2,length(ccu))))

    K = length(ccu)
    cum_array = Vector{Array{UInt32}}(undef,K)
    for i in 1:K
        cum_array[i] = Array{UInt32}([])
    end
    for (k,v) in pairs(ccu)
        ## update the cummulative cell count
        #cumulative[k]=collect(v_old:v+v_old)
        #@show(length(collect(v_old:v+v_old)[:]))
        push!(cum_array,Vector{UInt32}(collect(v_old:v+v_old)[:]))
        v_old=v+v_old
    end    
    @show(typeof(cum_array))
    cum_array = SVector{16,Array{UInt32}}(cum_array)
    syn_pol = Vector{UInt32}(zeros(length(ccu)))
    for (i,(k,v)) in enumerate(pairs(ccu))
        if occursin("E",k) 
            syn_pol[i] = 1
        else
            syn_pol[i] = 0
        end
    end
    syn_pol = SVector{8,UInt32}(syn_pol)
    return (cum_array,ccu,layer_names,conn_probs,syn_pol)
end


"""
A mechanism for scaling cell population sizes to suite hardware constraints.
While Int64 might seem excessive when cell counts are between 1million to a billion Int64 is required.
Only dealing with positive count entities so Usigned is fine.
"""
function auxil_potjans_param(scale=1.0::Float32)

	ccu = Dict{String, UInt32}("23E"=>20683,
		    "4E"=>21915, 
		    "5E"=>4850, 
		    "6E"=>14395, 
		    "6I"=>2948, 
		    "23I"=>5834,
		    "5I"=>1065,
		    "4I"=>5479)
	ccu = Dict{String, UInt32}((k,ceil(Int64,v*scale)) for (k,v) in pairs(ccu))
	Ncells = UInt64(sum([i for i in values(ccu)])+1)
	Ne = UInt64(sum([ccu["23E"],ccu["4E"],ccu["5E"],ccu["6E"]]))
    Ni = UInt64(Ncells - Ne)
    Ncells, Ne, Ni, ccu
end

"""
The entry point to building the whole Potjans model in SNN.jl
Also some of the current density parameters needed to adjust synaptic gain initial values.
"""
function potjans_layer(scale)
    Ncells,Ne,Ni, ccu = auxil_potjans_param(scale)    
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
    genStaticWeights = (;Ncells,g_strengths,ccu,scale)
    potjans_weights(genStaticWeights),Ne,Ni,ccu
end

"""
This function contains synapse selection logic seperated from iteration logic for readability only.
Used inside the nested iterator inside build_matrix.
Ideally iteration could flatten to support the readability of subsequent code.
"""

#matching 
#build_matrix_prot!(::SparseMatrixCSC{Float32, Int64}, ::SparseMatrixCSC{Float32, Int64}, ::SparseMatrixCSC{Float32, Int64}, ::SparseMatrixCSC{Float32, Int64}, ::Vector{Any}, ::SMatrix{8, 8, Float64, 64}, ::Int32, ::SVector{8, UInt64}, ::Vector{Float64})
#Closest candidates are:
#build_matrix_prot!(::SparseMatrixCSC{Float32, Int64}, ::SparseMatrixCSC{Float32, Int64}, ::SparseMatrixCSC{Float32, Int64}, ::SparseMatrixCSC{Float32, Int64}, ::Any, ::SMatrix{8, 8, Float64, 64}, ::Int32, ::SVector{8, Int64}, ::Vector{Float64}) 
function build_matrix_prot!(wig::Float32,jee::Float32,jei::Float32,Lxx::SparseMatrixCSC{Float32, UInt64},cumvalues::SVector{16, Array{UInt32}}, conn_probs::StaticArraysCore.SMatrix{8, 8, Float32, 64}, Ncells::UInt32, syn_pol::StaticArraysCore.SVector{8, UInt32},g_strengths::Vector{Float32})
    # excitatory weights.
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
                                if syn1==1 
                                    setindex!(Lxx,jee, src,tgt)
                                elseif syn1==0# meaning if the same as a logic: Inhibitory post synapse  is true                   
                                    setindex!(Lxx,jei, src,tgt)
                                end
                            elseif syn0==0         
                                if syn1==1 
                                    setindex!(Lxx,wig, src,tgt)
                                elseif syn1==0
                                    setindex!(Lxx,wig, src,tgt)
                                end
                            end 
                        end
                    end
                end
            end            
        end
    end
    #Lxx = Lee+Lei+Lii+Lie
    Lxx[diagind(Lxx)] .= 0.0

    #display(Lxx)
    ## Note to self function return annotations help.
    #Lxx#,Lee,Lei,Lii,Lie
end

function make_proj(xx,pop)
    rowptr, colptr, I, J, index, W = dsparse(xx)
    fireI, fireJ = pop.fire, pop.fire
    g = getfield(pop, :ge)
    SpikingSynapse(W,pre, post, sym)
    syn = SpikingSynapse(rowptr, colptr, I, J, index, W, fireI, fireJ, g)
    return syn
end
"""
Similar to the methods above except that cells and synapses are instantiated in place to cut down on code
"""
function build_neurons_connections(Lee::SparseMatrixCSC{Float32, Int64},Lei::SparseMatrixCSC{Float32, Int64},Lie::SparseMatrixCSC{Float32, Int64},Lii::SparseMatrixCSC{Float32, Int64},cumvalues, Ncells::Int32,syn_pol::StaticArraysCore.SVector{8, Int64})
    @inbounds @showprogress for (syn0,v) in zip(syn_pol,cumvalues)
        @inbounds for (syn1,v1) in zip(syn_pol,cumvalues)
            if syn0==1
                if syn1==1 
                    EE = SNN.IFNF(;N = length(cumvalues), param = SNN.IFParameter())
                    synEE = make_proj(Lee[v,v1],EE)
                elseif syn1==0# meaning if the same as a logic: Inhibitory post synapse  is true                   
                    EI = SNN.IFNF(;N = length(cumvalues), param = SNN.IFParameter())
                    synEI = make_proj(Lei[v,v1],EI)
                end
            elseif syn0==0         
                if syn1==1 
                    IE = SNN.IFNF(;N = length(cumvalues), param = SNN.IFParameter())
                    synIE = make_proj(Lie[v,v1],IE)
                elseif syn1==0
                    II = SNN.IFNF(;N = length(cumvalues), param = SNN.IFParameter())
                    synII = make_proj(Lii[v,v1],II)
                end
            end 
        end
                    
    end
    (EE,EI,IE,II,synII,synIE,synEI,synEE)
end

"""
Build the matrix from the Potjans parameterpotjans_layers.
"""
function potjans_weights(args)
    Ncells, g_strengths, ccu, scale = args
    (cumvalues,_,_,conn_probs,syn_pol) = potjans_params(ccu,scale)    
    #cumvalues = values(cumulative)
    
    Lxx = spzeros(Float32, (Ncells, Ncells))
    #Lie = spzeros(Float32, (Ncells, Ncells))
    #Lei = spzeros(Float32, (Ncells, Ncells))
    #Lii = spzeros(Float32, (Ncells, Ncells))
    (jee,_,jei,_) = g_strengths 
    # Relative inhibitory synaptic weight
    wig = Float32(-20*4.5)

    return build_matrix_prot!(jee,jei,wig,Lxx,cumvalues,conn_probs,Ncells,syn_pol,g_strengths)
    #(EE,EI,IE,II,synII,synIE,synEI,synEE) = build_neurons_connections(Lee,Lei,Lie,Lii,cumvalues, Ncells,syn_pol)
end




#=
Depreciated methods

function index_assignment!(item::NTuple{4, UInt64}, g_strengths::Vector{Float32}, lxx::SparseMatrixCSC{Float32, UInt64})#,lee::Vector{Vector{Tuple{Int64, UInt64}}},lie::Vector{Vector{Tuple{Int64, UInt64}}}, lii::Vector{Vector{Tuple{Int64, Int64}}}, lei::Vector{Vector{Tuple{Int64, Int64}}})
    # excitatory weights.
    (jee,_,jei,_) = g_strengths 
    # Relative inhibitory synaptic weight
    wig = -20*4.5
    (src,tgt,syn0,syn1) = item
    if syn0==1
        if syn1==1            
            setindex!(lxx,jee, src,tgt)
        elseif syn1==0# meaning if the same as a logic: Inhibitory post synapse  is true                   
            setindex!(lxx, jei, src,tgt)
        end
    elseif syn0==0# meaning if the same as a logic: Inhibitory post synapse  is true   
        if syn1==1
            setindex!(lxx, wig, src,tgt)
        elseif syn1==0# eaning meaning if the same as a logic: if occursin("I",k1)      is true               
            @assert syn1==0
            setindex!(lxx,wig, src,tgt)
            @assert syn1==0
        end
    end
end
function build_matrix(Ncells::Int32,just_iterator,g_strengths::Vector{Float64})
    Lee = Vector{Vector{Tuple{Int64, Int64}}}[]
    Lie = Vector{Vector{Tuple{Int64, Int64}}}[]
    Lii = Vector{Vector{Tuple{Int64, Int64}}}[]
    Lei = Vector{Vector{Tuple{Int64, Int64}}}[]
    @showprogress for i in just_iterator
        index_assignment!(i[:],g_strengths,Lxx)#,Lee,Lie,Lii,Lei)
    end
    Lee = Lxx[(i[1],i[2]) for i in Lee]
    Lie = Lxx[Lie[1,:],Lie[2,:]]
    Lii = Lxx[Lii[1,:],Lii[2,:]]
    Lei = Lxx[Lei[1,:],Lei[2,:]]
    Lexc = Lee+Lei
    Linh = Lie+Lii
    return Lee,Lie,Lei,Lii
end



"""
An optional container that is not yet utilized.
"""
#=
struct NetParameter 
    syn_pol::Vector{Float32}
    conn_probs::Matrix{Float32} 
    cumulative::Dict{String, Vector{Int64}}
    layer_names::Vector{String}
    columns_conn_probs::SubArray{Float32, 1, Matrix{Float32}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true}
end
=#


function build_matrix!(Lxx::SparseMatrixCSC{Float32, Int64},cumvalues, conn_probs::StaticArraysCore.SMatrix{8, 8, Float64, 64}, Ncells::Int32, syn_pol::StaticArraysCore.SVector{8, Int64},g_strengths::Vector{Float64})
    ##
    # use maybe threaded paradigm.
    ##
    #Threads.@threads for i = 1:10
    @inbounds for (i,(syn0,v)) in enumerate(zip(syn_pol,cumvalues))
        @inbounds for src in v
            @inbounds for (j,(syn1,v1)) in enumerate(zip(syn_pol,cumvalues))
                @inbounds for tgt in v1
                    if src!=tgt                        
                        prob = conn_probs[i,j]
                        if rand()<prob
                            item = src,tgt,syn0,syn1
                            index_assignment!(item,g_strengths,Lxx)#,Lee,Lie,Lii,Lei)
                        end
                    end
                end
            end
        end
    end
end
=#

#=

SGet in-degrees for each connection for the full-scale (1 mm^2) model
function get_indegrees()
    K = np.zeros([n_layers * n_pops_per_layer, n_layers * n_pops_per_layer])
    for target_layer in layers:
        for target_pop in pops:
            for source_layer in layers:
                for source_pop in pops:
                    target_index = structure[target_layer][target_pop]
                    source_index = structure[source_layer][source_pop]
                    n_target = N_full[target_layer][target_pop]
                    n_source = N_full[source_layer][source_pop]
                    K[target_index][source_index] = round(np.log(1. -
                        conn_probs[target_index][source_index]) / np.log(
                        (n_target * n_source - 1.) / (n_target * n_source))) / n_target
                end
            end
        end
    end
    return K
   
end
=#


#=
"""
Adjust synaptic weights and external drive to the in-degrees
to preserve mean and variance of inputs in the diffusion approximation
function adjust_w_and_ext_to_K(K_full, K_scaling, w, DC)
    K_ext_new = Dict()
    I_ext = Dict()
    for target_layer in layers:
        K_ext_new[target_layer] = {}
        I_ext[target_layer] = {}
        for target_pop in pops:
            target_index = structure[target_layer][target_pop]
            x1 = 0
            for source_layer in layers:
                for source_pop in pops:
                source_index = structure[source_layer][source_pop]
                x1 += w[target_index][source_index] * K_full[target_index][source_index] * \
                        full_mean_rates[source_layer][source_pop]
                end
            end

            if input_type == 'poisson'
                x1 += w_ext*K_ext[target_layer][target_pop]*bg_rate
                K_ext_new[target_layer][target_pop] = K_ext[target_layer][target_pop]*K_scaling
            end
            I_ext[target_layer][target_pop] = 0.001 * neuron_params['tau_syn_E'] * \
                (1. - np.sqrt(K_scaling)) * x1 + DC[target_layer][target_pop]
            w_new = w / np.sqrt(K_scaling)
            w_ext_new = w_ext / np.sqrt(K_scaling)

        end
    end
    return w_new, w_ext_new, K_ext_new, I_ext

end
"""
=#
#=

# Create cortical populations
self.pops = {}
layer_structures = {}
total_cells = 0 

x_dim_scaled = x_dimension * math.sqrt(N_scaling)
z_dim_scaled = z_dimension * math.sqrt(N_scaling)

default_cell_radius = 10 # for visualisation 
default_input_radius = 5 # for visualisation 

for layer in layers:
    self.pops[layer] = {}
    for pop in pops:
        
        y_offset = 0
        if layer == 'L6': y_offset = layer_thicknesses['L6']/2
        if layer == 'L5': y_offset = layer_thicknesses['L6']+layer_thicknesses['L5']/2
        if layer == 'L4': y_offset = layer_thicknesses['L6']+layer_thicknesses['L5']+layer_thicknesses['L4']/2
        if layer == 'L23': y_offset = layer_thicknesses['L6']+layer_thicknesses['L5']+layer_thicknesses['L4']+layer_thicknesses['Lhttps://github.com/russelljjarvis/SpikingNeuralNetworks.jl/blob/master/test/if_net_odessa.jl23']/2
        
        layer_volume = Cuboid(x_dim_scaled,layer_thicknesses[layer],z_dim_scaled)
        layer_structures[layer] = RandomStructure(layer_volume, origin=(0,y_offset,0))


https://github.com/OpenSourceBrain/PotjansDiesmann2014/blob/master/PyNN/scaling.py


=#



for layer in layers:
    self.pops[layer] = {}
    for pop in pops:
        
        y_offset = 0
        if layer == 'L6': y_offset = layer_thicknesses['L6']/2
        if layer == 'L5': y_offset = layer_thicknesses['L6']+layer_thicknesses['L5']/2
        if layer == 'L4': y_offset = layer_thicknesses['L6']+layer_thicknesses['L5']+layer_thicknesses['L4']/2
        if layer == 'L23': y_offset = layer_thicknesses['L6']+layer_thicknesses['L5']+layer_thicknesses['L4']+layer_thicknesses['Lhttps://github.com/russelljjarvis/SpikingNeuralNetworks.jl/blob/master/test/if_net_odessa.jl23']/2
        
        layer_volume = Cuboid(x_dim_scaled,layer_thicknesses[layer],z_dim_scaled)
        layer_structures[layer] = RandomStructure(layer_volume, origin=(0,y_offset,0))


https://github.com/OpenSourceBrain/PotjansDiesmann2014/blob/master/PyNN/scaling.py


=#

