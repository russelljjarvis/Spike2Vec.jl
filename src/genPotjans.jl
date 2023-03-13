#using Distributions
using SparseArrays
using StaticArrays
"""
This file consists of a function stack that seemed necessary to achieve a network with Potjans like wiring in Julia using TrainSpikeNet.jl to simulate.
This code draws heavily on the PyNN OSB Potjans implementation code found here:
https://github.com/OpenSourceBrain/PotjansDiesmann2014/blob/master/PyNN/network_params.py#L139-L146
"""

"""
Hard coded Potjans parameters follow.
and then the function outputs adapted Potjans parameters.
"""

struct NetParameter 
    syn_pol::Vector{Float32}
    conn_probs::Matrix{Float32} 
    cumulative::Dict{String, Vector{Int64}}
    layer_names::Vector{String}
    columns_conn_probs::SubArray{Float32, 1, Matrix{Float32}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true}
end

#function NetParameter()
    #syn_pol,conn_probs,cumulative)
    #return NetParameter(syn_pol,conn_probs,cumulative)
#end    
function potjans_params(ccu, scale=1.0::Float64)
    # a cummulative cell count
    cumulative = Dict{String, Vector{Int64}}()  
    layer_names = SVector{String}(["23E","23I","4E","4I","5E", "5I", "6E", "6I"])    
    # Probabilities for >=1 connection between neurons in the given populations. 
    # The first index is for the target population; the second for the source population
    #             2/3e      2/3i    4e      4i      5e      5i      6e      6i
    conn_probs = SMatrix{Float32}([0.1009  0.1689 0.0437 0.0818 0.0323 0.0     0.0076 0.    
                                    0.1346   0.1371 0.0316 0.0515 0.0755 0.     0.0042 0.    
                                    0.0077   0.0059 0.0497 0.135  0.0067 0.0003 0.0453 0.    
                                    0.0691   0.0029 0.0794 0.1597 0.0033 0.     0.1057 0.    
                                    0.1004   0.0622 0.0505 0.0057 0.0831 0.3726 0.0204 0.    
                                    0.0548   0.0269 0.0257 0.0022 0.06   0.3158 0.0086 0.    
                                    0.0156   0.0066 0.0211 0.0166 0.0572 0.0197 0.0396 0.2252
                                    0.0364   0.001  0.0034 0.0005 0.0277 0.008  0.0658 0.1443 ])
    # hard coded stuff is manipulated below:
    columns_conn_probs = [col for col in eachcol(conn_probs)][1]    
    v_old=1
    for (k,v) in pairs(ccu)
        ## update the cummulative cell count
        cumulative[k]=collect(v_old:v+v_old)
        v_old=v+v_old
    end    
    syn_pol = Vector{Int64}(zeros(length(ccu)))
    for (i,(k,v)) in enumerate(pairs(ccu))
        if occursin("E",k) 
            syn_pol[i] = 1
        else
            syn_pol[i] = 0
        end
    end
    syn_pol = SVector{Int64,(length(syn_pol))}(syn_pol)
    #net = NetParameter(syn_pol,conn_probs,cumulative,layer_names,columns_conn_probs)
    return (cumulative,ccu,layer_names,columns_conn_probs,conn_probs,syn_pol)
end
"""
This function contains synapse selection logic seperated from iteration logic for readability only.
Used inside the nested iterator inside build_matrix.
Ideally iteration could flatten to support the readability of subsequent code.
"""
function index_assignment!(item::NTuple{4, Int64}, w0Weights::SparseMatrixCSC{Float64, Int64}, g_strengths::Vector{Float64}, lee::SparseMatrixCSC{Float32, Int64}, lie::SparseMatrixCSC{Float32, Int64}, lii::SparseMatrixCSC{Float32, Int64}, lei::SparseMatrixCSC{Float32, Int64})
    # excitatory weights.
    (jee,_,jei,_) = g_strengths 
    # Relative inhibitory synaptic weight
    wig = -20*4.5
    (src,tgt,syn0,syn1) = item
    if syn0==1
        if syn1==1            
            setindex!(w0Weights, jee, src,tgt)    
            setindex!(lee,jee, src,tgt)
            @assert lee[src,tgt]>=0.0

        elseif syn1==0# meaning if the same as a logic: Inhibitory post synapse  is true                   
            setindex!(w0Weights, jei, src,tgt)
            setindex!(lei, jei, src,tgt)
            @assert lei[src,tgt]>=0.0

        end
    elseif syn0==0# meaning if the same as a logic: Inhibitory post synapse  is true   
        if syn1==1
            setindex!(w0Weights, wig, src,tgt)
            setindex!(lie, wig, src,tgt)
            @assert w0Weights[src,tgt]<=0.0
            @assert lie[src,tgt]<=0.0

        elseif syn1==0# eaning meaning if the same as a logic: if occursin("I",k1)      is true               
            setindex!(w0Weights, wig, src,tgt)
            setindex!(lii, wig, src,tgt)
            @assert w0Weights[src,tgt]<=0.0
            @assert syn1==0
            @assert lii[src,tgt]<=0.0

        end
    end
end


function build_matrix(cumulative::Dict{String, Vector{Int64}}, conn_probs::Matrix{Float32}, Ncells::Int32, g_strengths::Vector{Float64},syn_pol::Vector{Int64})    

    w0Weights = spzeros(Float64, (Ncells, Ncells))
    Lee = spzeros(Float32, (Ncells, Ncells))
    Lii = spzeros(Float32, (Ncells, Ncells))
    Lei = spzeros(Float32, (Ncells, Ncells))
    Lie = spzeros(Float32, (Ncells, Ncells))

    ##
    # use maybe threaded paradigm.
    # From BA.
    ##
    #Threads.@threads for i = 1:10
    cumvalues = values(cumulative)
    total_len = length(cumvalues)*length(cumvalues)*length(syn_pol)*length(syn_pol)
    iter_item::Vector{NTuple{4, Int64}} = zeros(total_len)
    
    
    
    @inbounds for (i,(syn0,v)) in enumerate(zip(syn_pol,cumvalues))
        @inbounds for src in v
            @inbounds for (j,(syn1,v1)) in enumerate(zip(syn_pol,cumvalues))
                @inbounds for tgt in v1
                    if src!=tgt                        
                        prob = conn_probs[i,j]
                        if rand()<prob
                            item = src,tgt,syn0,syn1
                            push!(iter_item,item)
                            #index_assignment!(item,w0Weights,g_strengths,Lee,Lie,Lii,Lei)                        end
                    end
                end
            end
        end
    end

    map!(index_assignment!, item for iter_item)
    @assert maximum(Lexc[:])>=0.0
    @assert maximum(Linh[:])<=0.0
    return w0Weights,Lee,Lie,Lei,Lii
end
"""
Build the matrix from the Potjans parameters.
 The motivation for this approach is a lower memory footprint motivations.
 a sparse matrix can be stored as a smaller dense matrix.
 A 2D matrix should be stored as 1D matrix of srcs,tgts
 A 2D weight matrix should be stored as 1 matrix, which is redistributed in loops using 
 the 1D matrix of srcs,tgts.

"""
function potjans_weights(args)
    _, _, _, _, _, ccu, scale = args
    (cumulative,ccu,layer_names,_,conn_probs,syn_pol) = potjans_params(ccu,scale)    
    g_strengths = Vector{Float64}([jee,jie,jei,jii])
    w0Weights,Lee,Lie,Lei,Lii = build_matrix(cumulative,conn_probs,Ncells,g_strengths,syn_pol)
    w0Weights,Lee,Lie,Lei,Lii
end

function potjans_layer(scale=1.0::Float64)
	ccu = Dict{String, Int32}("23E"=>20683,
		    "4E"=>21915, 
		    "5E"=>4850, 
		    "6E"=>14395, 
		    "6I"=>2948, 
		    "23I"=>5834,
		    "5I"=>1065,
		    "4I"=>5479)
	ccu = Dict{String, Int32}((k,ceil(Int64,v*scale)) for (k,v) in pairs(ccu))
	Ncells = Int32(sum([i for i in values(ccu)])+1)
	Ne = Int32(sum([ccu["23E"],ccu["4E"],ccu["5E"],ccu["6E"]]))
    Ni = Int32(Ncells - Ne)
    Ncells, Ne, Ni, ccu

end

function potjans_layer()
    scale = 1.0/40.0
    Ncells,Ne,Ni, ccu = potjans_layer(scale)    
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
    genStaticWeights_args = (;Ncells,jee,jie,jei,jii,ccu,scale)
    potjans_weights(genStaticWeights_args),Ne,Ni
end

