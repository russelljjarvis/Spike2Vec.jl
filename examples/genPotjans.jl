using JLD2
using Distributions, Random
using SparseArrays
using SnoopCompile
"""
This file consists of a function stack that seemed necessary to achieve a network with Potjans like wiring in Julia using TrainSpikeNet.jl to simulate.
This code draws heavily on the PyNN OSB Potjans implementation code found here:
https://github.com/OpenSourceBrain/PotjansDiesmann2014/blob/master/PyNN/network_params.py#L139-L146
"""

"""
Hard coded Potjans parameters follow.
and then the function outputs adapted Potjans parameters.
"""


function potjans_params(ccu, scale=1.0::Float64)
    cumulative = Dict{String, Vector{Int64}}() 
    layer_names = Vector{String}(["23E","23I","4E","4I","5E", "5I", "6E", "6I"])    
 
    # Probabilities for >=1 connection between neurons in the given populations. 
    # The first index is for the target population; the second for the source population
    #             2/3e      2/3i    4e      4i      5e      5i      6e      6i
    conn_probs = Matrix{Float64}([0.1009  0.1689 0.0437 0.0818 0.0323 0.0     0.0076 0.    
                                    0.1346   0.1371 0.0316 0.0515 0.0755 0.     0.0042 0.    
                                    0.0077   0.0059 0.0497 0.135  0.0067 0.0003 0.0453 0.    
                                    0.0691   0.0029 0.0794 0.1597 0.0033 0.     0.1057 0.    
                                    0.1004   0.0622 0.0505 0.0057 0.0831 0.3726 0.0204 0.    
                                    0.0548   0.0269 0.0257 0.0022 0.06   0.3158 0.0086 0.    
                                    0.0156   0.0066 0.0211 0.0166 0.0572 0.0197 0.0396 0.2252
                                    0.0364   0.001  0.0034 0.0005 0.0277 0.008  0.0658 0.1443 ])
    # hard coded stuff is manipulated below:
    columns_conn_probs = [col for col in eachcol(conn_probs)][1]
    
    ## this list gets reorganised to reflect top excitatory bottom inhibitory.
    ## Rearrange the whole matrix so that excitatory connections form a top partition 
    #and inhibitory neurons form a bottom partition.

    v_old=1
    for (k,v) in pairs(ccu)
        ## A cummulative cell count
        cumulative[k]=collect(v_old:v+v_old)
        v_old=v+v_old
    end    
    return (cumulative,ccu,layer_names,columns_conn_probs,conn_probs)
end
"""
This function contains synapse selection logic seperated from iteration logic for readability only.
Used inside the nested iterator inside build_matrix.
Ideally iteration could flatten to support the readability of subsequent code.
"""
#function index_assignment!(item,w0Weights,Lexc,Linh,g_strengths::Vector{Float64})#,Ne,Ni)  
function index_assignment!(item::Tuple{Int64, Int64, String, String}, w0Weights::SparseMatrixCSC{Float64, Int64}, g_strengths::Vector{Float64})
    (jee,jie,jei,jii) = g_strengths
    # Mean synaptic weight for all excitatory projections except L4e->L2/3e
    w_mean = 87.8e-3  # nA
    w_ext = 87.8e-3  # nA
    # Mean synaptic weight for L4e->L2/3e connections 
    # See p. 801 of the paper, second paragraph under 'Model Parameterization', 
    # and the caption to Supplementary Fig. 7
    w_234 = 2 * w_mean  # nA
    # Standard deviation of weight distribution relative to mean for L4e->L2/3e
    # This value is not mentioned in the paper, but is chosen to match the 
    # original code by Tobias Potjans
    w_rel_234 = 0.05
    # Relative inhibitory synaptic weight
    wig = -4.
    (src,tgt,k,k1) = item

    if occursin("E",k) 
        if occursin("E",k1)   
            
            if occursin("4E",k) & occursin("23E",k)
                d = Normal(w_234,w_rel_234)
                td = truncated(d, 0.0, Inf)
                weight = abs(rand(td, 1))
                setindex!(w0Weights, weight, src,tgt)
            else         
                setindex!(w0Weights, jee, src,tgt)
            end
    
        else# meaning if the same as a logic: occursin("I",k1) is true                   
            setindex!(w0Weights, jei, src,tgt)
        end
    else
        @assert occursin("I",k) 
        # meaning meaning if the same as a logic: elseif occursin("I",k) is true  
        if occursin("E",k1)    
            setindex!(w0Weights, wig, src,tgt)
            @assert w0Weights[src,tgt]<=0.0
            @assert w0Weights[src,tgt]<=0.0

        else# eaning meaning if the same as a logic: if occursin("I",k1)      is true               
            setindex!(w0Weights, wig, src,tgt)
            #@show(w0Weights[src,tgt])
            @assert w0Weights[src,tgt]<=0.0

            @assert occursin("I",k1) 
        end
    end        
end
"""
This function contains iteration logic seperated from synapse selection logic for readability only.
"""

function index_assignment!(w0Weights::SparseMatrixCSC{Float64, Int64},item::Tuple{Int64, Int64, String, String}, lee::SparseMatrixCSC{Float32, Int64}, lie::SparseMatrixCSC{Float32, Int64}, lii::SparseMatrixCSC{Float32, Int64}, lei::SparseMatrixCSC{Float32, Int64})
    (src,tgt,k,k1) = item
    if occursin("E",k) 
        if occursin("E",k1) 
            setindex!(lee, w0Weights[src,tgt], src,tgt)
            @assert lee[src,tgt]>=0.0

        else# meaning if the same as a logic: occursin("I",k1) is true                   
            setindex!(lei, w0Weights[src,tgt], src,tgt)
            @assert lei[src,tgt]>=0.0

        end
    else
        @assert occursin("I",k) 
        # meaning meaning if the same as a logic: elseif occursin("I",k) is true  
        if occursin("E",k1)    
            setindex!(lie, w0Weights[src,tgt], src,tgt)
            @assert lie[src,tgt]<=0.0

        else# eaning meaning if the same as a logic: if occursin("I",k1)      is true               
            setindex!(lii, w0Weights[src,tgt], src,tgt)
            @assert lii[src,tgt]<=0.0

            @assert occursin("I",k1) 
        end
    end        
end
#=
struct NetworkParameter{} 
    τm::FT = 20ms
    τe::FT = 5ms
    τi::FT = 10ms
    Vt::FT = -50mV
    Vr::FT = -60mV
    El::FT = Vr
end
=#
function build_matrix(cumulative::Dict{String, Vector{Int64}}, conn_probs::Matrix{Float64}, Ncells::Int32, g_strengths::Vector{Float64})    

    w0Weights = spzeros(Float64, (Ncells, Ncells))
    Lee = spzeros(Float32, (Ncells, Ncells))
    Lii = spzeros(Float32, (Ncells, Ncells))
    Lei = spzeros(Float32, (Ncells, Ncells))
    Lie = spzeros(Float32, (Ncells, Ncells))

    ##
    # use maybe threaded paradigm.
    # From BA.
    ##
    @inbounds for (i,(k,v)) in enumerate(pairs(cumulative))
        @inbounds for src in v
            @inbounds for (j,(k1,v1)) in enumerate(pairs(cumulative))
                @inbounds for tgt in v1
                    if src!=tgt                        
                        prob = conn_probs[i,j]
                        if rand()<prob
                            #append!(edge_dict[src],tgt)
                            item = src,tgt,k,k1
                            index_assignment!(item,w0Weights,g_strengths)
                            #@show(typeof(Lee))
                            index_assignment!(w0Weights,item,Lee,Lie,Lii,Lei)
                            #lee::Vector{Int32}, lie::Vector{Int32}, lii::Vector{Int32}, lie::Vector{Int32})
                        end
                        @assert src!=0
                        @assert tgt!=0
                    end
                end
            end
        end
    end
    #Lee,Lie,Lei,Lii = sort(Lee),sort(Lie),sort(Lei),sort(Lii)
    #Lee,Lie,Lei,Lii = filter!(x->x!=0,Lee),filter!(x->x≠0,Lie),filter!(x->x≠0,Lei),filter!(x->x≠0,Lii)
    #@show(Li)
    #Li = w0Weights[Lii,:]

    #@assert maximum(Li)<=0.0
    #Li = w0Weights[:,Lii]

    #@assert maximum(Li)<=0.0
    #@assert maximum(Lexc[:])>=0.0
    #@assert maximum(Linh[:])<=0.0

    return w0Weights,Lee,Lie,Lei,Lii#,Lexc,Linh
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
    #
    #(Ncells::Int64, jee::Float64, jie::Float64, jei::Float64, jii::Float64, ccu, scale=1.0::Float64) = args
    Ncells, jee, jie, jei, jii, ccu, scale = args
    (cumulative,ccu,layer_names,_,conn_probs) = potjans_params(ccu,scale)    
    g_strengths = Vector{Float64}([jee,jie,jei,jii])
    w0Weights,Lee,Lie,Lei,Lii = build_matrix(cumulative,conn_probs,Ncells,g_strengths)
    w0Weights = w0Weights.*0.1
    w0Weights,Lee,Lie,Lei,Lii
end

function genStaticWeights(args)
    #(_,w0Weights,Lexc,Linh) = 
    #dropzeros!(w0Weights)
    #Ncells = args[1]
    #nc0,w0Index = build_w0Index(edge_dict,Ncells)
    return potjans_weights(args)
end

"""
Syntactically necesarry for TrainingSpikeNet package, I am not sure what the reason is I suspect the reason is dense formats.
"""
#=
function build_w0Index(edge_dict,Ncells)
    nc0Max = 0
    # what is the maximum out degree of this ragged array?
    for (k,v) in pairs(edge_dict)
        templength = length(v)
        if templength>nc0Max
            nc0Max=templength
        end
    end
    # outdegree
    nc0 = Int.(nc0Max*ones(Ncells))
    ##
    # Force ragged array into smallest dense rectangle (contains zeros for undefined synapses) 
    ##
    w0Index = zeros(Int64, (nc0Max,Ncells))

    #w0Index = spzeros(Int64,nc0Max,Ncells)
    for pre_cell = 1:Ncells
        post_cells = edge_dict[pre_cell]
        stride_length = length(edge_dict[pre_cell])
        w0Index[1:stride_length,pre_cell] = post_cells
    end
    nc0,w0Index
end
=#
