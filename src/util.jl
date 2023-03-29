using CUDA


function connect!(c, j, i, σ = 1e-6)
    W = sparse(c.I, c.J, c.W, length(c.rowptr) - 1, length(c.colptr) - 1)
    W[i, j] = σ * randn(Float32)
    c.rowptr, c.colptr, c.I, c.J, c.index, c.W = dsparse(W)
    c.tpre, c.tpost, c.Apre, c.Apost = zero(c.W), zero(c.W), zero(c.W), zero(c.W)
    return nothing
end

function GetApplyLayerOffsets(layer_pre_index, layer_post_index)
    in_synapse_pol = :gi
    exc_synapse_pol = :ge

    rowptr, colptr, I, J, index, W = dense_from_sparse(w)
    fireI, fireJ = post.fire, pre.fire
    g = getfield(post, sym)
    #SpikingSynapse(;@symdict(rowptr, colptr, I, J, index, W, fireI, fireJ, g)..., kwargs...)
end

function Map_also_with_conversion_from_dense_from_sparse(A)
    """
    Create a unit test for this.
        
    Convert dense matrix from sparse matrix.
    """
    At = sparse(A') ## Post synapses
    tgts = A.colptr 
    @show(tgts)
    # transpose means we only have fast columnwise iteration.    
    srcs = At.colptr # Pre-synapses.
    I = rowvals(A) ## Get all rows of adjanency matrix the connectome.
    
    
    WW = nonzeros(A) # get all non-zero values of the adjancey matrix connectome.
    ## the dense product of the adjacency matrix that needs navigation.
    # Navigation steps are below.
    
    J = zero(I) # J is a collection to be filled that is the same shape as I.
    for j in 1:(length(tgts) - 1) ## fill J now.

        # J is filled sytematically monochromatically by the coloumn order of colptr.
        # J is probably the dense mapping from pre-synapse to post synapse.

        J[tgts[j]:(tgts[j+1] - 1)] .= j
    end
    # is J dense?

    @show(sizeof(J))

    @show(J)
    index = zeros(size(I))
    coldown = zeros(eltype(index), length(tgts) - 1)

    ##
    # Another hardcore conversion process.
    # rowptr is fast non-zero presynaptic matrix location found by transpose
    # colptr is fast non-zero post-synaptic matrix location found by transpose
    ##
    for i in 1:(length(srcs) - 1) # iterate presynapses
        for st in srcs[i]:(srcs[i+1] - 1) # grab a pre-synapse and its next neightbour
            j = At.rowval[st] # grab the whole row that this presynapse represents.
            index[st] = tgts[j] + coldown[j] # 
            coldown[j] += 1 # This must be the number of post synaptic locations.
        end
    end
    
    #colptr, I, W, 
    colptr, I, WW # V becomes W
end

function MapPreSynapticToPost(pre, post, layer_pre_index, layer_post_index, sym; σ = 0.0, p = 0.0, kwargs...)
    rowptr, colptr, I, J, index, W = dsparse(w)
    # fireJ is fire pre synaptic
    post_fireI, pre_fireJ = post.fire, pre.fire
    g = getfield(post, sym)
    #SpikingSynapse(;@symdict(rowptr, colptr, I, J, index, W, fireI, fireJ, g)..., kwargs...)
end

function anytype(A,I,J,V,At,rowptr,colptr,coldown,index,return_type::CuArray)
    for j in 1:(length(colptr) - 1)
        J[colptr[j]:(colptr[j+1] - 1)] .= j
    end
    for i in 1:(length(rowptr) - 1)
        for st in rowptr[i]:(rowptr[i+1] - 1)
            j = At.rowval[st]
            index[st] = colptr[j] + coldown[j]
            coldown[j] += 1
        end
    end
    rowptr=convert(typeof(return_type),rowptr)
    colptr=convert(CuArray{Int32},colptr)
    I=convert(CuArray{Int32},I)
    J=convert(typeof(return_type),J)
    index=convert(CuArray{Int32},index)
    rowptr,colptr,I,J,index,V
end


function anytype(A,I,J,V,At,rowptr,colptr,coldown,index,return_type::Array)
    for j in 1:(length(colptr) - 1)
        J[colptr[j]:(colptr[j+1] - 1)] .= j
    end
    for i in 1:(length(rowptr) - 1)
        for st in rowptr[i]:(rowptr[i+1] - 1)
            j = At.rowval[st]
            index[st] = colptr[j] + coldown[j]
            coldown[j] += 1
        end
    end
    rowptr=convert(typeof(return_type),rowptr)
    colptr=convert(Array{Int32},colptr)
    I=convert(Array{Int32},I)
    J=convert(typeof(return_type),J)
    index=convert(Array{Int32},index)
    rowptr,colptr,I,J,index,V


end


function dsparse(A,return_type)        
        At = sparse(A')
        colptr = A.colptr
        rowptr = At.colptr
        I = rowvals(A)
        V = nonzeros(A)
        J = zero(I)
        index = zeros(size(I))
        coldown = zeros(eltype(index), length(colptr) - 1)
        rowptr,colptr,I,J,index,V = anytype(A,I,J,V,At,rowptr,colptr,coldown,index,return_type)
    
    return rowptr,colptr,I,J,index,V
end


function dsparse(A)
    At = sparse(A')
    colptr = A.colptr
    rowptr = At.colptr
    I = rowvals(A)
    V = nonzeros(A)
    J = zero(I)
    # FIXME: Breaks when A is empty
    for j in 1:(length(colptr) - 1)
        J[colptr[j]:(colptr[j+1] - 1)] .= j
    end
    index = zeros(size(I))
    coldown = zeros(eltype(index), length(colptr) - 1)
    for i in 1:(length(rowptr) - 1)
        for st in rowptr[i]:(rowptr[i+1] - 1)
            j = At.rowval[st]
            index[st] = colptr[j] + coldown[j]
            coldown[j] += 1
        end
    end
    rowptr, colptr, I, J, index, V
end






function record!(obj)
    for (key, val) in obj.records
        if isa(key, Tuple)
            sym, ind = key
            push!(val, getindex(getfield(obj, sym),ind))
        else
            push!(val, copy(getfield(obj, key)))
        end
    end
end

function monitor(obj, keys)
    for key in keys
        if isa(key, Tuple)
            sym, ind = key
        else
            sym = key
        end
        typ = typeof(getfield(obj, sym))
        ##
        # bad performance line
        ##
        obj.records[key] = Vector{typ}()
    end
end
function monitor(objs::Array, keys)
    for obj in objs
        monitor(obj, keys)
    end
end


function getrecord(p, sym)
    key = sym
    for (k,val) in p.records
        isa(k, Tuple) && k[1] == sym && (key = k)
    end
    #@show(values(p.records))
    #@show(keys(p.records))
    p.records[key]
end

function clear_records(obj)
    for (key, val) in obj.records
        empty!(val)
    end
end

function record_spikes!(obj)
    push!(val, getindex(getfield(obj, :record_spikes),ind))
end
function monitor_spikes(obj)
    typ = typeof(getfield(obj, :fire))
    obj.record_spikes = Vector{typ}()
end

function monitor_spikes(objs::Array)
    for obj in objs
        monitor_spikes(obj, :fire)
    end
end


#=
cellsa = Array{Union{Missing,Any}}(undef, 1, Int(findmax(y)[1]))
nac = Int(findmax(y)[1])
for (inx, cell_id) in enumerate(1:nac)
    cellsa[inx] = []
end
@inbounds for cell_id in unique(y)
    @inbounds for (time, cell) in collect(zip(x, y))
        if Int(cell_id) == cell
            append!(cellsa[Int(cell_id)], time)
        end
    end
end
cellsa

end
=#




function divide_epoch(nodes,times,duration)
    t1=[]
    n1=[]
    t0=[]
    n0=[]
    for (n,t) in zip(nodes,times)
        if t<=duration
            append!(t0,t)
            append!(n0,n)            
        else
            append!(t1,t)
            append!(n1,n)
        end
    end
    (t0,n0,t1,n1)
end



function get_vector_coords()
    (t0,n0,t1,n1) = divide_epoch(nodes,times,duration)
    maxt = findmax(sort!(unique(vcat(spikes,ground_spikes))))[1]

end


#=
function nloss(E,ngt_spikes,ground_spikes)
    spikes = get_spikes(E)
    spikes = [s/1000.0 for s in spikes]
	maxt = findmax(sort!(unique(vcat(spikes,ground_spikes))))[1]
    if size(spikes)[1]>1
        t, S = SPIKE_distance_profile(spikes, ground_spikes;t0=0,tf = maxt)
        spkdistance = SpikeSynchrony.trapezoid_integral(t, S)/(t[end]-t[1]) # == SPIKE_distance(y1, y2)
    else
        spkdistance = 10.0
    end
	if length(spikes)>1
		custom_raster2(spikes,ground_spikes)
		custom_raster(spikes,ground_spikes)
	end
	spkdistance*=spkdistance

    delta = abs(size(spikes)[1] - ngt_spikes)
    return spkdistance+delta

end
=#

function get_trains(P::Array)    
    X = Float32[]
    y0 = UInt32[0]
    Y = UInt32[]
    for p in P
        x, y = get_trains(p)
        append!(X, x)
        append!(Y, y .+ sum(y0))
        push!(y0, p.N)
    end
    return (X,Y)
end

function get_trains(p)
    fire = p.records[:fire]
    x, y = Float32[], Int64[]
    for time in eachindex(fire)
        for neuron_id in findall(fire[time])
            push!(x, time)
            push!(y, neuron_id)
        end
    end
    return (x,y)
end
#get_trains = get_trains(p)  

@inline function exp32(x::Float32)
    x = ifelse(x < -10f0, -32f0, x)
    x = 1f0 + x / 32f0
    x *= x; x *= x; x *= x; x *= x; x *= x
    return x
end

@inline function exp256(x::Float32)
    x = ifelse(x < -10f0, -256f0, x)
    x = 1.0f0 + x / 256.0f0
    x *= x; x *= x; x *= x; x *= x
    x *= x; x *= x; x *= x; x *= x
    return x
end

macro symdict(x...)
    ex = Expr(:block)
    push!(ex.args, :(d = Dict{Symbol,Any}()))
    for p in x
        push!(ex.args, :(d[$(QuoteNode(p))] = $(esc(p))))
    end
    push!(ex.args, :(d))
    return ex
end
