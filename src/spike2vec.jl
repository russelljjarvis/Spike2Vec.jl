#
# Eventually this will need to move to source.
# 
using HDF5
using Plots
using OnlineStats
using Plots
using JLD2
using SpikeSynchrony
using LinearAlgebra
using Revise
using StatsBase
using ProgressMeter
using LinearAlgebra
using Revise
using UMAP
using Distances
using StaticArrays
using Clustering
using StatsBase
import OnlineStatsBase.CircBuff
using Infiltrator
#using InducingPoints
#using Arrow 
using RecurrenceAnalysis
#using DataStructures

using Statistics
#scale(A) = (A .- mean(A,dims=1)) ./ std(A,dims=1)
#=
function normalize!(A::Matrix)
    @inbounds for (ind,col) in enumerate(eachcol(A))
        A[:,ind] .= (col.-mean(A))./std(A)
    end
end
=#
import LinearAlgebra.normalize!
import LinearAlgebra.norm

#normalize(A) = (A .- mean(A,dims=1)) ./ std(A,dims=1)
#@show(help(svdvals))

"""
# statistics-Shinomoto2009_e1000433
# https://github.com/NeuralEnsemble/elephant/blob/master/elephant/statistics.py
"""
function lvr(time_intervals, R=5*0.001)

    N = length(time_intervals)
    t = time_intervals[1:N-1] .+ time_intervals[2,:]
    frac1 = 4.0 * time_intervals[1:N-1].*time_intervals[2,:] ./ t.^2.0
    frac2 = 4.0 * R ./ t
    lvr = (3.0 / (N-1.0)) * sum((1.0.-frac1) .* (1.0.+frac2))
    lvr
end


"""
On a RTSP packet, get a CV. 
"""
function CV(spikes_1d_vector::AbstractArray)
    isi_s = Float32[] # the total lumped population ISI distribution.        
    @inbounds for (ind,x) in enumerate(spikes_1d_vector)
        if ind>1
            isi_current = x-spikes_1d_vector[ind-1]
            push!(isi_s,isi_current)
        end
    end
    std(isi_s)/mean(isi_s)
end



"""
Divide epoch into windows
"""
function divide_epoch(nodes::AbstractVector,times::AbstractVector,start::Real,stop::Real)
    n0=Vector{UInt32}([])
    t0=Vector{Float32}([])
    @inbounds for (n,t) in zip(nodes,times)
        if start<=t && t<=stop
            append!(t0,t)#)
            append!(n0,n)
        end
    end
    time_raster =  Array{}([Float32[] for i in 1:maximum(nodes)+1])
    for (neuron,t) in zip(n0,t0)
        append!(time_raster[neuron],t)        
    end
    time_raster
end

function divide_epoch(times::AbstractVector,start::Real,stop::Real)
    t0=Vector{Float32}([])
    @inbounds for t in times
        if start<=t && t<=stop
            @assert start<=t<=stop
            push!(t0,abs(t))
        end
    end
    t0::Vector{Float32}
end

function get_vector_coords_uniform!(one_neuron_surrogate::AbstractArray, neurons_obs::AbstractArray, self_distances::AbstractArray;metric="kreuz")
    @inbounds for (ind,n1_) in enumerate(neurons_obs)
        if length(n1_) > 0 && length(one_neuron_surrogate) > 0
            pooledspikes = copy(one_neuron_surrogate)
            append!(pooledspikes,n1_)
            maxt = maximum(sort!(unique(pooledspikes)))
            t1_ = sort(unique(n1_))
            if length(t1_)>1

                if metric=="kreuz"
                    _, S = SPIKE_distance_profile(t1_,one_neuron_surrogate;t0=0,tf = maxt)
                    self_distances[ind] = abs(sum(S))
                elseif metric=="CV"
                        self_distances[ind] = CV(t1_)
                elseif metric=="autocov"
                        self_distances[ind] = autocov( t1_, [length(t1_)-1],demean=true)[1]

                elseif metric=="LV"

                        self_distances[ind] = lvr(t1_,maximum(t1_))
                elseif metric=="hybrid"
                        _, S = SPIKE_distance_profile(t1_,one_neuron_surrogate;t0=0,tf = maxt)
                        self_distances[ind] = abs(sum(S))
                        self_distances[ind] += lvr(t1_,maximum(t1_))
                        self_distances[ind] += sum(t1_)

                elseif metric=="count"
                        self_distances[ind] = sum(t1_)
                end

            else # If no spikes in this window, don't reflect that there was agreement
                 # ie self_distances[ind] = 0, reflects agreement
                 # reflect a big disagreement.
                self_distances[ind]=abs(length(n1_)-length(t1_))
            end
        end
    end
end

function array_of_empty_vectors(T, dims...)
    array = Array{Vector{T}}(undef, dims...)
    for i in eachindex(array)
        array[i] = Vector{T}()
    end
    array
end
"""
sw: start window a length used to filter spikes.
windex: current window index
times_associated: times (associated with) indices, before stop window length applied
"""
function spike_matrix_divided(spikes_raster::Vector{Any},number_divisions_size::Int,numb_neurons::Int,maxt::Real;displace=true)
    step_size = maxt/number_divisions_size
    end_windows = Vector{Float32}(collect(step_size:step_size:step_size*number_divisions_size))
    start_windows = Vector{Float32}(collect(0:step_size:(step_size*number_divisions_size)-step_size))

    mat_of_spikes = array_of_empty_vectors(Vector{Float32},(length(spikes_raster),length(end_windows)))
    spike_matrix_divided!(mat_of_spikes,spikes_raster,step_size,end_windows,start_windows,displace)

    mat_of_spikes::Matrix{Vector{Vector{Float32}}},start_windows::Vector{Float32},end_windows::Vector{Float32}
end
function spike_matrix_divided!(mat_of_spikes::Matrix{Vector{Vector{Float32}}},spikes_raster,step_size,end_windows,start_windows,displace)
    @inbounds for (neuron_id,only_one_neuron_spike_times) in enumerate(spikes_raster)
        @inbounds @showprogress for (windex,final_end_time) in enumerate(end_windows)
            sw = start_windows[windex]
            observed_spikes = divide_epoch(only_one_neuron_spike_times,sw,final_end_time)
            if displace
                push!(mat_of_spikes[neuron_id,windex],observed_spikes)

            else
                push!(mat_of_spikes[neuron_id,windex],observed_spikes.-sw)

            end
        end
    end
end


function get_window!(nlist,tlist,observed_spikes,sw)
    Nx=Vector{UInt32}([])
    Tx=Vector{Float64}([])
    @inbounds for (i, t) in enumerate(observed_spikes)
        for tt in t
            if length(t)!=0
                push!(Nx,i)
                push!(Tx,Float32(sw+tt))
            end
        end
    end
    push!(nlist,Nx)
    push!(tlist,Tx)
end


function ragged_to_lists(ragged_array::AbstractVecOrMat)
    Nx=Vector{UInt32}([])
    Tx=Vector{Float32}([])
    @inbounds for (i, t) in enumerate(ragged_array)
        for tt in t
            push!(Nx,i)
            push!(Tx,tt)
        end
    end
    (Nx,Tx)
end
function compute_metrics_on_matrix_divisions(div_spike_mat_no_displacement::Matrix{Vector{Vector{Float32}}};metric="kreuz",disk=false)
    (nrow::UInt32,ncol::UInt32)=size(div_spike_mat_no_displacement)
    mat_of_distances = Array{Float32}(undef, nrow, ncol)
    refspikes = div_spike_mat_no_displacement[:,:] 
    #max_spk_countst = Int32(trunc(maximum([length(times[2][1]) for times in enumerate(div_spike_mat_no_displacement)])))
    #@infiltrate
    max_spk_countst = Int32(trunc(maximum([length(times) for times in enumerate(div_spike_mat_no_displacement)])))

    maximum_time = maximum([times[2][1] for times in enumerate(div_spike_mat_no_displacement)])[1]
    
    temp = LinRange(0.0, maximum_time, max_spk_countst)
    linear_uniform_spikes = Vector{Float32}([i for i in temp[:]])

    compute_metrics_on_matrix_divisions!(div_spike_mat_no_displacement,mat_of_distances,linear_uniform_spikes,nrow;metric=metric)    
    (mat_of_distances::Array{Float32})
end

function compute_metrics_on_matrix_self_past_divisions(div_spike_mat_no_displacement::Matrix{Vector{Vector{Float32}}};disk=false)
    (nrow::UInt32,ncol::UInt32)=size(div_spike_mat_no_displacement)
    mat_of_distances = Array{Float64}(undef, nrow, ncol)

    compute_metrics_on_matrix_self_past_divisions!(div_spike_mat_no_displacement,mat_of_distances)    
    (mat_of_distances::Array{Float64})
end
function compute_metrics_on_matrix_self_past_divisions!(div_spike_mat_no_displacement::Matrix{Vector{Vector{Float32}}},mat_of_distances::Array{Float64})
    (nrow::Int32,ncol::UInt32)=size(div_spike_mat_no_displacement)
    neurons_old = div_spike_mat_no_displacement[:,1]
    @inbounds for (indc,neurons) in enumerate(eachcol(div_spike_mat_no_displacement))
        neurons = neurons[1]
        if indc!=1 && length(neurons_old)>0  && length(neurons)>0
            self_distances = Vector{Float32}(zeros(nrow))
            get_vector_coords_uniform!(neurons_old, neurons, self_distances; metric="kreuz")
            mat_of_distances[:,indc] = copy(self_distances)
            #neurons_old = neurons


        end
        neurons_old = copy(neurons)

    end
    #normalize!(mat_of_distances)
    #mat_of_distances[isnan.(mat_of_distances)] .= maximum(mat_of_distances[!isnan.(mat_of_distances)])
    #@ mat_of_distances .+ minimum(mat_of_distances)



    normalize!(mat_of_distances)
    @assert norm(mat_of_distances)==1
    #mat_of_distances[isnan.(mat_of_distances)] .= 0.0

    sum_varr=0
    @inbounds for row in eachrow(mat_of_distances) sum_varr+=var(row) end
    sum_varr
end


function compute_metrics_on_matrix_divisions!(div_spike_mat_no_displacement::Matrix{Vector{Vector{Float32}}},mat_of_distances::Array{<:Real},linear_uniform_spikes::Vector{Float32},nrow::UInt32;metric="kreuz")
    @inbounds for (indc,neurons) in enumerate(eachcol(div_spike_mat_no_displacement))
        self_distances = Vector{Float32}(zeros(nrow))
        if !isa(neurons[1], Number)
            neurons = [ n[1] for n in neurons ]
        end
        #if !isa(one_neuron_surrogate[1], Number)
        #    one_neuron_surrogate = one_neuron_surrogate[1]
        #end

        get_vector_coords_uniform!(linear_uniform_spikes, neurons, self_distances; metric=metric)
        mat_of_distances[:,indc] = self_distances

    end
    #mat_of_distances[isnan.(mat_of_distances)] .= 0.0
    #normalize!(mat_of_distances)
    #mat_of_distances[isnan.(mat_of_distances)] .= 0.0

    sum_varr=0
    @inbounds for row in eachrow(mat_of_distances)
        sum_varr+=var(row)
    end
    @show(sum_varr)
end

function compute_metrics_on_divisions(division_size::Integer,numb_neurons::Integer,maxt::Real;plot=false,file_name="stateTransMat.png",metric="kreuz",disk=false)

    spike_distance_size = length(end_windows)
    mat_of_distances = Array{Float64}(undef, numb_neurons, spike_distance_size)
    nlist = Array{Vector{UInt32}}([])
    tlist = Array{Vector{Float32}}([])
    sum_varr = 0.0
    (start_windows,end_windows,spike_distance_size,sum_varr) = compute_metrics_on_divisions!(mat_of_distances::Array{Float64},nlist::Array{Vector{UInt32}},tlist::Array{Vector{Float32}},nodes::Vector{UInt32},times::Vector{<:Real},division_size::Integer,numb_neurons::Integer,maxt::Real;sum_varr=sum_varr,plot=false,file_name="stateTransMat.png",metric="kreuz",disk=false)
    (mat_of_distances::Array{Float64},nlist::Array{Vector{UInt32}},tlist::Array{Vector{Float32}},sum_varr::Float32)
end

function compute_metrics_on_divisions!(mat_of_distances::Array{Float64},nodes::Vector{UInt32},times::Vector{<:Real},division_size::Integer,numb_neurons::Integer,maxt::Real;sum_varr=nothing,metric="kreuz",disk=false)
    step_size = maxt/division_size
    end_windows = Vector{Float64}(collect(step_size:step_size:step_size*division_size))
    start_windows = Vector{Float64}(collect(0:step_size:(step_size*division_size)-step_size))
    refspikes = divide_epoch(nodes,times,start_windows[2],end_windows[2])
    max_spk_counts = Int32(round(mean([length(times) for times in enumerate(refspikes[:])])))
    temp = LinRange(0.0, maximum(end_windows[2]), max_spk_counts)
    linear_uniform_spikes = Vector{Float64}([i for i in temp[:]])
    @inbounds for (ind,toi) in enumerate(end_windows)
        sw = start_windows[ind]
        observed_spikes = divide_epoch(nodes,times,sw,toi)    
        self_distances = Vector{Float64}(zeros(numb_neurons))
        get_vector_coords_uniform!(linear_uniform_spikes, observed_spikes, self_distances; metric=metric)
        mat_of_distances[:,ind] = self_distances
        get_window!(nlist,tlist,observed_spikes,sw)
    end
    mat_of_distances[isnan.(mat_of_distances)] .= 0.0
    normalize!(mat_of_distances)

    sum_varr=0
    @inbounds for row in eachrow(mat_of_distances)
        sum_varr+=var(row)
    end
    sum_varc=0
    @inbounds for col in eachcol(mat_of_distances)
        sum_varc+=var(col)
    end   

end

function label_exhuastively_distmat!(mat_of_distances::AbstractVecOrMat,distance_matrix::AbstractVecOrMat,sws,ews,times;threshold::Real=5)
    cnts_total = 0.0
    cnts_threshold = 0.0

    template_index_dict = Dict()
    @inbounds for (ind,row) in enumerate(eachcol(mat_of_distances))
        #stop_at_half = Int(trunc(length(eachcol(mat_of_distances))/2))
        #if ind <= stop_at_half

        ##
        ## TODO build time windows here!
        ##
        ##
        @inbounds for (ind2,row2) in enumerate(eachcol(mat_of_distances))
            if ind!=ind2
                cnts_total += 1.0
                distance = evaluate(Euclidean(),row,row2)
                if distance<threshold
                    cnts_threshold += 1.0
                    distance_matrix[ind,ind2] = abs(distance)
                    if !(haskey(template_index_dict, ind))
                        template_index_dict[ind] = []
                    end
                    subtimes = divide_epoch(times,sws[ind2],ews[ind2])
                    @show(subtimes)
                    push!(template_index_dict[ind],subtimes)
                    #@show(maximum(ews),ind2)
                end
            end
        end
    end
    #@infiltrate

    repeatitive = cnts_threshold/cnts_total
    (repeatitive,template_index_dict)

end
#=
"""
previously label_online_distmat
"""

function label_exhuastively_distmat!(mat_of_distances::AbstractVecOrMat,distance_matrix::AbstractVecOrMat,sws,ews;threshold::Real=5)
    cnts_total = 0.0
    cnts_threshold = 0.0


    @inbounds for (ind,row) in enumerate(eachcol(mat_of_distances))
        #stop_at_half = Int(trunc(length(eachcol(mat_of_distances))/2))
        #if ind <= stop_at_half
        @inbounds for (ind2,row2) in enumerate(eachcol(mat_of_distances))
            if ind!=ind2
                cnts_total += 1.0
                distance = evaluate(Euclidean(),row,row2)
                if distance<threshold
                    cnts_threshold += 1.0
                    distance_matrix[ind,ind2] = abs(distance)
                end
            end
        end
    end
    repeatitive = cnts_threshold/cnts_total
    repeatitive

end
=#
"""
label_exhuastively_distmat!
"""
function label_exhuastively_distmat(mat_of_distances::AbstractVecOrMat,sws,ews,times;threshold::Real=5, disk=false)
    if !disk
        distance_matrix = zeros(length(eachcol(mat_of_distances)),length(eachcol(mat_of_distances)))
    else
        io = open("/tmp/mmap.bin", "w+")
        distance_matrix = mmap(io, Matrix{Float32}, (length(eachcol(mat_of_distances)),length(eachcol(mat_of_distances))))
    end
    repeatitive,dict = label_exhuastively_distmat!(mat_of_distances,distance_matrix,sws,ews,times;threshold)
    distance_matrix::AbstractVecOrMat,repeatitive::Real,dict::Dict
end

"""
https://juliadynamics.github.io/RecurrenceAnalysis.jl/stable/quantification/#RQA-Measures
:RR: recurrence rate (see recurrencerate)
:DET: determinsm (see determinism)
:L: average length of diagonal structures (see dl_average)
:Lmax: maximum length of diagonal structures (see dl_max)
:DIV: divergence (see divergence)
:ENTR: entropy of diagonal structures (see dl_entropy)
:TREND: trend of recurrences (see trend)
:LAM: laminarity (see laminarity)
:TT: trapping time (see trappingtime)
:Vmax: maximum length of vertical structures (see vl_max)
:VENTR: entropy of vertical structures (see vl_entropy)
:MRT: mean recurrence time (see meanrecurrencetime)
:RTE recurrence time entropy (see rt_entropy)
:NMPRT: number of the most probable recurrence time (see nmprt)
"""
function recurrence_mat(mat_of_distances)#,assign,nlist,tlist,start_windows,end_windows,nodes,times,ε::Real=5)
    #@infiltrate
    sss =  StateSpaceSet(hcat(mat_of_distances))
    R = RecurrenceMatrix(sss, ε; metric = Euclidean(), parallel=true)
    xs, ys = RecurrenceAnalysis.coordinates(R)# -> xs, ys
    #network = RecurrenceAnalysis.SimpleGraph(R)
    #graphplot(network)
    #savefig("sanity_check_markov.png")
    #p=Plots.scatter()
    return rqa(R),xs, ys,sss,R
end
#=
"Cluster center and the number of observations"
mutable struct Cluster{T<:Number}
    value::Vector{T}
    n::Int
    Cluster(T, p::Integer = 0) = new{T}(zeros(T, p), 0)
end
Base.show(io::IO, o::Cluster) = print(io, "Cluster: nobs=$(o.n), value=$(o.value)")
Base.isless(a::Cluster, b::Cluster) = isless(a.n, b.n)
nobs(o::Cluster) = o.n
value(o::Cluster) = o.value

"""
    KMeans(k; rate=LearningRate(.6))

Approximate K-Means clustering of `k` clusters.

# Example

    x = [randn() + 5i for i in rand(Bool, 10^6), j in 1:2]

    o = fit!(KMeans(2, 2), eachrow(x))

    sort!(o; rev=true)  # Order clusters by number of observations

    classify(o, x[1])  # returns index of cluster closest to x[1]
"""
#import OnlineStats.VectorOb
const VectorOb{T} = Union{AbstractVector{<:T}, Tup{T}}
mutable struct KMeans{T, C <: NTuple{N, Cluster{T}} where N, W} <: OnlineStat{VectorOb{Number}}
    value::C
    buffer::Vector{T}
    rate::W
    n::Int
end
KMeans(T::Type{<:Number}, k::Integer; kw...) = KMeans(k, T; kw...)

function KMeans(k::Integer, T::Type{<:Number} = Float64; rate=LearningRate())
    KMeans(Tuple(Cluster(T) for i in 1:k), zeros(T, k), rate, 0)
end

Base.show(io::IO, o::KMeans) = AbstractTrees.print_tree(io, o)
AbstractTrees.printnode(io::IO, o::KMeans) = print(io, "KMeans($(length(o.value))) | n=$(nobs(o))")
AbstractTrees.children(o::KMeans) = value(o)
function Base.sort!(o::KMeans; kw...)
    o.value = Tuple(sort!(collect(o.value); kw...))
    o
end

function _fit!(o::KMeans{T}, x) where {T}
    o.n += 1
    if o.n == 1
        p = length(x)
        o.value = Tuple(Cluster(T, p) for _ in o.value)
    end
    if o.n ≤ length(o.value)
        cluster = o.value[o.n]
        cluster.value[:] = collect(x)
        cluster.n += 1
    else
        # fill!(o.buffer, 0.0)
        for k in eachindex(o.buffer)
            cluster = o.value[k]
            o.buffer[k] = norm(x .- cluster.value)
        end
        k_star = argmin(o.buffer)
        cluster = o.value[k_star]
        smooth!(cluster.value, x, o.rate(cluster.n += 1))
    end
end

classify(o::KMeans, x) = findmin(c -> norm(x .- c.value), o.value)[2]



#classify(o::KMeans, x) = findmin(c -> norm(x .- c.value), o.value)[2]
=#

function cluster_distmat_online(distmat)
    labels = Vector{UInt32}([])
    distmat = copy(transpose(distmat)[:])
    ncentres = 10
    km = KMeans(ncentres)
    #row1 = view(distmat',:,1)
    row1 = distmat'[:,1]

    o = fit!(km,row1)
    
    @inbounds for (x,row) in enumerate(eachrow(distmat))
        if x>1
             o = fit!(km,row'[:])
        end
    end
    sort!(o)
    
    @inbounds for row in eachrow(distmat)
        push!(labels,Int32(classify(o,row'[:])))
    end
    labels
end    
function cluster_distmat!(mat_of_distances)

    #display(mat_of_distances)
    R = affinityprop(mat_of_distances)
    sort_idx =  sortperm(assignments(R))
    assign = R.assignments
    R,sort_idx,assign
end
#=
function call_inducingpoints()
        #alg = OIPS(ncentres) # We expect 200 inducing points
    #x_1 = [row for row in eachrow(distmat)][1] # We have some initial data
    #alg = StreamKmeans(ncentres)
    #Z = inducingpoints(alg,distmat)
    #Z = inducingpoints(alg, x_1)
    #for row in eachrow(distmat)
    #    updateZ!(Z, alg,row)
    #end

    #findmin(c -> norm(x .- c.value), o.value)
    #@show(Z)
    #@infiltrate
    #Z = inducingpoints(alg, x_1)
    #for x in eachbatch([row for row in eachrow(distmat)])
    #    updateZ!(Z, alg, x)
    #end
    #Z = inducingpoints(alg, X; kernel=kernel) # We create an initial vector
    #X_new = [rand(5) for _ in 1:50] # We get some new data
    #for row in eachrow(distmat)
    #    updateZ!(Z, alg, row; kernel=kernel) # Points will be acordingly added (or removed!)
    #end
    #@show(Z)
    #@infiltrate
end
=#
function horizontal_sort_into_tasks(distmat::AbstractMatrix,div_spike_mat_no_displacement::AbstractMatrix)
    distmat = copy(transpose(distmat))
    labels = Vector{UInt32}([])
    ncentres = 10
    km = KMeans(ncentres)
    o = fit!(km,distmat'[:,1])
    @inbounds for (x,row) in enumerate(eachrow(distmat))
        if x>1
             o = fit!(km,row'[:])
        end
    end
    sort!(o)    
    @inbounds for row in eachrow(distmat)
        push!(labels,Int32(classify(o,row'[:])))
    end
    #@show(typeof(distmat))
    #@show(size(distmat))
    use_labels(distmat,labels,div_spike_mat_no_displacement)
end
"""
Use and populate labels
"""
function use_labels(distmat::AbstractVecOrMat,labels::Vector{UInt32},div_spike_mat_no_displacement::AbstractMatrix)
    jobs = unique(labels)
    sub_jobs = Dict()
    @inbounds for label_ in unique(labels)
        sub_jobs[label_] = Vector{UInt32}()
    end
    @inbounds for j in jobs
        @inbounds for (i,label) in enumerate(labels)
            if label==j
                push!(sub_jobs[label],i)
            end

        end
    end
    Array_of_arraysV = Vector{Any}()
    Array_of_arraysS = Vector{Any}()

    @inbounds for j in jobs
        # zeros(NROWS,NCOLUMNS)
        newArrayVectors = zeros(length(sub_jobs[j]),size(distmat)[2])
        #newArraySpikes = zeros(length(sub_jobs[j]),size(div_spike_mat_no_displacement)[2])
        #newArraySpikes = zeros(size(div_spike_mat_no_displacement))
        newArraySpikes = div_spike_mat_no_displacement[sub_jobs[j],:]
        # @show(size(newArraySpikes))
        #Plots.scatter()
        newArrayVectors = distmat[sub_jobs[j],:]
        #@inbounds for (enum,index) in enumerate(sub_jobs[j])
        #    newArrayVectors[enum,:] = distmat[index,:]
        #end

        #@inbounds for (enum,index) in enumerate(sub_jobs[j])
        #    newArraySpikes[enum,:] = 
        #    @show(newArraySpikes[enum,:])
        #end

        #newArraySpikes[vec(mapslices(col -> any(col .!= 0), newArraySpikes, dims = 2)), :]
        push!(Array_of_arraysV,newArrayVectors)
        push!(Array_of_arraysS,newArraySpikes)

    end
    Array_of_arraysV::Vector{Any},labels::Vector{UInt32},Array_of_arraysS::Vector{Any}
end
#=
function horizontal_sort_into_tasks(mat_of_distances::AbstractArray)
    cluster_centres = Vector{Any}([])
    #km = KMeans(size(distmat)[2],size(distmat)[1])
    ncentres = 10
    km = KMeans(ncentres)

    o = fit!(km,distmat[:,1])

    for (x,col) in enumerate(eachcol(distmat))
        if x>1
            col = distmat[:,x]
            o = fit!(km,col)
        end
    end
    sort!(o)
    for i in 1:ncentres
        push!(cluster_centres,o.value[i])
        @show(o.value[i])
    end
    return cluster_centres,o
    #R = affinityprop(mat_of_distances')
    #sort_idx =  sortperm(assignments(R))
    #assign = R.assignments
    #R,sort_idx,assign
end
=#

function get_repeated_scatter(nlist,tlist,start_windows,end_windows,repeated_windows,nodes,times,nslices;file_name::String="empty.png")
    p=Plots.scatter()
    cnt=0
    xlimits=0
    @inbounds @showprogress for (ind,toi) in enumerate(end_windows)        
        if repeated_windows[ind]!=-1
            Tx = tlist[ind]
            if length(Tx)>1

                xlimits = maximum(Tx)
                Nx = nlist[ind]
                Plots.scatter!(p,Tx,Nx,legend = false, markercolor=Int(repeated_windows[ind]),markersize = 0.8,markerstrokewidth=0,alpha=0.8, bgcolor=:snow2, fontcolor=:blue, xlims=(0, xlimits))
                    #Plots.scatter!(p,Tx,Nx,legend = false, markercolor=Int(repeated_windows[ind]),markersize = 0.8,markerstrokewidth=0,alpha=0.8, bgcolor=:snow2, fontcolor=:blue, xlims=(0, xlimits))
                #end
                cnt+=1

                #Plots.vspan!(p,[minimum(Tx),maximum(Tx)], color=Int(repeated_windows[ind]),alpha=0.2)                   
            end 
        end

    end
    nunique = length(unique(repeated_windows))
    #if file_name
    Plots.scatter!(p,xlabel="time (ms)",ylabel="Neuron ID", yguidefontcolor=:black, xguidefontcolor=:black,title = "N observed states $nunique", xlims=(0, xlimits))

    p2 = Plots.scatter(times,nodes,legend = false, markersize = 0.8,markerstrokewidth=0,alpha=0.5, bgcolor=:snow2, fontcolor=:blue, xlims=(0, xlimits))
    Plots.scatter!(p2,xlabel="time (ms)",ylabel="Neuron ID", yguidefontcolor=:black, xguidefontcolor=:black,title = "Un-labeled spike raster", xlims=(0, xlimits))
    Plots.plot(p,p2, layout = (2, 1))            
    
    savefig("genuinely_repeated_pattern$file_name.png")

    #end
end
"""
A method to get collect the Inter Spike Intervals (ISIs) per neuron, and then to collect them together to get the ISI distribution for the whole cell population
Also output a ragged array (Array of unequal length array) of spike trains. 
"""
function create_spikes_ragged(nodes::Vector{UInt32},times::Vector{<:Real};plot=false)
    spikes_ragged = Vector{Any}([])
    numb_neurons = UInt32(maximum(nodes))+1 # Julia doesn't index at 0.
    @inbounds for n in 1:numb_neurons
        push!(spikes_ragged,[])
    end
    @inbounds @showprogress for (n,t) in zip(nodes,times)
        @inbounds for i in 1:numb_neurons
            if i==n
                push!(spikes_ragged[n],t)
            end
        end
    end
    (spikes_ragged::Vector{Any},numb_neurons::Int)
end

    #if plot
        #p1 = Plots.plot()
    #    @inbounds for (neuron_id,only_one_neuron_spike_times) in enumerate(spikes_ragged)
    #        nodes = [Int32(neuron_id) for (_,_) in enumerate(only_one_neuron_spike_times)]
    #        display(Plots.scatter!(p1,spikes_ragged[neuron_id],nodes,legend = false,xlabel="time (Seconds)",ylabel="Cell Id"))
    #    end
    #end

function cluster_distmat(mat_of_distances)
    R = affinityprop(mat_of_distances)
    sort_idx =  sortperm(assignments(R))
    assign = R.assignments
    R,sort_idx,assign
end


function ragged_to_lists(ragged)
    Nx = []
    Tx = []
    for row in eachrow(ragged)
        for (ind_cell,times) in enumerate(row)
            for tt in times
                for t in tt
                    push!(Tx,t) 
                    push!(Nx,ind_cell)
                end
            end
        end
    end
    (Nx,Tx)
end
function cluster_get_jobs(distmat,spikes_ragged)
    spike_jobs = Vector{Any}([])

    classes = 10
    R = kmeans(distmat', classes; maxiter=2000, display=:iter)
    a = assignments(R) # get the assignments of points to clusters
    sort_idx =  sortperm(assignments(R))
    for i in unique(a)
        push!(spike_jobs,spikes_ragged[a.==i])
    end
    spike_jobs
end 

#function failedAnalysis(d)
        #Array_of_arraysV,_,Array_of_arraysSpikes = horizontal_sort_into_tasks(distmat,div_spike_mat_no_displacement)
    #=
    for (job,row) in enumerate(Array_of_arraysSpikes)
        Nx = []
        Tx = []
        for (ind_cell,times) in enumerate(row)
            for tt in times
                for t in tt
                    push!(Tx,t) 
                    push!(Nx,ind_cell)
                end
            end
        end
        Plots.scatter(Tx,Nx,legend = false,markersize = 1.1,markerstrokewidth=0,alpha=0.7)#, bgcolor=:snow2, fontcolor=:blue, xlims=(0, xlimits))
        savefig("spikesCheck_$job"*".png")
        println("gets here $job")

    end


    for (job,(distmat,spike_packet)) in enumerate(zip(Array_of_arraysV,Array_of_arraysSpikes))
        Plots.heatmap(distmat)
        savefig("matrices_$job"*".png")
        sqr_distmat = label_exhuastively_distmat(distmat;threshold=similarity_threshold,disk=false)
        assign = cluster_distmat_online(distmat)    
        #assing_progressions,assing_progressions_times,assing_progressions_time_indexs = get_state_transitions(start_windows,end_windows,sqr_distmat,assign;threshold= similarity_threshold)
        #reassign_no_pattern_group!(assing_progressions)
        #row = spike_packet[:,column]
  
        #Plots.scatter(Tx,Nx)
        Plots.scatter(Tx,Nx,legend = false,markersize = 1.1,markerstrokewidth=0,alpha=0.7)#, bgcolor=:snow2, fontcolor=:blue, xlims=(0, xlimits))

        savefig("spikes_old$job"*".png")

        #internal_validation_dict(assign,spike_packet;file_path=plotsdir()*"$job"*".png")

        f=(spike_packet,distmat,sqr_distmat,assign)
        push!(list_of_jobs,f)
    end
    list_of_jobs
    =#


function doanalysis(d)
    @unpack nodes,times,dataset, window_size, similarity_threshold = d
    list_of_jobs = Vector{Any}([])
    numb_neurons = length(unique(nodes))
    maxt = maximum(times)
    (spikes_ragged,numb_neurons) = create_spikes_ragged(nodes,times)
    div_spike_mat_with_displacement,sws,ews = spike_matrix_divided(spikes_ragged,window_size,numb_neurons,maxt;displace=true)
    distmat = compute_metrics_on_matrix_divisions(div_spike_mat_with_displacement,metric="kreuz")
    #RR,xs, ys,sss,R = recurrence_mat(div_spike_mat_with_displacement)
        
    spike_jobs = cluster_get_jobs(distmat,spikes_ragged)
    distmats=[]
    template_index_dicts = []
    for (ind,spikes_ragged) in enumerate(spike_jobs)
        @show(length(spikes_ragged))
        div_spike_mat_with_displacement,sws,ews = spike_matrix_divided(spikes_ragged,window_size,numb_neurons,maxt;displace=true)
        new_distmat = compute_metrics_on_matrix_divisions(div_spike_mat_with_displacement,metric="kreuz")
        #@show(RR,xs, ys,sss,R)
        (sqr_distmat,rep,template_dict) = label_exhuastively_distmat(new_distmat,sws,ews,times)
        #@show(template_dict)
        push!(distmats,sqr_distmat)
        push!(template_index_dicts,template_dict)

        #@show(rep)
    end
    template_index_dicts
end

function reassign_no_pattern_group!(assing_progressions)
    for i in 1:length(assing_progressions) 
        if assing_progressions[i]==1
            assing_progressions[i]=-1
        end
    end
end
#=
function create_ISI_histogram(nodes,spikes)
    spikes = []
    numb_neurons=Int(maximum(nodes))+1
    @inbounds for n in 1:numb_neurons
        push!(spikes,[])
    end
    @inbounds @showprogress for (i, _) in enumerate(spikes)
        for (n,t) in zip(nodes,times)
            if i==n
                push!(spikes[i],t)
            end
        end
    end
    global_isis = []
    isi_s = []
    @inbounds @showprogress for (i, times) in enumerate(spikes)
        push!(isi_s,[])
        for (ind,x) in enumerate(times)
            if ind>1
                isi_current = x-times[ind-1]
                push!(isi_s[i],isi_current)

            end
        end
        append!(global_isis,isi_s[i])
    end
    global_isis
end
=#
"""
Using the windowed spike trains for neuron0: a uniform surrogate spike train reference, versus neuron1: a real spike train in the  target window.
compute the intergrated spike distance quantity in that time frame.

SpikeSynchrony is a Julia package for computing spike train distances just like elephant in python
And in every window I get the population state vector by comparing current window to uniform spiking windows
But it's also a good idea to use the networks most recently past windows as reference windows 

"""


#@inbounds @showprogress for (ind,toi) in enumerate(end_windows)
    #sw = start_windows[ind]
    # @inbounds for (ii,xx) in enumerate(R[ind,:])
#=
Nx=nlist[ys]
Tx=tlist[xs]
if abs(xx)<ε

    Plots.scatter!(p,Tx,Nx, markercolor=Int(assign[ii]),legend = false, markersize = 0.70,markerstrokewidth=0,alpha=1.0, bgcolor=:snow2, fontcolor=:blue)
end
#end
#end
#nunique = length(unique(witnessed_unique))
nunique = length(nzval(R))
Plots.scatter!(p,xlabel="time (ms)",ylabel="Neuron ID", yguidefontcolor=:black, xguidefontcolor=:black,title = "N observed states $nunique")
p2 = Plots.scatter(times,nodes,legend = false, markersize =0.5,markerstrokewidth=0,alpha=0.5, bgcolor=:snow2, fontcolor=:blue,thickness_scaling = 1)
Plots.scatter!(p2,xlabel="time (ms)",ylabel="Neuron ID", yguidefontcolor=:black, xguidefontcolor=:black,title = "Un-labeled spike raster")

Plots.plot(p,p2, layout = (2, 1))

savefig("identified_unique_pattern_via_recurrence$file_name.png")
=#
         #


function get_division_scatter_identify(nlist,tlist,start_windows,end_windows,distmat,assign,nodes,times,repeated_windows;file_name::String="empty.png",threshold::Real=5)
    p=Plots.scatter()
    witnessed_unique=[]
    @inbounds @showprogress for (ind,toi) in enumerate(end_windows)
        sw = start_windows[ind]
        Nx=nlist[ind]
        Tx=tlist[ind]
        @inbounds for (ii,xx) in enumerate(distmat[ind,:])
            if abs(xx)<threshold
                push!(witnessed_unique,assign[ii])
                #div_spike_mat[ind,ii]
                Plots.scatter!(p,Tx,Nx, markercolor=Int(assign[ii]),legend = false, markersize = 0.70,markerstrokewidth=0,alpha=1.0, bgcolor=:snow2, fontcolor=:blue)

                #Plots.scatter!(p,div_spike_mat[ind,ii],[ii], markercolor=Int(assign[ii]),legend = false, markersize = 0.70,markerstrokewidth=0,alpha=1.0, bgcolor=:snow2, fontcolor=:blue)
                #if length(Tx)>1
                #    Plots.vspan!(p,[minimum(Tx),maximum(Tx)], color=Int(repeated_windows[ind]),alpha=0.2)                   
                #end 
            end
            
        end
    end
    nunique = length(unique(witnessed_unique))
    Plots.scatter!(p,xlabel="time (ms)",ylabel="Neuron ID", yguidefontcolor=:black, xguidefontcolor=:black,title = "N observed states $nunique")
    p2 = Plots.scatter(times,nodes,legend = false, markersize =0.7,markerstrokewidth=0,alpha=0.5, bgcolor=:snow2, fontcolor=:blue)
    Plots.scatter!(p2,xlabel="time (ms)",ylabel="Neuron ID", yguidefontcolor=:black, xguidefontcolor=:black,title = "Un-labeled spike raster")

    Plots.plot(p,p2, layout = (2, 1))

    savefig("identified_unique_pattern_normal$file_name.png")
end


function get_division_scatter_identify2(div_spike_mat,nlist,tlist,start_windows,end_windows,distmat,assign,nodes,times,repeated_windows;file_name::String="empty.png",threshold::Real=5)
    p=Plots.scatter()
    witnessed_unique=[]
    @inbounds @showprogress for (ind,toi) in enumerate(end_windows)
        sw = start_windows[ind]
        Nx=nlist[ind]
        Tx=tlist[ind]
        @inbounds for (ii,xx) in enumerate(distmat[ind,:])
            if abs(xx)<threshold
                push!(witnessed_unique,assign[ii])
                #div_spike_mat[ind,ii]
                #Plots.scatter!(p,Tx,Nx, markercolor=Int(assign[ii]),legend = false, markersize = 0.70,markerstrokewidth=0,alpha=1.0, bgcolor=:snow2, fontcolor=:blue)
                #temp = sw.+div_spike_mat[x,y]
                #@show(temp)
                #if length(div_spike_mat[ind,ii])!=0
                #if ind!=ii
                temp = div_spike_mat[ii,ind][1] #.+sw
                #temp = div_spike_mat[ind,ii][1] #.+sw


                Plots.scatter!(p,temp,[ii], markercolor=Int(assign[ii]),legend = false, markersize = 0.70,markerstrokewidth=0,alpha=1.0, bgcolor=:snow2, fontcolor=:blue,ylim=(0,maximum(nodes)))
            #end
                    #if length(Tx)>1
                #    Plots.vspan!(p,[minimum(Tx),maximum(Tx)], color=Int(repeated_windows[ind]),alpha=0.2)                   
                #end 
            end
            
        end
    end
    nunique = length(unique(witnessed_unique))
    Plots.scatter!(p,xlabel="time (ms)",ylabel="Neuron ID", yguidefontcolor=:black, xguidefontcolor=:black,title = "N observed states $nunique")
    p2 = Plots.scatter(times,nodes,legend = false, markersize =0.7,markerstrokewidth=0,alpha=0.5, bgcolor=:snow2, fontcolor=:blue)
    Plots.scatter!(p2,xlabel="time (ms)",ylabel="Neuron ID", yguidefontcolor=:black, xguidefontcolor=:black,title = "Un-labeled spike raster")

    Plots.plot(p,p2, layout = (2, 1))

    savefig("identified_unique_pattern2$file_name.png")
end

function get_division_via_recurrence(R,xs, ys,sss,div_spike_mat,start_windows;file_name="nothing.png")
    p=Plots.scatter()
    for (x,row) in enumerate(eachrow(R))
        for (y,logic_val) in enumerate(row)
            #@show(y)
            #for (y,x) in zip(xs,ys)
            sw = start_windows[x]
            #@show(sw)
            #@show(div_spike_mat[x,y])
            #val_comp = R[x,y]
            #if val_comp!=0.0
                #if length(div_spike_mat[x,y])!=0
            #temp = [i for i in 1:length(div_spike_mat[x,y])]
            if logic_val

                temp = div_spike_mat[x,y][1] #.+sw
                #@show(temp)
                Plots.scatter!(p,temp,[y],xlabel="time (ms)",ylabel="Neuron ID", markercolor=Int(x), markersize = 0.70,markerstrokewidth=0,alpha=1.0, bgcolor=:snow2, yguidefontcolor=:black, xguidefontcolor=:black,title = "Recurrence-labeled spike raster",legend=false)
                #last +=sw #maximum(div_spike_mat[ind,ii])

                #end
            end
        end
    end
    savefig("recurrence_identified_unique_pattern$file_name.png")
end


function get_state_transitions(start_windows,end_windows,distmat,assign;threshold::Real=5)
    nunique = length(unique(assign))
    assing_progressions=[]
    assing_progressions_times=[]
    assing_progressions_time_indexs=[]

    @assert size(distmat)[1]==length(start_windows)
    @inbounds for (xi,row) in enumerate(eachrow(distmat))
        @inbounds for (ii,xx) in enumerate(row)
            if abs(xx)<threshold
                sw = start_windows[ii]#+end_windows[ii]
                push!(assing_progressions,assign[ii])
                push!(assing_progressions_times,sw)
                push!(assing_progressions_time_indexs,ii)
    
            end
        end
    end
    assing_progressions[unique(i -> assing_progressions[i], 1:length(assing_progressions))].=-1

    assing_progressions,assing_progressions_times,assing_progressions_time_indexs
end



function return_spike_item_from_matrix(div_spike_mat_no_displacement::AbstractVecOrMat,ind::Integer)
    Nx = Float32[]
    Tx = Float32[]
    for (indy,row) in enumerate(div_spike_mat_no_displacement[:,ind])
        for (indx,x) in enumerate(row)
            if length(x)!=0
                append!(Nx,indy)
                append!(Tx,x)                
            end
        end
    end
    (Nx::Vector{Float32},Tx::Vector{Float32})
end
function state_transition_trajectory(start_windows,end_windows,distmat,assign,assing_progressions,assing_progressions_times;plot=false, file_name::String="stateTransMat.png")

    repeated_windows = Vector{UInt32}([])

    nunique = length(unique(assign))
    stateTransMat = zeros(nunique,nunique)
    @inbounds for (ind,x) in enumerate(assing_progressions)
        if ind < length(assing_progressions)
            stateTransMat[x,assing_progressions[ind+1]]+=1
        end 
    end
    repititions = zeros(size(stateTransMat))
    @inbounds for (ind,_) in enumerate(eachrow(stateTransMat))
        @inbounds for (y,val) in enumerate(stateTransMat[ind,:])
            if val==1
                repititions[ind,y] = 0.0

            elseif val>1
                repititions[ind,y] = val 
                push!(repeated_windows,y)

            end 
        end
    end
    
    assing_progressions[unique(i -> assing_progressions[i], 1:length(assing_progressions))].=-1
    if plot

        p1 = Plots.plot()
        Plots.scatter!(p1,assing_progressions_times,assing_progressions,markercolor=assing_progressions,legend=false)
        #https://github.com/open-risk/transitionMatrix

        savefig("state_transition_trajectory$file_name.png")

        (n,m) = size(stateTransMat)
        
        #cols = columns(stateTransMat)
        Plots.heatmap(cor(stateTransMat), fc=cgrad([:white,:dodgerblue4]), xticks=(1:n,m), xrot=90, yticks=(1:m,n), yflip=true)
        Plots.annotate!([(j, i, (round(stateTransMat[i,j],digits=3), 8,"Computer Modern",:black)) for i in 1:n for j in 1:m])
        #Plots.heatmap(stateTransMat)
        savefig("corr_state_transition_matrix$file_name.png")
        Plots.heatmap(stateTransMat, fc=cgrad([:white,:dodgerblue4]), xticks=(1:n,m), xrot=90, yticks=(1:m,n), yflip=true)
        savefig("state_transition_matrix$file_name.png")
    
    end
 
    repeated_windows
end


function simple_plot_umap(mat_of_distances; file_name::String="empty.png")
    #model = UMAP_(mat_of_distances', 10)
    #Q_embedding = transform(model, amatrix')
    #cs1 = ColorScheme(distinguishable_colors(length(ll), transform=protanopic))

    Q_embedding = umap(mat_of_distances',20,n_neighbors=20)#, min_dist=0.01, n_epochs=100)
    display(Plots.plot(Plots.scatter(Q_embedding[1,:], Q_embedding[2,:], title="Spike Time Distance UMAP, reduced precision", marker=(1, 1, :auto, stroke(0.07)),legend=true)))
    #Plots.plot(scatter!(p,model.knns)
    savefig(file_name)
    Q_embedding
end

function sort_by_row(distmat,nodes,times,resolution,numb_neurons,maxt,spikes)
    horizontalR = kmeans(distmat,3)
    horizontalR_sort_idx =  sortperm(assignments(horizontalR))
    spikes = spikes[horizontalR_sort_idx]
    nodes=Vector{UInt32}([])
    times=Vector{Float32}([])

    @inbounds @showprogress for (i, t) in enumerate(spikes)
        @inbounds for tt in t
            if length(t)!=0
                push!(nodes,i)
                push!(times,Float32(tt))
            end
        end
    end
    (distmat,tlist,nlist,start_windows,end_windows,spike_distance_size) = get_divisions(nodes,times,resolution,numb_neurons,maxt,plot=false,metric="kreuz")
end

#=
function get_ts(nodes,times)
    #num_spikes = length(nodes)
    # The temporal resolution of the final timesurface
    dt = 10
    num_neurons = Int(length(unique(nodes)))+1#int(df.max(axis=0)['x1'])
    total_time =  Int(maximum(times))
    time_resolution = Int(round(total_time/dt))
    # Final output. 
    final_timesurf = zeros((num_neurons, time_resolution+1))
    # Timestamp and membrane voltage store for generating time surface
    timestamps = zeros((num_neurons)) .- Inf
    mv = zeros((num_neurons))
    tau = 200
    last_t = 0
    for (tt,nn) in zip(times,nodes)
        #Get the current spike
        neuron = Int(nn) 
        time = Int(tt)        
        # If time of the next spikes leaps over, make sure to generate 
        # timesurfaces for all the intermediate dt time intervals and fill the 
        # final_timesurface.
        if time > last_t
            timesurf = similar(final_timesurf[:,1])
            for t in collect(last_t:dt:time)
                @. timesurf = mv*exp((timestamps-t)/tau)
                final_timesurf[:,1+Int(round(t/dt))] = timesurf
            end
            last_t = time
        end
        # Update the membrane voltage of the time surface based on the last value and time elapsed
        mv[neuron] =mv[neuron]*exp((timestamps[neuron]-time)/tau) +1
        timestamps[neuron] = time
        # Update the latest timestamp at the channel. 
    end
    # Generate the time surface for the rest of the time if there exists no other spikes. 
    timesurf = similar(final_timesurf[:,1])
    for t in collect(last_t:dt:total_time)
        @. timesurf = mv*exp((timestamps-t)/tau)
        final_timesurf[:,1+Int(round(t/dt))] = timesurf
    end
    return final_timesurf

end
=#
"""
Divide epoch into windows.
"""
#=
function divide_epoch(nodes::AbstractVector,times::AbstractVector,sw::Real,toi::Real)
    #t1=[]
    #n1=[]
    t0=Vector{Float32}([])
    n0=Vector{UInt32}([])
    @assert sw< toi
    third = toi-sw
    @inbounds for (n,t) in zip(nodes,times)
        if sw<=t && t<=toi+third
            append!(t0,abs(t-toi))
            append!(n0,n)            
        elseif t>=toi && t<=toi+third
            append!(t0,abs(t-toi))
            append!(n0,n)
        end
    end
    time_raster =  Array{}([Vector{Float32}([]) for i in 1:maximum(nodes)+1])
    for (neuron,t) in zip(n0,t0)
        append!(time_raster[neuron],t)        
    end
    #=
    #time_raster =  Array{}([SVector{Float32}([]) for i in 1:maximum(nodes)+1])
    
    static_fast = Array{SVector}([])
    for neuron in time_raster
        temp = SVector{length(neuron0[neuron]),Float32}(time_raster[neuron])
        push!(static_fast,temp)
    end
    #neuron0
    =#
    #static_fast
    time_raster
end
=#

#=
function divide_epoch(nodes::SVector,times::SVector,sw::Real,toi::Real)
    t0=Vector{Float32}([])
    n0=Vector{UInt32}([])
    #@assert sw< toi
    third = toi-sw
    @inbounds for (n,t) in zip(nodes,times)
        if sw<=t && t<=toi+third
            append!(t0,abs(t-toi))
            append!(n0,n)            
        elseif t>=toi && t<=toi+third
            append!(t0,abs(t-toi))
            append!(n0,n)
        end
    end
    neuron0 =  Array{}([Float32[] for i in 1:maximum(nodes)+1])
    for (neuron,t) in zip(n0,t0)
        append!(neuron0[neuron],t)        
    end
    neuron0
end
=#
#=
function divide_epoch(nodes::SVector,times::SVector,sw::Real,toi::Real)
    t1=Vector{Float32}([])
    t0=Vector{Float32}([])
    n1=Vector{UInt32}([])
    n0=Vector{UInt32}([])
    @assert sw< toi
    third = toi-sw
    for (n,t) in zip(nodes,times)
        if sw<=t && t<toi
            append!(t0,t-sw)
            append!(n0,n)            
        elseif t>=toi && t<=toi+third
            append!(t1,abs(t-toi))
            @assert t-toi>=0
            append!(n1,n)
        end
    end
    neuron0 =  Array{}([Float32[] for i in 1:maximum(nodes)+1])
    for (neuron,t) in zip(n0,t0)
        append!(neuron0[neuron],t)        
    end
    neuron0
end
=#


#=
function get_vector_coords(neuron0::Vector{Vector{Float32}}, neuron1::Vector{Vector{Float32}}, self_distances::Vector{Float32};metric="kreuz")
    for (ind,(n0_,n1_)) in enumerate(zip(neuron0,neuron1))        
        if length(n0_) != 0 && length(n1_) != 0
            pooledspikes = vcat(n0_,n1_)
            maxt = maximum(sort!(unique(pooledspikes)))
            t1_ = sort(unique(n0_))
            t0_ = sort(unique(n1_))
            if metric=="kreuz"
                t, S = SPIKE_distance_profile(t0_,t1_;t0=0,tf = maxt)
                self_distances[ind]=abs(sum(S))
            elseif metric=="CV"
                if length(t1_)>1
                    self_distances[ind] = CV(t1_)
                    @show(self_distances[ind])
                else
                    self_distances[ind]=0
                end


            elseif metric=="autocov"
                if length(t1_)>1
                    self_distances[ind] = autocov( t1_, [1],demean=true)[1]
                else
                    self_distances[ind]=0
                end

            elseif metric=="LV"
                self_distances[ind]=lvr(time_intervals, [1], 5*0.001)#std(t0_)/mean(t0_)
            end
        else
            self_distances[ind]=0
        end
    end
    self_distances
end
#get_vector_coords_uniform!(::SVector{2, Float64}, ::Vector{SVector}, ::Vector{Float32})
=#
#using StatsBase
#=
function get_vector_coords_uniform!(uniform::Vector{Float32}, neuron1::Vector{Vector{Float32}}, self_distances::Vector{Float32})
    @inbounds for (ind,n1_) in enumerate(neuron1)
        if length(n1_) != 0
            pooledspikes = vcat(uniform,n1_)
            maxt = maximum(sort!(unique(pooledspikes)))
            t0_ = sort(unique(n1_))
            t, S = SPIKE_distance_profile(t0_,uniform;t0=0,tf = maxt)
            self_distances[ind]=abs(sum(S))
        else
            self_distances[ind]=0
        end

    end
end

function get_vector_coords_uniform!(uniform::Vector{Float32}, neuron1::Vector{Any}, self_distances::Vector{Float32})
    @inbounds for (ind,n1_) in enumerate(neuron1)
        if length(n1_) != 0
            pooledspikes = vcat(uniform,n1_)
            maxt = maximum(sort!(unique(pooledspikes)))
            t0_ = sort(unique(n1_))
            t, S = SPIKE_distance_profile(t0_,uniform;t0=0,tf = maxt)
            self_distances[ind]=abs(sum(S))
        else
            self_distances[ind]=0
        end
    end
end
=#
#=
function final_plots2(mat_of_distances)
    angles0,distances0,angles1,distances1 = post_proc_viz(mat_of_distances)
    plot!(angles1,distances1,marker =:circle, arrow=(:closed, 3.0)) 
    savefig("statemvements_nmn.png")   
    (angles1,distances1)
end
=#
