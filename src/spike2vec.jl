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
#using UMAP
using Distances
using StaticArrays
using Clustering
using StatsBase
#import OnlineStatsBase.CircBuff
using Infiltrator
#using RecurrenceAnalysis
using Statistics
import LinearAlgebra.normalize!
import LinearAlgebra.norm
#using ProfileView
#using Cthulhu
#using StatProfilerHTML

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
    #time_raster =  Array{}([Float32[] for i in 1:maximum(nodes)+1])
    #for (neuron,t) in zip(n0,t0)
    #    append!(time_raster[neuron],t)        
    #end
    #time_raster,
    n0::Vector{UInt32},t0::Vector{Float32}
end



function divide_epoch(times::AbstractVector,start::Real,stop::Real)
    t0=Vector{Float32}([])
    @inbounds for t in times
        if start<=t && t<stop
                @assert start<=t<stop
                push!(t0,t)
            #end
        end
    end
    t0::Vector{Float32}
end

#function get_vector_coords_uniform!(one_neuron_surrogate::AbstractArray, spikes_ragged::AbstractArray, self_distances::AbstractArray;metric=:kreuz)
function get_vector_coords_uniform!(one_neuron_surrogate::AbstractArray, neurons_obs::Vector{Vector{Float32}}, self_distances::AbstractArray;metric=:kreuz)
    @inbounds for (ind,n1_) in enumerate(neurons_obs)
        if length(n1_) > 0 && length(one_neuron_surrogate) > 0
            pooledspikes = copy(one_neuron_surrogate)
            append!(pooledspikes,n1_)
            maxt = maximum(sort!(unique(pooledspikes)))
            t1_ = sort(unique(n1_))
            if length(t1_)>1

                if metric==:kreuz
                    _, S = SPIKE_distance_profile(t1_,one_neuron_surrogate;t0=0,tf = maxt)
                    self_distances[ind] = abs(sum(S))
                elseif metric==:CV
                        self_distances[ind] = CV(t1_)
                elseif metric==:autocov
                        self_distances[ind] = autocov( t1_, [length(t1_)-1],demean=true)[1]

                elseif metric==:LV

                        self_distances[ind] = lvr(t1_,maximum(t1_))
                elseif metric==:hybrid
                        _, S = SPIKE_distance_profile(t1_,one_neuron_surrogate;t0=0,tf = maxt)
                        self_distances[ind] = abs(sum(S))
                        self_distances[ind] += lvr(t1_,maximum(t1_))
                        self_distances[ind] += sum(t1_)

                elseif metric==:count
                        self_distances[ind] = sum(t1_)
                end

            else # If no spikes in this window, don't reflect that there was agreement
                 # ie self_distances[ind] = 0, reflects agreement
                 # reflect a big disagreement.
                self_distances[ind] = abs(length(n1_)-length(t1_))
            end
        end
    end
end
#=
function get_vector_coords_uniform!(one_neuron_surrogate::AbstractArray, spikes_ragged::AbstractArray, self_distances::AbstractArray;metric=:kreuz)

    @inbounds for (ind,times_obs) in enumerate(spikes_ragged)
        if length(times_obs) > 0 && length(one_neuron_surrogate) > 0
            pooledspikes = copy(one_neuron_surrogate)
            append!(pooledspikes,times_obs)
            maxt = maximum(sort!(unique(pooledspikes)))
            t1_ = sort(unique(times_obs))
            if length(t1_)>1

                if metric==:kreuz
                    _, S = SPIKE_distance_profile(t1_,one_neuron_surrogate;t0=0,tf = maxt)
                    self_distances[ind] = abs(sum(S))
                elseif metric==:CV
                        self_distances[ind] = CV(t1_)
                elseif metric==:autocov
                        self_distances[ind] = autocov( t1_, [length(t1_)-1],demean=true)[1]

                elseif metric==:LV

                        self_distances[ind] = lvr(t1_,maximum(t1_))
                elseif metric==:hybrid
                        _, S = SPIKE_distance_profile(t1_,one_neuron_surrogate;t0=0,tf = maxt)
                        self_distances[ind] = abs(sum(S))
                        self_distances[ind] += lvr(t1_,maximum(t1_))
                        self_distances[ind] += sum(t1_)

                elseif metric==:count
                        self_distances[ind] = sum(t1_)
                end

            else # If no spikes in this window, don't reflect that there was agreement
                 # ie self_distances[ind] = 0, reflects agreement
                 # reflect a big disagreement.
                self_distances[ind] = abs(length(spikes_ragged[1])-length(t1_))
            end
        end
    end
end
=#
function array_of_empty_vectors(T, dims...)
    array = Array{Vector{T}}(undef, dims...)
    for i in eachindex(array)
        array[i] = Vector{T}()
    end
    array
end

function make_sliding_window(start_windows,end_windows,step_size)
    full_sliding_window_starts = Vector{Float32}([])
    full_sliding_window_ends = Vector{Float32}([])
    offset = 0.0
    @inbounds for (start,stop) in zip(start_windows,end_windows)
        @inbounds for _ in 1:5
            push!(full_sliding_window_starts,start+offset) 
            push!(full_sliding_window_ends,stop+offset)
            offset+=step_size/5.0
        end
    end
    (full_sliding_window_starts,full_sliding_window_ends)
end

function spike_matrix_slices(nodes::Vector{UInt32},times::Vector{Float32},number_divisions_size::Int,maxt::Real)
    step_size = maxt/number_divisions_size
    full_sliding_window_ends = Vector{Float32}(collect(step_size:step_size:step_size*number_divisions_size))
    full_sliding_window_starts = Vector{Float32}(collect(0:step_size:(step_size*number_divisions_size)-step_size))
    nodes_per_slice = Vector{Any}([])
    times_per_slice = Vector{Any}([])

    ##
    # To implement the sliding window
    ##
    #@inbounds for (neuron_id,only_one_neuron_spike_times) in enumerate(spikes_raster)
    for (windex,stop) in enumerate(full_sliding_window_ends)
        sw = full_sliding_window_starts[windex]
        push!(times_per_slice,[])
        push!(nodes_per_slice,[])
        n0,t0 = divide_epoch(nodes,times,sw,stop)
        times_per_slice[windex] = t0
        nodes_per_slice[windex] = n0
    end
    times_per_slice::AbstractVecOrMat,nodes_per_slice::AbstractVecOrMat,full_sliding_window_starts::Vector{Float32},full_sliding_window_ends::Vector{Float32}
end
"""
sw: start window a length used to filter spikes.
windex: current window index
times_associated: times (associated with) indices, before stop window length applied
"""
function spike_matrix_divided(spikes_raster::AbstractVecOrMat,number_divisions_size::Int,maxt::Real;displace=true)
    step_size = maxt/number_divisions_size
    full_sliding_window_ends = Vector{Float32}(collect(step_size:step_size:step_size*number_divisions_size))
    full_sliding_window_starts = Vector{Float32}(collect(0:step_size:(step_size*number_divisions_size)-step_size))


    ##
    # To implement the sliding window
    ##
    
    #@time (full_sliding_window_starts,full_sliding_window_ends) = make_sliding_window(full_sliding_window_starts,full_sliding_window_ends,step_size)
    Ncells = length(spikes_raster)
    mat_of_spikes = array_of_empty_vectors(Vector{Float32},(Ncells,length(full_sliding_window_ends)))
    
    spike_matrix_divided!(mat_of_spikes,spikes_raster,full_sliding_window_ends,full_sliding_window_starts,displace)
    @assert sum(sum(sum(mat_of_spikes[:,1:5])))>0.0

    mat_of_spikes::Matrix{Vector{Vector{Float32}}},full_sliding_window_starts::Vector{Float32},full_sliding_window_ends::Vector{Float32}
end
function spike_matrix_divided!(mat_of_spikes::Matrix{Vector{Vector{Float32}}},spikes_raster,end_windows,start_windows,displace)
    @inbounds for (neuron_id,only_one_neuron_spike_times) in enumerate(spikes_raster)
        @inbounds for (windex,end_window_time) in enumerate(end_windows)
            sw = start_windows[windex]
            observed_spikes = divide_epoch(only_one_neuron_spike_times,sw,end_window_time)
            @show(only_one_neuron_spike_times)
            @show(observed_spikes)
            @infiltrate
            if !displace
                push!(mat_of_spikes[neuron_id,windex],observed_spikes)

            else
                @show(observed_spikes)
                @show(observed_spikes.-sw)
                push!(mat_of_spikes[neuron_id,windex],observed_spikes.-sw)

            end
        end
    end
    @assert sum(sum(sum(mat_of_spikes[:,1:5])))>0.0

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


function compute_metrics_on_matrix_divisions(div_spike_mat_no_displacement::Matrix{Vector{Vector{Float32}}},NCells,NodeList,timesList;metric=:kreuz,disk=false)
    #(nrow::UInt32,ncol::UInt32)=size(div_spike_mat_no_displacement)
    (nrow::UInt32,ncol::UInt32)=size(div_spike_mat_no_displacement)
    mat_of_distances = Array{Float32}(undef, nrow, ncol)
    
    #mat_of_distances = Array{Float32}(undef, NCells, length(timesList))
    max_spk_countst = Int32(trunc(maximum([length(times) for times in timesList])))
    #@infiltrate
    maximum_time = maximum(maximum(timesList)) #maximum([times for times in timesList])[1]
    temp = LinRange(0.0, maximum_time, max_spk_countst)
    linear_uniform_spikes = Vector{Float32}([i for i in temp[:]])
    
    sum_var = compute_metrics_on_slices!(NCells,NodeList,timesList,mat_of_distances,linear_uniform_spikes;metric=metric)    
    (mat_of_distances::Array{Float32},sum_var::Float32)
end



function compute_metrics_on_matrix_divisions(div_spike_mat_no_displacement::Matrix{Vector{Vector{Float32}}},timesList;metric=:kreuz,disk=false)
    (nrow::UInt32,ncol::UInt32)=size(div_spike_mat_no_displacement)
    mat_of_distances = Array{Float32}(undef, nrow, ncol)

    max_spk_countst = Int32(trunc(maximum([length(times) for times in timesList])))
    maximum_time = maximum(maximum(timesList)) #maximum([times for times in timesList])[1]

    #max_spk_countst = Int32(trunc(maximum([length(times) for times in enumerate(div_spike_mat_no_displacement)])))
    #maximum_time = maximum([times[2][1] for times in enumerate(div_spike_mat_no_displacement)])[1]
    
    temp = LinRange(0.0, maximum_time, max_spk_countst)
    linear_uniform_spikes = Vector{Float32}([i for i in temp[:]])

    sum_var = compute_metrics_on_matrix_divisions!(div_spike_mat_no_displacement,mat_of_distances,linear_uniform_spikes,timesList,nrow;metric=metric)    
    (mat_of_distances::Array{Float32},sum_var::Float32)
end

function compute_metrics_on_matrix_self_past_divisions(div_spike_mat_no_displacement::Matrix{Vector{Vector{Float32}}};disk=false)
    (nrow::UInt32,ncol::UInt32)=size(div_spike_mat_no_displacement)
    mat_of_distances = Array{Float64}(undef, nrow, ncol)

    compute_metrics_on_matrix_self_past_divisions!(div_spike_mat_no_displacement,mat_of_distances)    
    (mat_of_distances::Array{Float64})
end
function compute_metrics_on_matrix_self_past_divisions!(div_spike_mat_no_displacement::Matrix{Vector{Vector{Float32}}},mat_of_distances::Array{Float64})
    (nrow::UInt32,ncol::UInt32)=size(div_spike_mat_no_displacement)
    sum_varr::Float32=0.0
    neurons_old = div_spike_mat_no_displacement[:,1]
    @inbounds for (indc,neurons) in enumerate(eachcol(div_spike_mat_no_displacement))
        neurons = neurons[1]
        if indc!=1 && length(neurons_old)>0  && length(neurons)>0
            self_distances = Vector{Float32}(zeros(nrow))
            get_vector_coords_uniform!(neurons_old, neurons, self_distances; metric=:kreuz)
            mat_of_distances[:,indc] = copy(self_distances)
            #neurons_old = neurons


        end
        neurons_old = copy(neurons)

    end
    #mat_of_distances[isnan.(mat_of_distances)] .= maximum(mat_of_distances[!isnan.(mat_of_distances)])
    mat_of_distances = copy(transpose(mat_of_distances))
    normalize!(mat_of_distances)
    @assert norm(mat_of_distances)==1
    mat_of_distances = copy(transpose(mat_of_distances))

    
    @inbounds for row in eachrow(mat_of_distances) sum_varr+=var(row) end
    sum_varr
end


function compute_metrics_on_matrix_divisions!(div_spike_mat_no_displacement::Matrix{Vector{Vector{Float32}}},mat_of_distances::Array{<:Real},linear_uniform_spikes::Vector{Float32},timesList,nrow::UInt32;metric=:kreuz)
    sum_varr::Float32=0.0

    @inbounds for (indc,neuron_times) in enumerate(eachcol(div_spike_mat_no_displacement))
        #@show(typeof(neuron_times))

        self_distances = Vector{Float32}(zeros(nrow))
        if !isa(neuron_times[1], Number)
            neuron_times = [ n[1] for n in neuron_times ]
        end

        get_vector_coords_uniform!(linear_uniform_spikes, neuron_times, self_distances; metric=metric)
        mat_of_distances[:,indc] = copy(self_distances)
    end
    normalize!(mat_of_distances)
    #@assert norm(mat_of_distances)==1
    sum_varr=0
    @inbounds for row in eachrow(mat_of_distances)
        sum_varr+=var(row)
    end
    sum_varr
end



function compute_metrics_on_slices!(Ncells,nodesList::Vector{Any},timesList::Vector{Any},mat_of_distances::Array{<:Real},linear_uniform_spikes::Vector{Float32};metric=:kreuz)
    sum_varr::Float32=0.0

    @inbounds for (indc,times) in enumerate(timesList)
        self_distances = Vector{Float32}(zeros(Ncells))
        #display(Plots.scatter(times,nodesList[indc]))
        nodes = nodesList[indc]
        if length(nodes)>0
            (spikes_ragged,numb_neurons) = create_spikes_ragged(nodes,times) 

        else
            spikes_ragged = [[0],[0]]
            get_vector_coords_uniform!(linear_uniform_spikes, spikes_ragged, self_distances; metric=metric)
        end
        mat_of_distances[:,indc] = self_distances
    end

    #create_spikes_ragged(times,nodes)
    
    #display(Plots.heatmap(mat_of_distances))
    sum_varr=0
    @inbounds for row in eachrow(mat_of_distances)
        #@show(var(row))
        sum_varr+=var(row)
    end
    sum_varr
    @show(sum_varr)
end

function compute_metrics_on_divisions(division_size::Integer,numb_neurons::Integer,maxt::Real;plot=false,file_name="stateTransMat.png",metric=:kreuz,disk=false)

    spike_distance_size = length(end_windows)
    mat_of_distances = Array{Float64}(undef, numb_neurons, spike_distance_size)
    nlist = Array{Vector{UInt32}}([])
    tlist = Array{Vector{Float32}}([])
    sum_varr = 0.0
    (_,_,spike_distance_size,sum_varr) = compute_metrics_on_divisions!(mat_of_distances::Array{Float64},nlist::Array{Vector{UInt32}},tlist::Array{Vector{Float32}},nodes::Vector{UInt32},times::Vector{<:Real},division_size::Integer,numb_neurons::Integer,maxt::Real;sum_varr=sum_varr,plot=false,file_name="stateTransMat.png",metric=:kreuz,disk=false)
    (mat_of_distances::Array{Float64},nlist::Array{Vector{UInt32}},tlist::Array{Vector{Float32}},sum_varr::Float32)
end

function compute_metrics_on_divisions!(mat_of_distances::Array{Float64},nodes::Vector{UInt32},times::Vector{<:Real},division_size::Integer,numb_neurons::Integer,maxt::Real;sum_varr=nothing,metric=:kreuz,disk=false)
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
    #mat_of_distances[isnan.(mat_of_distances)] .= 0.0
    #normalize!(mat_of_distances)
    #@assert norm(mat_of_distances)==1



    sum_varr=0
    @inbounds for row in eachrow(mat_of_distances)
        sum_varr+=var(row)
    end
    sum_varc=0
    @inbounds for col in eachcol(mat_of_distances)
        sum_varc+=var(col)
    end   

end

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
        #@show(o.value[i])
    end
    return cluster_centres,o
    #R = affinityprop(mat_of_distances')
    #sort_idx =  sortperm(assignments(R))
    #assign = R.assignments
    #R,sort_idx,assign
end


function templates_using_cluster_centres!(mat_of_distances::AbstractVecOrMat,distance_matrix::AbstractVecOrMat,sws,ews,times,nodes,div_spike_mat_with_displacement;threshold::Real=5)
    #cnts_total = 0.0
    classes = 10
    R = kmeans(mat_of_distances, classes; maxiter=1000, display=:iter)
    #a = assignments(R) # get the assignments of points to clusters
    #sort_idx =  sortperm(assignments(R))
    centres = R.centers # get the cluster centers
    #@infiltrate
    NURS = 0.0
    template_times_dict = Dict()
    template_nodes_dict = Dict()
    state_frequency_histogram = Vector{Int32}(zeros(length(mat_of_distances)))

    @inbounds for (ind2,template) in enumerate(eachcol(centres))


        ##
        ## TODO build time windows here!
        ##
        ##
        @inbounds for (ind,row) in enumerate(eachcol(mat_of_distances))
    
            distance = evaluate(Euclidean(),row,template)
            if distance<threshold
                state_frequency_histogram[ind2]+=1

                NURS += 1.0
                distance_matrix[ind,ind2] = abs(distance)
                if !(haskey(template_times_dict, ind))
                    template_times_dict[ind] = []
                    template_nodes_dict[ind] = []
                end
                #div_spike_mat_with_displacement[:,ind]
                (n0,t0) = divide_epoch(nodes,times,sws[ind],ews[ind])
                push!(template_times_dict[ind],t0)
                push!(template_nodes_dict[ind],n0)

            end
        end
    end
    number_windows = length(eachcol(mat_of_distances))
    window_duration = last(ews)-last(sws)
    repeatitive = NURS/(number_windows*window_duration)
    (repeatitive,template_times_dict,template_nodes_dict,NURS::Real,state_frequency_histogram)
end


function label_exhuastively_distmat!(mat_of_distances::AbstractVecOrMat,distance_matrix::AbstractVecOrMat,sws,ews,times,nodes;threshold::Real=5)
    #cnts_total = 0.0
    NURS = 0.0
    template_times_dict = Dict()
    template_nodes_dict = Dict()

    #nodes,times = ragged_to_lists(spikes_ragged)

    @inbounds for (ind,row) in enumerate(eachcol(mat_of_distances))
        #stop_at_half = Int(trunc(length(eachcol(mat_of_distances))/2))
        #if ind <= stop_at_half

        ##
        ## TODO build time windows here!
        ##
        ##
        @inbounds for (ind2,row2) in enumerate(eachcol(mat_of_distances))
            if ind!=ind2
                distance = evaluate(Euclidean(),row,row2)
                if distance<threshold
                    NURS += 1.0
                    distance_matrix[ind,ind2] = abs(distance)
                    if !(haskey(template_times_dict, ind))
                        template_times_dict[ind] = []
                        template_nodes_dict[ind] = []
                    end
                    (n0,t0) = divide_epoch(nodes,times,sws[ind2],ews[ind2])
                    push!(template_times_dict[ind],t0)
                    push!(template_nodes_dict[ind],n0)

                end
            end
        end
    end
    number_windows = length(eachcol(mat_of_distances))
    window_duration = last(ews)-last(sws)
    repeatitive = NURS/(number_windows*window_duration)
    (repeatitive,template_times_dict,template_nodes_dict,NURS::Real)
end
"""
label_exhuastively_distmat!
"""
function label_exhuastively_distmat(mat_of_distances::AbstractVecOrMat,sws,ews,times,nodes,div_spike_mat_with_displacement;threshold::Real=5, disk=false)
    if !disk
        distance_matrix = zeros(length(eachcol(mat_of_distances)),length(eachcol(mat_of_distances)))
    else
        io = open("/tmp/mmap.bin", "w+")
        distance_matrix = mmap(io, Matrix{Float32}, (length(eachcol(mat_of_distances)),length(eachcol(mat_of_distances))))
    end
    repeatitive,dict0,dict1,NURS,state_frequency_histogram = templates_using_cluster_centres!(mat_of_distances,distance_matrix,sws,ews,times,nodes,div_spike_mat_with_displacement;threshold)
    #repeatitive,dict0,dict1,NURS = label_exhuastively_distmat!(mat_of_distances,distance_matrix,sws,ews,times;threshold)
    distance_matrix::AbstractVecOrMat,repeatitive::Real,dict0::Dict,dict1::Dict,NURS::Real,state_frequency_histogram
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
        newArrayVectors = zeros(length(sub_jobs[j]),size(distmat)[2])
        newArraySpikes = div_spike_mat_no_displacement[sub_jobs[j],:]
        newArrayVectors = distmat[sub_jobs[j],:]
        push!(Array_of_arraysV,newArrayVectors)
        push!(Array_of_arraysS,newArraySpikes)

    end
    Array_of_arraysV::AbstractVecOrMat,labels::Vector{UInt32},Array_of_arraysS::AbstractVecOrMat
end

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
function create_spikes_ragged(nodes::Vector{UInt32},times::Vector{<:Real})
    spikes_ragged = Vector{Any}([])
    numb_neurons = UInt32(maximum(nodes)) # Julia doesn't index at 0.
    @inbounds for n in 1:numb_neurons
        push!(spikes_ragged,Vector{Float32}([]))
    end
    @inbounds for (n,t) in zip(nodes,times)
        @inbounds for i in 1:numb_neurons
            if i==n
                push!(spikes_ragged[UInt32(n)],t)
            end
        end
    end
    (spikes_ragged::AbstractVecOrMat,numb_neurons::UInt32)
end



function cluster_distmat(mat_of_distances)
    R = affinityprop(mat_of_distances)
    sort_idx =  sortperm(assignments(R))
    assign = R.assignments
    R,sort_idx,assign
end

function ragged_to_lists(ragged_array::AbstractVecOrMat)
    Nx=Vector{UInt32}([])
    Tx=Vector{Float32}([])
    @inbounds for (i, t) in enumerate(ragged_array)
        @inbounds for tt in t
            push!(Nx,i)
            push!(Tx,tt)
        end
    end
    (Nx::AbstractVecOrMat,Tx::AbstractVecOrMat)
end
#=
function ragged_to_lists(ragged::AbstractVecOrMat)
    Nx = Vector{UInt32}([])
    Tx = Vector{Float32}([])
    @inbounds for row in eachrow(ragged)
        @inbounds for (ind_cell,times) in enumerate(row)
            @inbounds for tt in times
                @inbounds for t in tt
                    push!(Tx,t) 
                    push!(Nx,ind_cell)
                end
            end
        end
    end
    (Nx::Vector{UInt32}, Tx::Vector{Float32})
end
=#
function cluster_get_jobs(distmat,spikes_ragged)
    spike_jobs = Vector{Any}([])

    classes = 10
    R = kmeans(distmat', classes; maxiter=2000, display=:iter)
    a = assignments(R) # get the assignments of points to clusters
    sort_idx =  sortperm(assignments(R))
    for i in unique(a)
        push!(spike_jobs,spikes_ragged[a.==i])
    end
    #println("done")
    spike_jobs,sort_idx
end 

function do_memory_intensive_plot(p1,offset,template_time_dict,template_node_dict)    
    temp_node_offset=[]
    temp_time = []
    told =0
    IStateI = Vector{Float32}([])
    offset=0.0
    
 
    @inbounds for (t0,v1) in zip(values(template_time_dict),values(template_node_dict))
        @inbounds for vmx in v1 
            push!(temp_node_offset,vmx.+offset)
        end
        @inbounds for t in t0
            push!(IStateI,abs(mean(t)-told))
            told=mean(t)
            push!(temp_time,t)
        end
        Plots.scatter!(p1,temp_time,temp_node_offset,legend = false,markersize = 0.7,markerstrokewidth=0,alpha=0.7)
    end 
end                

function doanalysis(d)
    @unpack nodes,times, window_size, similarity_threshold = d
    sfs=Vector{Any}([])
    sum_of_rep=0.0
    NURS_sum=0.0
    Ncells = length(unique(nodes))
    maxt = maximum(times)
    times_per_slice,nodes_per_slice,full_sliding_window_starts,full_sliding_window_ends = spike_matrix_slices(nodes,times,window_size,maxt)
    @assert length(nodes)>0.0
    @assert length(times)>0.0
    (spikes_ragged,numb_neurons) = create_spikes_ragged(nodes,times) 

    #@infiltrate
    div_spike_mat_with_displacement,sws,ews = spike_matrix_divided(spikes_ragged, window_size, maxt ;displace=true)
    @assert sum(div_spike_mat_with_displacement[:,1:5])>0.0
    @assert size(div_spike_mat_with_displacement)[2] == window_size
    @assert size(div_spike_mat_with_displacement)[1] == Ncells
    @infiltrate
    (distmat,k_variance) = compute_metrics_on_matrix_divisions(div_spike_mat_with_displacement,times_per_slice,metric=:count)
    Plots.heatmap(distmat)
    ylabel!("Number cells: $Ncells")
    xlabel!("Number of windows: $window_size")
    savefig("bug_fixed.png")
    spike_jobs,sort_idx = cluster_get_jobs(distmat,spikes_ragged)

    #=
    @show(k_variance)
    Plots.heatmap(distmat)
    savefig("fixed_code_Kreuz.png")

    (distmat,lv_variance) = compute_metrics_on_matrix_divisions(Ncells,nodes_per_slice,times_per_slice,metric=:LV)
    @show(lv_variance)
    Plots.heatmap(distmat)
    savefig("fixed_code_LV.png")

    (distmat,count_variance) = compute_metrics_on_matrix_divisions(Ncells,nodes_per_slice,times_per_slice,metric=:count)
    @show(count_variance)
    Plots.heatmap(distmat)
    savefig("fixed_code_count.png")

    (distmat,hybrid_variance) = compute_metrics_on_matrix_divisions(Ncells,nodes_per_slice,times_per_slice,metric=:hybrid)
    @show(hybrid_variance)
    @assert length(nodes)>0.0
    =#
                                              #spike_matrix_divided(spikes_raster::AbstractVecOrMat,number_divisions_size::Int,maxt::Real;displace=true)
    #println("gets here")
    #@time (distmat,variance) = compute_metrics_on_matrix_divisions(div_spike_mat_with_displacement,metric=:kreuz)
    #list_of_template_time_dicts = Vector{Any}([])
    #list_of_template_node_dicts = Vector{Any}([])


    @inbounds for (_,spikes_ragged_packet) in enumerate(spike_jobs)
        
        div_spike_mat_with_displacement,sws,ews = spike_matrix_divided(spikes_ragged_packet,window_size,maxt;displace=true)
        nodes,times = ragged_to_lists(spikes_ragged_packet)
        times_per_slice,nodes_per_slice,full_sliding_window_starts,full_sliding_window_ends = spike_matrix_slices(nodes,times,window_size,maxt)
        (new_distmat,_) = compute_metrics_on_matrix_divisions(div_spike_mat_with_displacement,times_per_slice,metric=:kreuz)

        (_,rep,_,_,NURS,state_frequency_histogram) = label_exhuastively_distmat(new_distmat,sws,ews,times,nodes,div_spike_mat_with_displacement)
        #@infiltrate
        @show(state_frequency_histogram)
        push!(sfs,state_frequency_histogram)
        sum_of_rep+=rep
        NURS_sum+=NURS
        #push!(distmats,sqr_distmat)
        #@show(state_frequency_histogram)
        #push!(list_of_template_time_dicts,template_time_dict)
        #push!(list_of_template_node_dicts,template_node_dict)
        #offset += UInt32(length(spikes_ragged_packet))   
    end
    
    #p1 = nothing
    ##
    # Remember distmat is transposed
    ##
    (distmat,div_spike_mat_with_displacement,spikes_ragged,sort_idx,NURS_sum,sum_of_rep,sfs)
end

function more_plotting1(spikes_ragged,sort_idx,IStateI,p1)

    #(nx,tx)=ragged_to_lists(spikes_ragged[sort_idx])
    #p2 = Plots.scatter(tx,nx,legend = false,markersize = 0.7,markerstrokewidth=0,alpha=0.7)
    IStateI[isnan.(IStateI)] .= 0.0
    b_range = range(0, maximum(IStateI), length=21)
    p3 = Plots.histogram(IStateI,legend=false, bins=b_range, normalize=:pdf, color=:gray)
    title!("Inter State Intervals")
    ylabel!("Occurances")
    xlabel!("Intervals")
    #p4=Plots.scatter()


    Plots.plot(p1,p3, layout = (2, 1))
    savefig("scatterp_single.png")
end

function more_plotting0(distmat,div_spike_mat_with_displacement,nodes,times)
    sum_of_variabilities = Vector{Float32}([])
    rates = Vector{Float32}([])

    positions = [ cnt for (cnt,_) in enumerate(eachcol(distmat)) ]
    @inbounds for (cnt,col) in enumerate(eachcol(distmat))
        push!(sum_of_variabilities,sum(col))
    end

    ##
    # Yet div_spike_mat_with_displacement is not transposed
    ##
    @inbounds for (cnt,col) in enumerate(eachcol(div_spike_mat_with_displacement))
        running_sum = 0.0
        @inbounds for j in col
            running_sum+=size(j[1])[1]
        end
        push!(rates,running_sum)        
    end
    positions = [ cnt for (cnt,_) in enumerate(eachcol(div_spike_mat_with_displacement)) ]
    p2 = Plots.plot()
    Plots.scatter!(p2,times,nodes,legend = false,markersize = 0.8,markerstrokewidth=0,alpha=0.8, fontcolor=:blue,xlabel="time (Seconds)",ylabel="Cell Id")

    #p2 = Plots.scatter(nodes,times,legend=false)
    p3 = Plots.heatmap(distmat)
    p4 = Plots.plot(positions,sum_of_variabilities,legend=false)    
    ylabel!("Local Variation")
    xlabel!("Time Window")
    p5 = Plots.plot(positions,rates,legend=false)    
    ylabel!("Firing Rate")
    xlabel!("Time Window")
    Plots.plot(p4, p5,p3,p2,layout = (4, 1))
    savefig("saved_content.png")    

end
function reassign_no_pattern_group!(assing_progressions)
    for i in 1:length(assing_progressions) 
        if assing_progressions[i]==1
            assing_progressions[i]=-1
        end
    end
end
"""
Using the windowed spike trains for neuron0: a uniform surrogate spike train reference, versus neuron1: a real spike train in the  target window.
compute the intergrated spike distance quantity in that time frame.

SpikeSynchrony is a Julia package for computing spike train distances just like elephant in python
And in every window I get the population state vector by comparing current window to uniform spiking windows
But it's also a good idea to use the networks most recently past windows as reference windows 

"""

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
            sw = start_windows[x]
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
    (distmat,tlist,nlist,start_windows,end_windows,spike_distance_size) = get_divisions(nodes,times,resolution,numb_neurons,maxt,plot=false,metric=:kreuz)
end

