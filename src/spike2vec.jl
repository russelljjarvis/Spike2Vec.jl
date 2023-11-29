#
# Eventually this will need to move to source.
# 
using KernelDensity

using HDF5
using Plots
Plots.gr(fmt=:png)
using OnlineStats
#using Plots
using JLD2
using SpikeSynchrony
using LinearAlgebra
using Revise
#using StatsBase
using ProgressMeter
#using LinearAlgebra
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
#using DataFrames
#using MatrixProfile, Plots
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

function kreuz_diff(t1_::Vector{Float32},one_neuron_surrogate::Vector{Float32},difference::Int,window_size::Real,maxt::Real)
    if difference==-1
        append!(one_neuron_surrogate,window_size/2)
        one_neuron_surrogate = sort(unique(one_neuron_surrogate))
    elseif difference==1
        append!(t1_,window_size/2)
        t1_ = sort(unique(t1_))
    end
    if abs(difference)>1
        temp = abs(difference)+1
        temp = LinRange(minimum(one_neuron_surrogate), window_size,temp)
        linear_uniform_spikes = Vector{Float32}([i for i in temp[:]])
        if difference>0.0
            #surrogate bigger
            # put extra spikes in sample
            append!(one_neuron_surrogate,linear_uniform_spikes)
            one_neuron_surrogate = sort(unique(one_neuron_surrogate))
        elseif difference < 0.0
            #sample/observation bigger
            # put extra spikes in surrogate

            append!(t1_,linear_uniform_spikes)
            t1_ = sort(unique(t1_))
        
        elseif difference == 0
            one_neuron_surrogate = sort(unique(one_neuron_surrogate))
        end
        #@show(length(one_neuron_surrogate),length(t1_))
    
        #@assert length(one_neuron_surrogate)==length(t1_)
    
    end
    _, S = SPIKE_distance_profile(t1_,one_neuron_surrogate;t0=0,tf = maxt)
    S
end
"""
On a RTSP packet, get a CV. 
"""
function CV(spikes_1d_vector::AbstractArray)
    cv = 0.0
    if length(spikes_1d_vector)>1
        isi_s = Float32[] # the total lumped population ISI distribution.        
        @inbounds for (ind,x) in enumerate(spikes_1d_vector)
            if ind>1
                isi_current = x-spikes_1d_vector[ind-1]
                push!(isi_s,isi_current)
            end
        end
        if length(spikes_1d_vector)>1
            cv = std(isi_s)/mean(isi_s)
        else
            cv = 0.0
        end
    end
    cv
end

function get_vector_coords_uniform!(one_neuron_surrogate::AbstractArray, neurons_obs::Vector{Vector{Float32}}, self_distances::AbstractArray,window_size;metric=:kreuz)
    @inbounds for (ind,n1_) in enumerate(neurons_obs)
        if length(n1_) > 0 #&& length(one_neuron_surrogate) > 0
            pooledspikes = copy(one_neuron_surrogate)
            append!(pooledspikes,n1_)
            maxt = maximum(sort(unique(pooledspikes)))
            t1_ = sort(unique(n1_))
            if length(t1_)>1

                if metric==:kreuz
                    #difference = length(t1_)-length(one_neuron_surrogate)
                    _, S = SPIKE_distance_profile(t1_,one_neuron_surrogate;t0=0,tf = maxt)
                    #self_distances[ind] = abs(sum(S))
                    #S = kreuz_diff(sort(t1_),sort(one_neuron_surrogate),difference,window_size,maxt)
                    self_distances[ind] = abs(sum(S))-sum(t1_)
                    
                elseif metric==:CV
                        @assert length(t1_)>0.0
                        self_distances[ind] = CV(t1_)
                        #@show(self_distances)
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
    #display(Plots.plot(self_distances))
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
    array = Array{Vector{T}}(undef,dims...)
    for i in eachindex(array)
        array[i] = Vector{T}()
    end
    array
end

function make_sliding_window(start_windows,end_windows,step_size)
    full_sliding_window_starts = Vector{Float32}([])
    full_sliding_window_ends = Vector{Float32}([])
    offset = 0.0
    ending=length(end_windows)
    @inbounds for (ind,(start,stop)) in enumerate(zip(start_windows,end_windows))
        if ind!=ending
            @inbounds for _ in 1:2
                push!(full_sliding_window_starts,start+offset) 
                push!(full_sliding_window_ends,stop+offset)
                offset=step_size/2.0
            end
        else
            push!(full_sliding_window_starts,start) 
            push!(full_sliding_window_ends,stop)
        end
    end
    #@show(last(full_sliding_window_ends),last(end_windows))
    @assert last(full_sliding_window_ends) == last(end_windows)
    (full_sliding_window_starts,full_sliding_window_ends)
end

function spike_matrix_slices(nodes::Vector{UInt32},times::Vector{Float32},number_divisions::Int,maxt::Real,start_time::Real)

    step_size = (maxt-start_time)/number_divisions
    full_sliding_window_ends = Vector{Float32}(collect(start_time+step_size:step_size:start_time+step_size*number_divisions))
    full_sliding_window_starts = Vector{Float32}(collect(start_time:step_size:start_time+(step_size*number_divisions)-step_size))
    @assert length(full_sliding_window_starts) == number_divisions


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
function spike_matrix_divided(spikes_raster::AbstractVecOrMat,step_size::Real,number_divisions::Real,maxt::Real,start_time;displace=true,sliding=true)

    step_size = (maxt-start_time)/number_divisions
    full_sliding_window_ends = Vector{Float32}(collect(start_time+step_size:step_size:start_time+step_size*number_divisions))
    full_sliding_window_starts = Vector{Float32}(collect(start_time:step_size:start_time+(step_size*number_divisions)-step_size))
    @assert length(full_sliding_window_starts) == number_divisions
    ##
    # To implement the sliding window
    ##
    if sliding    
        (full_sliding_window_starts,full_sliding_window_ends) = make_sliding_window(full_sliding_window_starts,full_sliding_window_ends,step_size)
    end
    Ncells = length(spikes_raster)
    mat_of_spikes = array_of_empty_vectors(Vector{Float32},(Ncells,length(full_sliding_window_ends)))
    
    spike_matrix_divided!(mat_of_spikes,spikes_raster,full_sliding_window_ends,full_sliding_window_starts,displace)
 
    #@assert size(mat_of_spikes)[2]==number_divisions

    mat_of_spikes::Matrix{Vector{Vector{Float32}}},full_sliding_window_starts::Vector{Float32},full_sliding_window_ends::Vector{Float32},step_size
end
function spike_matrix_divided!(mat_of_spikes::Matrix{Vector{Vector{Float32}}},spikes_raster,end_windows,start_windows,displace)
    @inbounds for (neuron_id,only_one_neuron_spike_times) in enumerate(spikes_raster)
        @inbounds for (windex,end_window_time) in enumerate(end_windows)
            sw = start_windows[windex]
            observed_spikes = divide_epoch(only_one_neuron_spike_times,sw,end_window_time)
            if !displace
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

#=
function compute_metrics_on_matrix_divisions(div_spike_mat_no_displacement::Matrix{Vector{Vector{Float32}}},NCells,NodeList,timesList;metric=:kreuz,disk=false)
    #(nrow::UInt32,ncol::UInt32)=size(div_spike_mat_no_displacement)
    
    (nrow::UInt32,ncol::UInt32)=size(div_spike_mat_no_displacement)
    #println("get s here")
    mat_of_distances = Array{Float32}(0.0, nrow, ncol)
    
    #mat_of_distances = Array{Float32}(undef, NCells, length(timesList))
    max_spk_countst = Int32(trunc(maximum([length(times) for times in timesList])))
    #@infiltrate
    maximum_time = maximum(maximum(timesList)) #maximum([times for times in timesList])[1]
    temp = LinRange(0.0, maximum_time, max_spk_countst)
    linear_uniform_spikes = Vector{Float32}([i for i in temp[:]])
    
    sum_var = compute_metrics_on_slices!(NCells,NodeList,timesList,mat_of_distances,linear_uniform_spikes;metric=metric)    
    (mat_of_distances::Array{Float32},sum_var::Float32)
end
=#
function get_var_of_mat(mat_of_distances::AbstractVecOrMat)
    sum_varr=0
    @inbounds for row in eachrow(mat_of_distances)
        sum_varr+=var(row)
    end
    sum_varr::Real
end


function compute_metrics_on_matrix_divisions(div_spike_mat_no_displacement::Matrix{Vector{Vector{Float32}}},ews,step_size;metric=:kreuz,disk=false)
    (nrow::UInt32,ncol::UInt32)=size(div_spike_mat_no_displacement)
    mat_of_distances = Array{Float32}(undef, nrow, ncol)
    @. mat_of_distances = 0.0
    compute_metrics_on_matrix_divisions!(div_spike_mat_no_displacement,mat_of_distances,nrow,ews,step_size;metric=metric)    
    normalize!(mat_of_distances)
    sum_var = get_var_of_mat(mat_of_distances)
    (mat_of_distances::Array{Float32},sum_var::Float32)
end

function compute_metrics_on_matrix_self_past_divisions(div_spike_mat_no_displacement::Matrix{Vector{Vector{Float32}}};disk=false)
    (nrow::UInt32,ncol::UInt32)=size(div_spike_mat_no_displacement)
    mat_of_distances = Array{Float64}(undef, nrow, ncol)
    @. mat_of_distances = 0.0
    compute_metrics_on_matrix_self_past_divisions!(div_spike_mat_no_displacement,mat_of_distances)    
    (mat_of_distances::Array{Float64})
end
#=
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
=#
#compute_metrics_on_matrix_divisions!(div_spike_mat_no_displacement,mat_of_distances,timesList,nrow,ews;metric=metric)    

function compute_metrics_on_matrix_divisions!(div_spike_mat_no_displacement::Matrix{Vector{Vector{Float32}}},mat_of_distances::Array{<:Real},nrow::UInt32,ews::Vector{Float32},window_size::Real;metric=:kreuz)
    sum_varr::Float32=0.0
    list_of_spike_counts = []
    @inbounds for (indc,neuron_times) in enumerate(eachcol(div_spike_mat_no_displacement))
        for row in neuron_times
            push!(list_of_spike_counts,length(row[1]))
        end
    end    
    @inbounds for (indc,neuron_times) in enumerate(eachcol(div_spike_mat_no_displacement))
        self_distances = Vector{Float32}(zeros(nrow))
        if !isa(neuron_times[1], Number)
            neuron_times = [ n[1] for n in neuron_times ]
        end
        #@show(list_of_spike_counts)
        temp = Int(trunc(round(maximum(list_of_spike_counts))))
        temp = LinRange(0.0, window_size,temp)
        linear_uniform_spikes = Vector{Float32}([i for i in temp[:]])
    
        get_vector_coords_uniform!(linear_uniform_spikes, neuron_times, self_distances,window_size; metric=metric)
        mat_of_distances[:,indc] = self_distances
    end
end



function compute_metrics_on_slices!(Ncells,nodesList::Vector{Any},timesList::Vector{Any},mat_of_distances::Array{<:Real},linear_uniform_spikes::Vector{Float32};metric=:kreuz)
    sum_varr::Float32=0.0
    @inbounds for (indc,times) in enumerate(timesList)
        self_distances = Vector{Float32}(zeros(Ncells))
        nodes = nodesList[indc]
        if length(nodes)>0
            (spikes_ragged,numb_neurons) = create_spikes_ragged(nodes,times) 
        else
            spikes_ragged = [[]]
            get_vector_coords_uniform!(linear_uniform_spikes, spikes_ragged, self_distances; metric=metric)
        end
        mat_of_distances[:,indc] = self_distances
    end
    sum_varr=0
    @inbounds for row in eachrow(mat_of_distances)
        sum_varr+=var(row)
    end
    sum_varr
end

function compute_metrics_on_divisions(division_size::Integer,numb_neurons::Integer,maxt::Real;plot=false,file_name="stateTransMat.png",metric=:kreuz,disk=false)

    spike_distance_size = length(end_windows)
    mat_of_distances = Array{Float64}(undef, numb_neurons, spike_distance_size)
    @. mat_of_distances = 0.0

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


function templates_using_cluster_centres!(mat_of_distances::AbstractVecOrMat,distance_matrix::AbstractVecOrMat,sws,ews,times,nodes,div_spike_mat_with_displacement,times_per_slice,nodes_per_slice,indlabel;threshold::Real=5)
    #cnts_total = 0.0
    classes = 10
    R = kmeans(mat_of_distances', classes; maxiter=1000, display=:iter)
    #a = assignments(R) # get the assignments of points to clusters
    sort_idx =  sortperm(assignments(R))
    #p3=Plots.heatmap(mat_of_distances[sort_idx,:])
    #savefig("sortedHeatmap0.png")
    p2=Plots.heatmap(mat_of_distances[:,sort_idx])
    #savefig("sortedHeatmap1.png")
    centres = R.centers # get the cluster centers
    p1=Plots.heatmap(centres)
    Plots.plot(p1,p2)
    savefig("Centres.png")
    

    #@infiltrate
    NURS = 0.0
    template_times_dict = Dict()
    template_nodes_dict = Dict()
    state_frequency_histogram = Vector{Int32}(zeros(length(mat_of_distances)))
    number_windows = length(eachcol(mat_of_distances))
    window_duration = last(ews)-last(sws)
    repeatitive = NURS/(number_windows*window_duration)
    distances = []
    @inbounds for (ind2,template) in enumerate(eachcol(centres))
        @inbounds for (ind,col) in enumerate(eachcol(mat_of_distances))    
            distance = evaluate(Euclidean(),col,template)
            push!(distances,distance)
            #@show(distance,threshold)
            if distance<threshold
                state_frequency_histogram[ind2]+=1

                NURS += 1.0
                distance_matrix[ind,ind2] = abs(distance)
                if !(haskey(template_times_dict, ind2))
                    template_times_dict[ind2] = []
                    template_nodes_dict[ind2] = []
                end
                #(t0,n0) = times_per_slice[ind],nodes_per_slice[ind]
                (n0,t0) = divide_epoch(nodes,times,sws[ind],ews[ind])
                #append!(n0x,n0)
                #@show(unique([n0,n0x]))
                push!(template_times_dict[ind2],t0)
                push!(template_nodes_dict[ind2],n0)
            end
        end
    end
    Plots.plot(distances)
    savefig("distance_distribution.png")

    ###
    # Figure left panel
    ###
    p1= Plots.scatter()
    maxt=0.0
    for (k,v) in pairs(template_nodes_dict)
        tx=template_times_dict[k][1].+maxt
        nx=template_nodes_dict[k][1]
        Plots.scatter!(p1,tx,nx,legend=false)
        maxt=maximum(tx)
    end
    savefig("left_panel_templates.png")

    for (k,v) in pairs(template_nodes_dict)
        p2 = Plots.scatter()
        #maxt=0.0
        cnt=0
        for i in 1:length(template_times_dict[k][i])
            tx=template_times_dict[k][i]#.+maxt
            nx=template_nodes_dict[k][i]
            cnt+=1
            Plots.scatter!(p2,tx,nx,legend=false,color=cnt)
            # maxt=maximum(tx)
        end
        savefig("template_instances$k.png")

    end

    @inbounds for (ind,(nodes_list,times_list)) in enumerate(zip(values(template_nodes_dict),values(template_times_dict)))
        p1=Plots.plot()
        cnt=0
        @inbounds for (nodes,times) in zip(nodes_list,times_list)
            cnt+=1
            if length(unique(nodes))>1
                Plots.scatter!(p1,times,nodes,legend=false)
            end

            savefig("template$indlabel.$ind.$cnt.png")

        end
    end
    (repeatitive,template_times_dict,template_nodes_dict,NURS::Real,state_frequency_histogram)
end
function get_proto_templates(mat_of_distances::AbstractVecOrMat,enrich::AbstractVecOrMat,threshold,distance_distributions::Vector,nodes::Vector,times::Vector,ews,sws)
    #indold = -1
    #indold2 = -1
    ##
    # Survey the whole population for repititions.
    ##
    @inbounds for (template_ind,col) in enumerate(eachcol(mat_of_distances))
        @inbounds for (ind2,col2) in enumerate(eachcol(mat_of_distances))
            if template_ind>ind2
                distance = evaluate(Euclidean(),col,col2)
                if distance<threshold
                    (repn0,_) = divide_epoch(nodes,times,sws[Int32(template_ind)],ews[Int32(template_ind)])
                    (repn1,_) = divide_epoch(nodes,times,sws[Int32(ind2)],ews[Int32(ind2)])
                    if  issetequal(repn0, repn1)
                        #@show(issetequal(repn0, repn1))
                        push!(distance_distributions,distance)
                    end
                end
            end
        end
    end
    new_thresh = mean(distance_distributions)-std(distance_distributions)
    @inbounds for (template_ind,col) in enumerate(eachcol(mat_of_distances))
        @inbounds for (ind2,col2) in enumerate(eachcol(mat_of_distances))
            if template_ind>ind2
                distance = evaluate(Euclidean(),col,col2)
                if distance<new_thresh
                    (repn0,_) = divide_epoch(nodes,times,sws[Int32(template_ind)],ews[Int32(template_ind)])
                    (repn1,_) = divide_epoch(nodes,times,sws[Int32(ind2)],ews[Int32(ind2)])
                    if issetequal(repn0, repn1)
                        push!(distance_distributions,distance)
                        enrich[:,template_ind] = col
                        enrich[:,ind2] = col2
                    end
                end
            end
        end
    end
    (enrich)
end
#=
function refine_templates(mat_of_distances,List_of_templates,threshold,distance_distributions)
    @inbounds for template_ind1 in List_of_templates
        @inbounds for template_ind0 in List_of_templates
            if template_ind1!=template_ind0
 
                distance = evaluate(Euclidean(),mat_of_distances[:,template_ind1],mat_of_distances[:,template_ind0])
                push!(distance_distributions,distance)

                if distance==0#threshold

                    deleteat!(List_of_templates, List_of_templates .== template_ind0)
                end
            end
        end
    end
    List_of_templates,distance_distributions
end
=#

function cluster(enrich)
    enrich = enrich[:, vec(mapslices(col -> any(col .!= 0), enrich, dims = 1))]
    #k = length(unique(List_of_templates))
    #display(Plots.heatmap(enrich))
    #k = Int(trunc((3/4)*size(enrich)[1]))
    #@show(k)
    #@show(size(enrich))
    #@show(k)
    #@infiltrate
    
    #@show(k)
    @show(size(enrich))
    R = kmeans(enrich, 1; maxiter=1000, display=:iter)
    a = assignments(R) # get the assignments of points to clusters
    sort_idx =  sortperm(assignments(R))
    enrich = enrich[:,sort_idx]
    centres = R.centers # get the cluster centers
    template_distances = []
    for (template_ind,col) in enumerate(eachcol(centres))
        @inbounds for (time_ind,col2) in enumerate(eachcol(centres))
            if time_ind>template_ind
                distance = evaluate(Euclidean(),col,col2)
                push!(template_distances,distance)
            end
        end

    end
    new_threshold = mean(template_distances) - std(template_distances)/3
    centres,new_threshold
end
function compare_heatmaps(repn,rept,repn_old,rept_old)#,x,state_time)
    #if length(rept_old)>1

        #rows_to_keep = A[:,10] .!= 0
        min_time = minimum([minimum(rept_old),minimum(rept)])
        min_node = minimum([minimum(repn),minimum(repn_old)])
        max_time = maximum([maximum(rept_old),maximum(rept)])
        max_node = maximum([maximum(repn),maximum(repn_old)])

        o0 = HeatMap(min_node:1:max_node, min_time:1/length(rept):max_time)
        fit!(o0, zip(repn, rept))
        ts0 = copy(o0.counts)
        t0 = ts0[:, vec(mapslices(col -> any(col .!= 0), ts0, dims = 1))]
        o1 = HeatMap(min_node:1:max_node, min_time:1/length(rept_old):max_time)
        fit!(o1, zip(repn_old, rept_old))                    
        ts1 = copy(o1.counts)
        t1 = ts1[:, vec(mapslices(col -> any(col .!= 0), ts1, dims = 1))]


        #Difference_between_heats = sum(t0.-t1)
        #@show(Difference_between_heats)
        #@infiltrate
        #=
        if length(ts0)>1
            p2=Plots.heatmap(t0,legend=false)
            p3=Plots.heatmap(t1,legend=false)
            layout = @layout [a ; b ]
            Plots.plot(p2, p3, layout=layout,legend=false)
            savefig("blah$x$state_time.png")
        end
        =#
    #end
    t1,t0
end

function plot_same_category(state_time,state_number,nodes,times,sws,ews,skip_times)
    #window = ews[1]-sws[1]
    ##
    # Same category plots
    ##
    #pa = Plots.plot()
    for x in unique(state_number)
        pz = Plots.plot()
        rept_old = []
        repn_old = []
        for (state_time,sn) in zip(state_time,state_number)
            #@show(state_time,sn)
            if !(state_time in(skip_times))
                if x == sn

                    (repn,rept) = divide_epoch(nodes,times,sws[Int32(state_time)],ews[Int32(state_time)])
                    #rept = rept.-state_time
                    #@show(issetequal(repn, repn_old))
                    
                    
                    Plots.scatter!(pz,rept,repn,markersize=2.0,markerstrokewidth=0,color="green",legend=false)
                    #Plots.plot!(pz,rept,repn,markersize=2.0,markerstrokewidth=0,color="green",legend=false)
                    if length(rept)>1
                        vline!(pz,[minimum(rept), maximum(rept)],fill=true,alpha=0.5,colors=["green","green"])
                    end
                    savefig("category$x.png")

                    #compare_heatmaps(repn,rept,repn_old,rept_old,x,state_time)
                    rept_old = rept
                    repn_old = repn

                end
            end

        end
        #=
        py = Plots.plot()
        for (state_time,sn) in zip(state_time,state_number)
            if state_time in(skip_times) || x != sn
                (repn,rept) = divide_epoch(nodes,times,sws[Int32(state_time)],ews[Int32(state_time)])
                rept = rept.-state_time
                Plots.scatter!(py,rept,repn,markersize=2.0,markerstrokewidth=0,color="blue",legend=false)
                #Plots.plot!(pz,rept,repn,markersize=2.0,markerstrokewidth=0,color="green",legend=false)
                if length(rept)>1
                    vline!(pz,[minimum(rept), maximum(rept)],fill=true,alpha=0.5,colors=["blue","blue"])
                end                    
            end
        #end
        end
        Plots.plot(pz,py,layout=(2,1),size=(1000,1000))
        =#
    end
end
function CompareObsVsCentresWorst(centres,mat_of_distances,new_threshold,fit,tmd,state_time,state_number,state_versus_time,nodes,times,sws,ews,state_spike_nodes,state_spike_times)
    cnt = 0
    hull_areas_Dict()
    hull_area =  0.0
    @inbounds for (template_ind,col) in enumerate(eachcol(centres))
        px=Plots.scatter!()
        cnt = 0
        hull_areas_Dict[template_ind] = []
        @inbounds for (time_ind,col2) in enumerate(eachcol(mat_of_distances))
            distance = evaluate(Euclidean(),col,col2)
            if distance>new_threshold
                cnt+=1
                #
                #if distance<=minimum(fit[template_ind])
                #    push!(fit[template_ind],distance)
                #    push!(tmd[template_ind],time_ind)
                #    push!(state_time,time_ind)
                #    push!(state_number,template_ind)
                 #   state_versus_time[time_ind] = template_ind
                (repn,rept) = divide_epoch(nodes,times,sws[time_ind],ews[time_ind])
                descriptor=(mean(repn),mean(rept),var(repn),var(rept))
                (hull,hull_area) = concave_hull_pc(repn,rept)
                push!(hull_areas_Dict[template_ind],hull_area)
                push!(hull_areas_Dict[template_ind],hull)
                old_hull_area = hull_area
                hull_difference = old_hull_area-hull_area
                #p1=Plots.plot(col)
                    #title!("vector of cluster center")
                    #p2 = Plots.plot(col2)
                    #title!("vector of observed spikes")

                    #Plots.plot(p1,p2,layout = (2,1))
                    #savefig("fit$distance.png")
                    #push!(state_spike_nodes,repn)
                    #push!(state_spike_times,rept)
                if hull_difference<2
                    Plots.scatter!(px,rept.-time_ind*(ews[time_ind]-sws[time_ind]),repn,color=time_ind,legend=false,markersize=0.1,marker=0.5)
                    savefig("Worst$template_ind$cnt.png")
                else 
                    cnt=0
                end
                #end
            end

        end
    end
    (fit,tmd,state_time,state_number,state_spike_nodes,state_spike_times)
end        
#hull_area =  0.0
#@inbounds for (template_ind,col) in enumerate(eachcol(centres))
#    px=Plots.scatter!()
#    cnt = 0
#    hull_areas_Dict[template_ind] = []

function ReHeat(nodes, times, denom_for_bins;timeBoundary,nodeBoundary)
    #=
    templ = Vector{Any}[]
    for (_) in collect(1:nodeBoundary+1)
        push!(templ,[])
    end
    for (cnt,n) in enumerate(nodes)
        push!(templ[n+1],times[cnt])    
    end
    =#
    ragged = create_spikes_ragged(nodes,times)
    temp_vec = collect(0:Float64(timeBoundary/denom_for_bins):timeBoundary)
    data = zeros(nodeBoundary+1, Int(length(temp_vec)-1))#-1))
    cnt = 1
    for i in 1:length(ragged)
        #@show(times[i])
        psth = fit(Histogram,times,temp_vec)        
        #if sum(psth.weights[:]) != 0.0
            #@show(psth.weights[:])
            #@show(size(psth.weights[:]))


            #@show(size(data))
        data[i,:] = psth.weights[:]
            #@assert sum(data[cnt,:])!=0
        #end
        #cnt +=1
    end
    data::Matrix{Float64}
end

function CompareObsVsCentres(centres,mat_of_distances,new_threshold,fit,tmd,state_time,state_number,state_versus_time,nodes,times,sws,ews,state_spike_nodes,state_spike_times)
    old_hull_area =  0.0
    hull_areas_Dict = Dict()
    hull_Dict = Dict()
    list_of_descriptors = []
    hull_difference = 100
    @inbounds for (template_ind,col) in enumerate(eachcol(centres))
        hull_areas_Dict[template_ind] = []
        #py=Plots.plots()

        px=Plots.scatter!()
        @inbounds for (time_ind,col2) in enumerate(eachcol(mat_of_distances))
            distance = evaluate(Euclidean(),col,col2)
            @assert new_threshold>=0.0
            #@show(distance,new_threshold)
            if distance<new_threshold
                if distance<=minimum(fit[template_ind])
                    push!(fit[template_ind],distance)
                    push!(tmd[template_ind],time_ind)
                    push!(state_time,time_ind)
                    push!(state_number,template_ind)
                    state_versus_time[time_ind] = template_ind
                    (repn,rept) = divide_epoch(nodes,times,sws[time_ind],ews[time_ind])

                    #=
                    #repn = convert(Vector{Float32},repn)
                    if length(repn) > 1
                        rept = rept.-minimum(rept)#time_ind*(ews[time_ind]-sws[time_ind])

                        (hull,hull_area) = concave_hull_pc(repn,rept)
                        
                        push!(hull_areas_Dict[template_ind],hull_area)
                        #push!(hull_areas_Dict[template_ind],hull)
                        hull_diff = abs(old_hull_area-hull_area)
                        old_hull_area = hull_area
                        #B = kde((repn, rept))
                        repto = fit!(OnlineStats.Moments(), rept)
                        #mean(repto)
                        #var(repto)
                        #std(repto)
                        skewTime = skewness(repto)
                        kurtTime = kurtosis(repto)
                        generalVar = StatsBase.genvar([rept;repn])
                        @show(generalVar)
                        sample=StatsBase.totalvar([rept;repn])
                        @show(sample)

                        repno = fit!(OnlineStats.Moments(), repn)
                        #mean(repno)
                        #var(repno)
                        #std(repno)
                        #SkewNode = skewness(repno)
                        #KurtTime = kurtosis(repno)
                        
                        #descriptor=Vector{<:Any}([mean(repno),mean(repto),var(repno),var(repto),minimum(repn),maximum(repn),minimum(rept),maximum(rept),median(repn),mode(repn),median(rept),mode(rept),hull_area,skewTime,kurtTime,SkewNode,KurtTime,generalVar])#B.x[1],B.x[2],B.x[3],B.y[1],B.y[2],B.y[3]])
                        descriptor=Vector{<:Any}([hull_area,sum(repn),sum(rept)])#B.x[1],B.x[2],B.x[3],B.y[1],B.y[2],B.y[3]])

                        #repn = convert(Vector{Float64},repn)
                        #@show()

                        #@show(B.y[1],B.y[2],B.y[3])
                        #@infiltrate

                    else
                        descriptor=Vector{<:Any}([0.0 for _ in 1:3])


                    end        
                    push!(list_of_descriptors,descriptor)

                    =#
                    #@show(descriptor)

                    #@show(hull_areas_Dict)
                    p1=Plots.plot(col)
                    title!("vector of cluster center")
                    p2 = Plots.plot(col2)
                    title!("vector of observed spikes")

                    Plots.plot(p1,p2,layout = (2,1))
                    savefig("fit$distance.png")
                    push!(state_spike_nodes,repn)
                    push!(state_spike_times,rept)
                #if hull_difference<2
                    #@show(hull_difference)
                    if length(rept)>1
                        min_= minimum(rept)
                        @show(min_)
                        #Plots.scatter!(px,rept.-min_,repn,color=time_ind,legend=false)
                        savefig("New_scatter$template_ind.png")
                    end
                end
            end

        end
    end
    #@show(list_of_descriptors)
    #distance_matrix = ones(length(list_of_descriptors),length(list_of_descriptors)).*20

end

function usingAveragedComparison(threshold,spike_mat::AbstractVecOrMat,enrich::AbstractVecOrMat,nodes::AbstractVecOrMat,times::AbstractVecOrMat,sws::AbstractVecOrMat,ews::AbstractVecOrMat,MutatedKernel::Dict)
    patternCnt = Dict()
    PatternTimeInstances = Dict()   
    @inbounds for (k,_) in  enumerate(eachcol(spike_mat))
        patternCnt[k] = 0
    end
    @inbounds for (k,_) in  enumerate(eachcol(spike_mat))
        PatternTimeInstances[k] = []
    end

    @inbounds for (key,value) in pairs(MutatedKernel)
        @inbounds for (k1,_) in  enumerate(eachcol(spike_mat))
            if key>k1
                (repn,rept) = divide_epoch(nodes,times,sws[k1],ews[k1])
                if length(rept)>1
                    rept = rept.-minimum(rept)
                    B1 = kde((rept,repn))
                    r = colwise(Euclidean(), value, B1.density)
                    if sum(r)<= threshold
                        patternCnt[key] += 1
                        push!(PatternTimeInstances[key],k1)
                    end
                end
            end
        end
    end
    (patternCnt::Dict,PatternTimeInstances::Dict)    
end

function KernelComparison(threshold,spike_mat::AbstractVecOrMat,enrich::AbstractVecOrMat,nodes,times,sws,ews)
    kernel_cnt = Dict()
    @inbounds for (k,_) in  enumerate(eachcol(spike_mat))
        kernel_cnt[k] = 0
    end
    current_kernel = Dict()
    MutatedKernel = Dict()
    spike_mass_times, spike_mass_nodes = Dict(),Dict()
    @inbounds for (k,_) in  enumerate(eachcol(spike_mat))
        spike_mass_times[k] = []
        spike_mass_nodes[k] = []
    end

    @inbounds for (k,_) in  enumerate(eachcol(spike_mat))
        @inbounds for (k1,_) in  enumerate(eachcol(spike_mat))
            if k>k1        
                (repn,rept) = divide_epoch(nodes,times,sws[k],ews[k])
                (repn1,rept1) = divide_epoch(nodes,times,sws[k1],ews[k1])
                if length(rept)>1 && length(rept1)>1
                    stored_min = minimum(rept)
                    stored_min1 = minimum(rept1)

                    rept = rept.-stored_min
                    rept1 = rept1.-stored_min1
                    nodeBoundary=maximum([maximum(repn),maximum(repn1)])
                    timeBoundary = ews[k]-sws[k]
                    B0 = kde((rept,repn))
                    B1 = kde((rept1,repn1))
                    r = colwise(Euclidean(),B0.density,B1.density)
    
                    if sum(r)<= threshold                    
                        append!(spike_mass_nodes[k],repn)
                        append!(spike_mass_times[k],rept)
    
                        avg_kernel = (B0.density+B1.density)/2.0
                        if kernel_cnt[k]>0.0
                            MutatedKernel[k] = (MutatedKernel[k]+avg_kernel)/2.0
                        else
                            MutatedKernel[k] = avg_kernel
                        end
                        kernel_cnt[k]+=1
                    end 
                end
            end
        end
    end
    (MutatedKernel::Dict,kernel_cnt::Dict,spike_mass_nodes::Dict,spike_mass_times::Dict)    
end
#using StatsBase

function final_similarity_test(cat_cnt,spike_mat,threshold,centres,mat_of_distances,distance_distributions,nodes,times,state_spike_nodes,state_spike_times,state_time,state_number,sws,ews,color_coded_mat)
    color_coded_mat = zeros(size(mat_of_distances))
    fit=Dict()
    tmd = Dict()
    state_versus_time = Vector{Int32}([])
    @inbounds for (template_ind,_) in enumerate(eachcol(centres))
        tmd[template_ind] = []
    end
    @inbounds for (time_ind,_) in enumerate(eachcol(mat_of_distances))
        fit[time_ind] = [Inf]
        push!(state_versus_time,-1.0)
    end    

    new_threshold = mean(distance_distributions) - std(distance_distributions)
    @assert new_threshold>=0.0
    #(fit,tmd,state_time,state_number,state_spike_nodes,state_spike_times) = CompareObsVsCentresWorst(centres,mat_of_distances,new_threshold,fit,tmd,state_time,state_number,state_versus_time,nodes,times,sws,ews,state_spike_nodes,state_spike_times)

    (fit,tmd,state_time,state_number,state_spike_nodes,state_spike_times) = CompareObsVsCentres(centres,mat_of_distances,new_threshold,fit,tmd,state_time,state_number,state_versus_time,nodes,times,sws,ews,state_spike_nodes,state_spike_times)
    skip_times = []
    cntR = 0
    @inbounds for (time_ind,template_ind) in enumerate(state_versus_time)
        if template_ind!=-1
            temp = count(i->(i==template_ind),state_versus_time)
            if temp == 1
                push!(skip_times,time_ind)
            elseif temp != -1 && temp != 1
                cntR+=1
                (repn,_) = divide_epoch(nodes,times,sws[time_ind],ews[time_ind])
                color_coded_mat[repn,time_ind] .= cat_cnt
                cat_cnt+=1

            end
        end
    end

    
    repn_old = []
    rept_old = []

    p1 = Plots.scatter()
    for (time_ind,col) in enumerate(eachcol(color_coded_mat))
        if sum(col)!=0
            (repn,rept) = divide_epoch(nodes,times,sws[time_ind],ews[time_ind])
            column2 = convert(Vector{Int32},col)   
            #@show(column2)         
            #display(Plots.scatter!(p1,rept,repn,legend=false,colors=column2,markersize=2.0,markerstrokewidth=0,xlims=(0,25)))
            Plots.scatter!(p1,rept,repn,legend=false,colors=cat_cnt)#,markersize=2.0,markerstrokewidth=0)#markersize=2.0,markerstrokewidth=0,xlims=(0,25)))
            
            if length(rept)>1
                min_ = minimum(rept)
                rept = rept.-min_
                #@show(rept)
                #@show(min_)
                vline!(p1,[minimum(rept), maximum(rept)],fill=true,alpha=0.5,colors=["green","blue"])
            end
        
            Plots.plot(p1)
            savefig("grandLabels$cat_cnt$time_ind.png")
            #compare_heatmaps(repn,rept,repn_old,rept_old,time_ind,time_ind)
            repn_old = repn
            rept_old = rept
            #@infiltrate
        end
    end

    #plot_same_category(state_time,state_number,nodes,times,sws,ews,skip_times)
    (state_spike_times,state_spike_nodes,cat_cnt,color_coded_mat)
end
    #display(Plots.plot(p1))
    #@infiltrate

    #=

    #@show(state_versus_time)
    not_reoccuring_states = []
    for j in unique(state_versus_time)
        #divisor = count(i->(i==state_versus_time_[ind]), state_versus_time_)

        temp = count(i->(i==j),state_versus_time)
        if temp ==1
            push!(not_reoccuring_states,j)
            #@show(j)
        end
    end
    =#
    #=
    repeated_windows = Vector{UInt32}([])
    state_versus_time_ = [s+2 for s in state_versus_time]
    maxind = maximum(state_versus_time)+1
    stateTransMat = -2*ones(maxind,maxind)
    @inbounds for (ind,state_val) in enumerate(state_versus_time_)
        if ind < length(state_versus_time_)
            stateTransMat[state_val,state_versus_time_[ind]]+=1

            if stateTransMat[state_val,state_versus_time_[ind]]>0
                divisor = count(i->(i==state_versus_time_[ind]), state_versus_time_)
                if state_versus_time_[ind]/divisor == 1
                    @show(divisor/state_versus_time_[ind])

                    #@show(divisor,state_val,state_versus_time_[ind])#,length(state_versus_time.=state_val))
                    #@infiltrate
                end
            end
        end 

    end
    =#


function heatCompareRapper!(cat_cnt,mat_of_distances,distance_matrix::AbstractVecOrMat,sws,ews,times,nodes,spike_mat,step_size,color_coded_mat;threshold::Real=30)
    NURS = 0.0
    #displacetime=0
    #NumberTemplates = 0.0#::AbstractVecOrMat
    #List_of_templates = Vector{UInt32}([])
    distance_distributions = Vector{Float32}([])
    state_time = Vector{Float32}([])
    state_number = Vector{Float32}([])
    state_spike_nodes = Vector{Any}([])
    state_spike_times = Vector{Any}([])
    enrich = zeros(length(sws),length(sws))
    #@show(size(mat_of_distances))
    #@show(size(mat_of_distances))
    if size(spike_mat)[1]>2
        #fit = Dict()
        @time (MutatedKernel,kernelCntDict,spkn,spkt) = KernelComparison(threshold,spike_mat,enrich,nodes,times,sws,ews)
        @time (patternCnt,PatternTimeInstances) = usingAveragedComparison(threshold,spike_mat,enrich,nodes,times,sws,ews,MutatedKernel) 
        plot_patterns_found(patternCnt,PatternTimeInstances,MutatedKernel,sws,ews,nodes,times,spkn,spkt)
        plot_patterns_found_vs_time_hits(patternCnt,PatternTimeInstances,MutatedKernel,sws,ews,nodes,times,spkn,spkt)

    end
end
function plot_patterns_found_vs_time_hits(patternCnt,PatternTimeInstances,MutatedKernel,sws,ews,nodes,times,spkn,spkt)
    Tcnt = 0
    p0=Plots.scatter()
    p1=Plots.plot()
    for (k,v) in pairs(patternCnt)
        nn = []
        tt = []
        if v>0
            Plots.vline!(p1,PatternTimeInstances[k],alpha=0.5,legend=false)#,color=k)
            Plots.scatter!(p0,spkt[k].+Tcnt,spkn[k],legend=false,markersize=1.0,markerstrokewidth=0)
            Tcnt+=1
        end
    end
    p3 = Plots.scatter(times,nodes,legend=false,markersize=1.0,markerstrokewidth=0)
    Plots.plot(p0,p1,p3,layout=(2,2),size=(1000,1000))
    savefig("Templates.png")

end
function plot_patterns_found(patternCnt,PatternTimeInstances,MutatedKernel,sws,ews,nodes,times,spkn,spkt)
    for (k,v) in pairs(patternCnt)
        nn = []
        tt = []
        if v>0
            p0=Plots.scatter()
            for time_index in PatternTimeInstances[k]
                (repn,rept) = divide_epoch(nodes,times,sws[Int32(time_index)],ews[Int32(time_index)])
                Plots.scatter!(p0,rept,repn,legend=false,markersize=1.0,markerstrokewidth=0,color=k)
            end
            p2=Plots.scatter(spkt[k],spkn[k],legend=false,markersize=1.0,markerstrokewidth=0)

            p1=Plots.heatmap(MutatedKernel[k])#,legend=false)
            Plots.plot(p0,p1,p2)
            savefig("patterns$k.png")
            Plots.plot(tt,nn,legend=false,markersize=1.0,markerstrokewidth=0)
        end
    end
end


#=
        threshold_attribute = sum(r)
        timeBoundaryMin=minimum([minimum(rept),minimum(rept1)])
        p1 = Plots.heatmap(B0.density)
        title!("Sample $k")
        p0 = Plots.heatmap(B1.density)
        title!("Sample $k1")
        nspike0 = length(rept)
        p2 = Plots.plot()#hull0)
        Plots.scatter!(p2,rept,repn,legend=false,ylim=(0.0,nodeBoundary),xlim=(timeBoundaryMin,timeBoundary))
        difference = sum(r) - threshold 
        title!(p2,"N. spike: $nspike0")
        xlabel!("Time (ms)")
        ylabel!("Neuron ID")

        nspike1 = length(rept1)
        p3 = Plots.plot()#hull1)
        Plots.scatter!(p3,rept1,repn1,legend=false,ylim=(0.0,nodeBoundary),xlim=(timeBoundaryMin,timeBoundary))
        title!(p3,"N. spike: $nspike1. Threshold delta $difference")
        xlabel!("Time (ms)")
        ylabel!("Neuron ID")

        #p4 = Plots.plot()#hull1)
        difference_density = B1.density - B0.density
        p4 = Plots.heatmap(difference_density)
        title!(p4,"Difference between kernel densities")
        #=
        (hull1,hull_area) = concave_hull_pc(repn1,rept1)
        isis0,_,_ = create_ISI_histogram(repn,rept)
        isis1,_,_ = create_ISI_histogram(repn1,rept1)
        
        rates0 = create_rates_histogram(repn,rept)
        rates1 = create_rates_histogram(repn1,rept1)
        #@show(length(rates0),length(rates1))
        #@assert
        if length(rates0) == length(rates1)
            rrate = cor(rates0,rates1)
            #@show(rrate)
        end
        min_binr = minimum([minimum(rates0),minimum(rates1)])
        max_binr = maximum([maximum(rates0),maximum(rates1)])
        =#
        =#
        #p4 = Plots.plot([i for i in 1:length(rates0)],rates0,legend=false)#,xlim=(min_binr,max_binr))
        #title!(p4,"Firing Rate Histogram")
        #p5 = Plots.plot([i for i in 1:length(rates1)],rates1,legend=false)#,xlim=(min_binr,max_binr))
        #title!(p5,"Firing Rate Histogram")
        #min_bini = minimum([minimum(isis0),minimum(isis1)])
        #max_bini = maximum([maximum(isis0),maximum(isis1)])

        #=
        p6 = Plots.plot(hull0,legend=false)#,fillrange = [i for i in 1:size(hull0)[1]])#,xlim=(min_bini,max_bini))
        title!(p4,"Firing Rate Histogram")
        p7 = Plots.plot(hull1,legend=false)#,fillrange = [i for i in 1:size(hull1)[1]])#,xlim=(min_bini,max_bini))
        title!(p5,"Firing Rate Histogram")
        =#
        #=
        Plots.plot(p0,p1,p2,p3,p4,layout=(3,2),size=(1000,1000))
        #@infiltrate
        savefig("NewHeat$k$k1$threshold_attribute.png")
        =#
        #Plots.closeall()

        #println("save")

        #temp = length(rept)
        #title!("$temp")

        #plot(p6,plot(framestyle = :none),
        #p3,Plots.histogram(rates0,legend=false,orientation = :horizontal),
        #link = :both)
        #savefig("heat$k$k1$attribute.second.png")

        #@show(sum(values(kernelCntDict)))
        #66
        
        #infil> 
        #@show(sum(values(patternCnt)))
        #113
        
        #@infiltrate


function label_spikes!(cat_cnt,mat_of_distances,distance_matrix::AbstractVecOrMat,sws,ews,times,nodes,spike_mat,step_size,color_coded_mat;threshold::Real=5)
    NURS = 0.0
    #displacetime=0
    #NumberTemplates = 0.0#::AbstractVecOrMat
    #List_of_templates = Vector{UInt32}([])
    distance_distributions = Vector{Float32}([])
    state_time = Vector{Float32}([])
    state_number = Vector{Float32}([])
    state_spike_nodes = Vector{Any}([])
    state_spike_times = Vector{Any}([])

    #@show(size(mat_of_distances))
    if size(mat_of_distances)[1]>2
        enrich = zeros(size(mat_of_distances))
        #fit = Dict()
        enrich = heatComparison(spike_mat,enrich,nodes,times,sws,ews)
    end
        #=
        @infiltrate
        enrich_ = get_proto_templates(mat_of_distances,enrich,threshold,distance_distributions,nodes,times,sws,ews)
            #@show(enrich)
        if size(enrich_)[2] != 0
            @show(size(enrich_)[2])
        
            #@time List_of_templates = refine_templates(mat_of_distances,List_of_templates,threshold,distance_distributions)
            centres,_ = cluster(enrich_)
            #if !fail
            (state_spike_times,state_spike_nodes,cat_cnt,color_coded_mat) = final_similarity_test(cat_cnt,spike_mat,threshold,centres,mat_of_distances,distance_distributions,nodes,times,state_spike_nodes,state_spike_times,state_time,state_number,sws,ews,color_coded_mat)

            #(state_spike_times,state_spike_nodes,cat_cnt,color_coded_mat) = final_similarity_test(cat_cnt,spike_mat,threshold,centres,mat_of_distances,distance_distributions,nodes,times,state_spike_nodes,state_spike_times,state_time,state_number,sws,ews,color_coded_mat)
            #end
            number_windows = length(eachcol(mat_of_distances))
            window_duration = last(ews)-last(sws)
            repeatitive = NURS/(number_windows*window_duration)
        else
            repeatitive= nothing
            (state_spike_times,state_spike_nodes,cat_cnt,color_coded_mat) = (nothing,nothing,nothing,nothing)
        end
    else
        repeatitive= nothing
        (state_spike_times,state_spike_nodes,cat_cnt,color_coded_mat) = (nothing,nothing,nothing,nothing)
    end
    =#
    (repeatitive,NURS::Real,state_spike_times,state_spike_nodes,state_number,state_time,cat_cnt,color_coded_mat)
end
function old_label_distmat!()
    #display(Plots.heatmap(enrich))
    px=Plots.scatter(state_time,state_number,color=state_number,legend=false,markersize=1.5,markerstrokewidth=1)
    #pz=Plots.scatter(times,nodes,legend=false,markersize=2.5,markerstrokewidth=1)

    #layout=
    display(Plots.plot(px,py, layout=(2, 1)))
    #py=Plots.scatter(state_time,state_number,color=state_number,legend=false)


    

    #not_templates = [other_ind for (other_ind,row) in enumerate(eachcol(mat_of_distances)) ]
    #@inbounds for template_ind1 in List_of_templates
    #    deleteat!(not_templates, not_templates .== template_ind1)
    #end
    state_time = []
    state_number = []
    py=Plots.scatter()
    fit=Dict()
    @inbounds for (time_ind,col2) in enumerate(eachcol(mat_of_distances))
        fit[time_ind] = []
    end    
    @inbounds for template_ind1 in List_of_templates
        @inbounds for (time_ind,col2) in enumerate(eachcol(mat_of_distances))
            if template_ind1!=time_ind
                @assert other_ind!=template_ind1
                col1 = mat_of_distances[:,template_ind1]
                if (abs(sum(col1.-col2)))<0.5
                    push!(fit[time_ind],abs(sum(col1.-col2)))
                    check_min = minimum(fit[time_ind])
                    if abs(sum(col1.-col2))<=check_min

                        push!(state_time,time_ind)
                        push!(state_number,template_ind1)
                        (repn,rept) = divide_epoch(nodes,times,sws[time_ind],ews[time_ind])
                        Plots.scatter!(py,rept,repn,color=template_ind1,legend=false,markersize=2.5,markerstrokewidth=1)
                    end
                end
            end
        end
    end
    #px=Plots.scatter(state_time,state_number,color=state_number,legend=false,markersize=2.5,markerstrokewidth=1)
    pz=Plots.scatter(times,nodes,legend=false,markersize=2.5,markerstrokewidth=1)

    #layout=
    #display(Plots.plot(px,py,pz, layout=(3, 1)))
    px=Plots.scatter(state_time,state_number,color=state_number,legend=false,markersize=1.5,markerstrokewidth=1)
    #pz=Plots.scatter(times,nodes,legend=false,markersize=2.5,markerstrokewidth=1)

    #layout=
    Plots.plot(px,py, layout=(2, 1))
    savefig("quality_check.png")
    #py=Plots.scatter(state_time,state_number,color=state_number,legend=false)



    @inbounds for template_ind in List_of_templates
        (templaten,templatet) = divide_epoch(nodes,times,sws[template_ind],ews[template_ind])
        Plots.scatter!(pt,templatet,templaten,color=template_ind,xlims=(minimum(times),maximum(times)),legend=false,ylims=(minimum(nodes),maximum(nodes)),markersize=1.9,markerstrokewidth=1,markershape =:vline)

    end
    savefig("Template_only_plot.png")
end

function label_exhuastively_distmat!(mat_of_distances::AbstractVecOrMat,distance_matrix::AbstractVecOrMat,sws,ews,times,nodes,div_spike_mat_with_displacement,step_size;threshold::Real=5)
    #cnts_total = 0.0
    NURS = 0.0
    #template_times_dict = Dict()
    #template_nodes_dict = Dict()
    indold = -1
    indold2 = -1
    displacetime=0
    #nodes,times = ragged_to_lists(spikes_ragged)
    #p1=Plots.scatter()
    #py = Plots.scatter()

    NumberTemplates = 0.0


    List_of_templates = []
    enrich = zeros(size(mat_of_distances))
    @inbounds for (template_ind,col) in enumerate(eachcol(mat_of_distances))

    #stop_at_half = Int(trunc(length(eachcol(mat_of_distances))/2))
    #if ind <= stop_at_half

        ##
        ## TODO build time windows here!
        ##
        ##
    
        @inbounds for (ind2,col2) in enumerate(eachcol(mat_of_distances))
            if template_ind!=ind2
                if (abs(sum(col.-col2)))<0.5
                    distance = evaluate(Euclidean(),col,col2)
                    if template_ind!=indold && ind2!=indold2

                        append!(List_of_templates,template_ind)
                    end
                    enrich[:,template_ind] = col
                    enrich[:,ind2] = col2

                    indold=template_ind
                    indold2=ind2
                    #=
                    #@show(distance)
                    p1=Plots.plot(row)
                    p2=Plots.plot(row2)
                    p3=Plots.plot(row.-row2)
                    Plots.plot(p1,p2,p3)
                    savefig("vectorsim$ind$ind2.png")
                    =#
    
                    #t#emp= vcat
                    #@#show(typeof(temp))
                    #display(Plots.heatmap(hcat(col',col')))
                    #Plots.heatmap(hcat(row',row2'))
                    NURS += 1.0
                    #distance_matrix[template_ind,ind2] = abs(distance)
                    #if !(haskey(template_times_dict, template_ind))
                    #    template_times_dict[template_ind] = []
                    #    template_nodes_dict[template_ind] = []
                    #end
                    (n0,t0) = divide_epoch(nodes,times,sws[ind2],ews[ind2])
                    (templaten,templatet) = divide_epoch(nodes,times,sws[template_ind],ews[template_ind])

                    #(templaten,templatet) = divide_epoch(nodes,times,sws[ind],ews[ind])
                    #@assert ind!=ind2
                    #=
                        NumberTemplates+=1
                        Plots.scatter!(py,[template_ind],[NumberTemplates],color=template_ind,xlims=(minimum(times),maximum(times)),legend=false,ylims=(minimum(nodes),maximum(nodes)),markersize=1.5,markerstrokewidth=1,markershape =:vline)
                        Plots.scatter!(pt,templatet,templaten,color=template_ind,xlims=(minimum(times),maximum(times)),legend=false,ylims=(minimum(nodes),maximum(nodes)),markersize=1.5,markerstrokewidth=1,markershape =:vline)
                        title!("Number of templates $NumberTemplates")
                        #displacetime+=maximum(templatet)

                    end
                    =#


                    #p1=Plots.scatter(templatet,templaten,color=ind,xlims=(minimum(times),maximum(times)),legend=false,ylims=(minimum(nodes),maximum(nodes)))
                    #Plots.scatter!(p1,t0,n0,color=ind2,legend=false,xlims=(minimum(times),maximum(times)),ylims=(minimum(nodes),maximum(nodes)))
                    #savefig("matchedPattern$ind.$ind2.png")

                    #push!(template_times_dict[ind],t0)
                    #push!(template_nodes_dict[ind],n0)

                end
            end
        #end
        #Plots.plot(py,pt)
        end
        #savefig("scatter_match_rebootxy$ind.png")

    end


    #@show(unique(List_of_templates))
    #@show(length(unique(List_of_templates)))
    #pt = Plots.scatter()


    #@inbounds for (ind2,row2) in enumerate(eachcol(mat_of_distances))
        #px=Plots.scatter()

            #@show(ind,ind2)
            
            #if distance<threshold && distance!=0
    #new_list_of_templates=[]
    @inbounds for template_ind1 in List_of_templates
        @inbounds for template_ind0 in List_of_templates
            if template_ind1!=template_ind0
                if (abs(sum(mat_of_distances[:,template_ind1].-mat_of_distances[:,template_ind0])))<0.5
                    #println("not unique")
                    deleteat!(List_of_templates, List_of_templates .== template_ind1)
                end
            end
        end
    end

    

    enrich = enrich[:, vec(mapslices(col -> any(col .!= 0), enrich, dims = 1))]
    classes = length(unique(List_of_templates))
    R = kmeans(enrich, classes; maxiter=2000, display=:iter)
    a = assignments(R) # get the assignments of points to clusters
    sort_idx =  sortperm(assignments(R))
    enrich = enrich[:,sort_idx]
    centres = R.centers # get the cluster centers
    state_time = []
    state_number = []
    py=Plots.scatter()
    state_spike_nodes = []
    #state_spike_times = []
    fit=Dict()
    @inbounds for (time_ind,_) in enumerate(eachcol(mat_of_distances))
        fit[time_ind] = []
    end    
    for (template_ind1,col) in enumerate(eachcol(centres))
        @inbounds for (time_ind,col2) in enumerate(eachcol(mat_of_distances))
            if (abs(sum(col2.-col)))<1.0
                push!(fit[time_ind],abs(sum(col2.-col)))
                check_min = minimum(fit[time_ind])
                if abs(sum(col2.-col))<=check_min
                    push!(state_time,time_ind)
                    push!(state_number,template_ind1)
                    (repn,rept) = divide_epoch(nodes,times,sws[time_ind],ews[time_ind])
                    push!(state_spike_nodes,repn)
                    push!(state_spike_nodes,rept)

                    Plots.scatter!(py,rept,repn,color=template_ind1,legend=false,markersize=2.5,markerstrokewidth=1)
                end
            end
        end
    end
    #display(Plots.heatmap(enrich))
    px=Plots.scatter(state_time,state_number,color=state_number,legend=false,markersize=1.5,markerstrokewidth=1)
    #pz=Plots.scatter(times,nodes,legend=false,markersize=2.5,markerstrokewidth=1)

    #layout=
    display(Plots.plot(px,py, layout=(2, 1)))
    #py=Plots.scatter(state_time,state_number,color=state_number,legend=false)


    

    #not_templates = [other_ind for (other_ind,row) in enumerate(eachcol(mat_of_distances)) ]
    #@inbounds for template_ind1 in List_of_templates
    #    deleteat!(not_templates, not_templates .== template_ind1)
    #end
    state_time = []
    state_number = []
    py=Plots.scatter()
    fit=Dict()
    @inbounds for (time_ind,col2) in enumerate(eachcol(mat_of_distances))
        fit[time_ind] = []
    end    
    @inbounds for template_ind1 in List_of_templates
        @inbounds for (time_ind,col2) in enumerate(eachcol(mat_of_distances))
            if template_ind1!=time_ind
                @assert other_ind!=template_ind1
                col1 = mat_of_distances[:,template_ind1]
                if (abs(sum(col1.-col2)))<0.5
                    push!(fit[time_ind],abs(sum(col1.-col2)))
                    check_min = minimum(fit[time_ind])
                    if abs(sum(col1.-col2))<=check_min

                        push!(state_time,time_ind)
                        push!(state_number,template_ind1)
                        (repn,rept) = divide_epoch(nodes,times,sws[time_ind],ews[time_ind])
                        Plots.scatter!(py,rept,repn,color=template_ind1,legend=false,markersize=2.5,markerstrokewidth=1)
                    end
                end
            end
        end
    end
    #px=Plots.scatter(state_time,state_number,color=state_number,legend=false,markersize=2.5,markerstrokewidth=1)
    pz=Plots.scatter(times,nodes,legend=false,markersize=2.5,markerstrokewidth=1)

    #layout=
    #display(Plots.plot(px,py,pz, layout=(3, 1)))
    px=Plots.scatter(state_time,state_number,color=state_number,legend=false,markersize=1.5,markerstrokewidth=1)
    #pz=Plots.scatter(times,nodes,legend=false,markersize=2.5,markerstrokewidth=1)

    #layout=
    Plots.plot(px,py, layout=(2, 1))
    #savefig("quality_check$.png")
    #py=Plots.scatter(state_time,state_number,color=state_number,legend=false)


    @inbounds for template_ind in List_of_templates
        (templaten,templatet) = divide_epoch(nodes,times,sws[template_ind],ews[template_ind])
        Plots.scatter!(pt,templatet,templaten,color=template_ind,xlims=(minimum(times),maximum(times)),legend=false,ylims=(minimum(nodes),maximum(nodes)),markersize=1.9,markerstrokewidth=1,markershape =:vline)

    end
    savefig("Template_only_plot.png")

    #@infiltrate

    number_windows = length(eachcol(mat_of_distances))
    window_duration = last(ews)-last(sws)
    repeatitive = NURS/(number_windows*window_duration)
    (repeatitive,NURS::Real)
end
"""
label_exhuastively_distmat!
"""
function label_spikes(mat_of_distances::AbstractVecOrMat,sws,ews,times,nodes,div_spike_mat_with_displacement,step_size,cat_cnt;threshold::Real=5, disk=false)
    if !disk
        distance_matrix = zeros(length(eachcol(mat_of_distances)),length(eachcol(mat_of_distances)))
    else
        io = open("/tmp/mmap.bin", "w+")
        distance_matrix = mmap(io, Matrix{Float32}, (length(eachcol(mat_of_distances)),length(eachcol(mat_of_distances))))
    end
    color_coded_mat = zeros(size(div_spike_mat_with_displacement))
    #enrich = 
    heatCompareRapper!(cat_cnt,mat_of_distances,distance_matrix,sws,ews,times,nodes,div_spike_mat_with_displacement,step_size,color_coded_mat;threshold)
    #repeatitive,NURS,spike_state_times,spike_state_nodes,state_number,state_time,cat_cnt = label_spikes!(cat_cnt,mat_of_distances,distance_matrix,sws,ews,times,nodes,div_spike_mat_with_displacement,step_size,color_coded_mat;threshold)
    #repeatitive,dict0,dict1,NURS,state_frequency_histogram = templates_using_cluster_centres!(mat_of_distances,distance_matrix,sws,ews,times,nodes,div_spike_mat_with_displacement,times_per_slice,nodes_per_slice,ind;threshold)
    #repeatitive,NURS = label_exhuastively_distmat!(mat_of_distances,distance_matrix,sws,ews,times,nodes,div_spike_mat_with_displacement,step_size;threshold)
    #sws,ews,times,nodes

    #repeatitive,NURS,spike_state_times,spike_state_nodes,state_number,state_time,cat_cnt,color_coded_mat
    #enrich
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

function create_spikes_ragged(nodes::Vector{Any},times::Vector{Any})
    nodes = convert(Vector{UInt32},nodes)
    times = convert(Vector{Float32},times)
    (spikes_ragged,numb_neurons) = create_spikes_ragged(nodes,times)
    (spikes_ragged::AbstractVecOrMat,numb_neurons::UInt32)
end
"""
A method to get collect the Inter Spike Intervals (ISIs) per neuron, and then to collect them together to get the ISI distribution for the whole cell population
Also output a ragged array (Array of unequal length array) of spike trains. 
"""
function create_spikes_ragged(nodes::Vector{UInt32},times::Vector{<:Real})
    spikes_ragged = Vector{Any}([])
    numb_neurons = UInt32(maximum(nodes))#+UInt32(1) # Julia doesn't index at 0.
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
    #matrix_ind = Vector{UInt32}([])
    classes = 10
    R = kmeans(distmat', classes; maxiter=2000, display=:iter)
    a = assignments(R) # get the assignments of points to clusters
    sort_idx =  sortperm(assignments(R))
    for i in unique(a)
        push!(spike_jobs,spikes_ragged[a.==i])
        #push!(matrix_ind,)
    end
    #println("done")
    spike_jobs,a
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

function get_repeat_times(M::AbstractMatrix, tol,neuron0,div_spike_mat_with_displacement,p1)
    masked = copy(div_spike_mat_with_displacement)
    rows=[]
    tx=[] 

    for (time_index0,row) in enumerate(eachrow(M))
        for (time_ind,time_window) in enumerate(row)
            if time_window<tol && time_window!=0
                #push!(rows,time_ind)
                push!(rows,time_index0)

                #println("$neuron0 repeats itself at times")
                #println("time index $time_ind repeats at $time_index0")
                #p1=Plots.scatter(div_spike_mat_with_displacement[neuron0,time_index0],[1 for i in 1:length(div_spike_mat_with_displacement[neuron0,time_index0])])
                #p2=Plots.scatter(div_spike_mat_with_displacement[neuron0,time_ind],[1 for i in 1:length(div_spike_mat_with_displacement[neuron0,time_ind])])
                #display(Plots.plot(p1,p2))
                #@show(div_spike_mat_with_displacement[neuron0,time_ind])
                #@show(div_spike_mat_with_displacement[neuron0,time_index0])
                #div_spike_mat_with_displacement[neuron0,time_ind]

            end
        end
    end
    for times in div_spike_mat_with_displacement[neuron0,rows]
        append!(tx,times[1][1])
    end
    #@show(tx)

    nodes = [neuron0 for i in 1:length(div_spike_mat_with_displacement[neuron0,rows])]
    #p1 = Plots.scatter!(p1,div_spike_mat_with_displacement[neuron0,rows],[neuron0 for i in 1:length(div_spike_mat_with_displacement[neuron0,rows])],legend=false)
    #savefig("LetsSee.png")
    p1 = nothing
    #p1,rows
    (tx,nodes,p1)
end


function compare_every_time_block_to_all_others(div_spike_mat_with_displacement)
    p1=Plots.scatter()
    tt=Vector{Float32}([])
    nn=Vector{UInt32}([])
    @inbounds for (indrow,row) in enumerate(eachrow(div_spike_mat_with_displacement))
        distance_matrix = zeros(size(div_spike_mat_with_displacement)[2],size(div_spike_mat_with_displacement)[2])        
        @inbounds for (ind0,neuron0) in enumerate(row)
            @inbounds if length(neuron0[1])>1
                @inbounds for (ind1,neuron1) in enumerate(row)
                    if length(neuron1[1])>1
                        if ind1!=ind0
                            t1_ = sort(unique(neuron1[1]))
                            t0_ = sort(unique(neuron0[1]))
                            maxt = maximum([maximum(t1_),maximum(t0_)])
                            _, S = SPIKE_distance_profile(t1_,t0_;t0=0,tf = maxt)
                            distance_matrix[ind0,ind1] = sum(S)
                        else
                            distance_matrix[ind0,ind1] = 1.0
                        end
                    else 
                        distance_matrix[ind0,ind1] = 1.0
                    end
                end
            end
        end
        tol = mean(distance_matrix)-(1*std(distance_matrix)/3.0)
        (times,nodes,p1) = get_repeat_times(distance_matrix, tol,indrow,div_spike_mat_with_displacement,p1)
        append!(tt,times)
        append!(nn,nodes)
        #p1=Plots.scatter!(p1)
        #savefig("LetsSee.png")
        #display(p2)
  
        #display(Plots.heatmap(distance_matrix))
    end 
    (tt,nn)
end
#=
function min_key(d)
  minkey, minvalue = next(d, start(d))[1]
  for (key, value) in d
    if value < minvalue
      minkey = key
      minvalue = value
    end
  end
  minkey
end
=#
function detect_unstructed_noise(spikes_ragged,a,ts,job_list)
    variabilities = Dict()

    for ind in 1:length(unique(a))
        temp = a.==ind
        ts_ = ts[temp,:]
        variability = []
        for spike in spikes_ragged[temp]
            push!(variability,CV(spike))    
        end    
        variability = sum(variability)/length(variability)
        variabilities[ind] = variability
    end
    min_key(d) = reduce((x, y) -> d[x] ≤ d[y] ? x : y, keys(d))
    #min_key(d) = collect(keys(d))[indmin(collect(values(d)))]
    key = min_key(variabilities)
    #job_list = unique(a)
    deleteat!(job_list, key)
    job_list
end

function detect_small(spikes_ragged,a,ts)
    cluster_sizes = Dict()

    for ind in 1:length(unique(a))
        temp = a.==ind
        #@infiltrate
        #ts_ = ts[temp,:]
        cluster_size = []
        #for spike in spikes_ragged[temp]
        #    push!(cluster_size,sum(ind))    
        #end    
        cluster_size = sum(temp)#/length(variability)
        cluster_sizes[ind] = cluster_size
    end
    min_key(d) = reduce((x, y) -> d[x] ≤ d[y] ? x : y, keys(d))
    key = min_key(cluster_sizes)
    job_list = unique(a)
    #deleteat!(job_list, key)
    #delete!(cluster_sizes, key)

    #key = min_key(cluster_sizes)
    #deleteat!(job_list, key)
    #delete!(cluster_sizes, key)

    job_list
end
using ImageFeatures, Images, ImageDraw#, CoordinateTransformations
function doanalysCV(d)
    
    @unpack nodes,times, number_divisions, similarity_threshold = d
    Ncells = length(unique(nodes))+1
    maxt = maximum(times)
    sum_of_rep=0.0
    NURS_sum=0.0
    recording_start_time = minimum(times)
    # step_size = dt = 0.5

    step_size = dt = 3.0
    tau = 0.5
    img = get_ts(nodes,times,step_size,tau)#;disk=false)
    img1 = Gray.(img)
    Plots.heatmap(img1)
    savefig("blah.png")
    keypoints_1 = Keypoints(fastcorners(img1, 12, 0.4))
    #@show(size(img1))
    brief_params = BRIEF(size = 201, window = 10, seed = 123)

    desc_1, ret_keypoints_1 = create_descriptor(img1, keypoints_1, brief_params)
    desc_2, ret_keypoints_2 = create_descriptor(img1, keypoints_1, brief_params)
    #@show(ret_keypoints_2)
    #@show(desc_2)
    #@infiltrate
    
    matches = match_keypoints(ret_keypoints_1, ret_keypoints_2, desc_1, desc_2, 1.5)

    for m in matches 
        if m[1] != m[2] 
            @show(m)
        end
    end
    #@infiltrate



    #grid = hcat(img1, img1)
    #offset = CartesianIndex(0, size(img1, 2))
    #map(m -> draw!(grid, LineSegment(m[1], m[2] + offset)), matches)
    #savefig("TimeSurfaceMatches.png")
    #save("brisk_example.jpg", grid); 
    nothing # hide



end
#img = testimage("lighthouse")
function concave_hull_pc(nodes,times)
    points = [[ti, no] for (ti,no) in zip(nodes,times)];
    x = [p[1] for p in points];
    y = [p[2] for p in points];

    hull = concave_hull(points)
    hull_area = area(hull)

    #scatter(x,y,ms=1,label="",axis=false,grid=false,markerstrokewidth=0.0)
    #display(plot!(hull))

    #annotate!(pi/2,0.5,"K = $(hull.k)")
    #annotate!(pi/2,0.25,"Area $(round(hull_area, digits=3))")
    (hull,hull_area)
end
function make_jobs(spikes_ragged,a)
    job_list = unique(a)
    small_job_list = []
    finished_jobs = []
    index_list = []
    @inbounds for ind in job_list
        temp = a.==ind
        temp = convert(Vector{Bool},temp)

        if sum(temp)<=10
            push!(small_job_list,temp)
        else
            push!(index_list,temp)
            push!(finished_jobs,spikes_ragged[temp])
        end
    end
    collector = zeros(length(spikes_ragged))
    @inbounds for s in small_job_list
        collector += s
    end
    push!(index_list,collector)

    collector = convert(Vector{Bool},collector)
    push!(finished_jobs,spikes_ragged[collector])
    finished_jobs,index_list
end

function doanalysisrev(d)
    @unpack nodes,times, number_divisions, similarity_threshold = d
    #@show(nodes)
    Ncells = length(unique(nodes))
    maxt = maximum(times)
    sum_of_rep=0.0
    NURS_sum=0.0
    recording_start_time = minimum(times)
    # step_size = dt = 0.5

    step_size = dt = 3.0
    tau = 0.5
    ts = get_ts(nodes,times,step_size,tau)#;disk=false)
    (spikes_ragged,numb_neurons) = create_spikes_ragged(nodes,times) 
    ##
    # Don't actually normalize, I believe it makes things worse.
    #normalize!(ts)
    ##
    classes = 2
    R = kmeans(ts', classes; maxiter=1000, display=:iter)
    a = assignments(R) # get the assignments of points to clusters
    (spikes_ragged,numb_neurons) = create_spikes_ragged(nodes,times) 
    spikeMat,sws,ews,window_size = spike_matrix_divided(spikes_ragged, step_size,number_divisions, maxt,recording_start_time ;displace=true,sliding=false)
    cat_cnt = 0.0
    #enrich = 
    label_spikes(ts,sws,ews,times,nodes,spikeMat,step_size,cat_cnt;threshold=similarity_threshold)    
    
    
    #enrich = label_spikes(ts,sws,ews,times,nodes,spikeMat,step_size,cat_cnt;threshold=similarity_threshold)
    #enrich = sparse(enrich)
    #display(enrich)
    
    
    #repeatitive,NURS,spike_state_times,spike_state_nodes,state_number,state_time,cat_cnt = label_spikes(ts,sws,ews,times,nodes,spikeMat,step_size,cat_cnt;threshold=similarity_threshold)

    #repeatitive,NURS,spike_state_times,spike_state_nodes,state_number,state_time = label_spikes(ts,sws,ews,times,nodes,spikeMat,step_size;threshold=similarity_threshold)
    
    ##
    # Assume one of the clustering classes is unstructured noise, 
    # Disgard all associated channels from the analysis.
    # 
    ##
    #job_list = detect_small(spikes_ragged,a,ts)

    finished_jobs,cluster_ind = make_jobs(spikes_ragged,a)
    sst = []
    ssn=[]
    sn = []
    cat_cnt = 0.0
    enrich = zeros(length(sws),length(sws))

    for (spikes_ragged_packet,cl_ind) in zip(finished_jobs,cluster_ind)
        ts_ = ts[cl_ind,:]
        nodes,times = ragged_to_lists(spikes_ragged_packet)
        spikeMat,sws,ews,window_size = spike_matrix_divided(spikes_ragged_packet, step_size,number_divisions, maxt,recording_start_time ;displace=true,sliding=false)    
        label_spikes(ts_,sws,ews,times,nodes,spikeMat,step_size,cat_cnt;threshold=similarity_threshold)
    end

    #for e in enrichs
    #    enrich+=e
    #end
    #enrich = sparse(enrich)
    #display(enrich)
    #show([sum(enrich) for enrich in enrichs])
    #@infiltrate
    #=
    p1 = Plots.scatter()
    offset = 0
    for (spike_state_times,spike_state_nodes,state_number) in zip(sst,ssn,sn)

        for (t,n,st) in zip(spike_state_times,spike_state_nodes,state_number)
            #@show(t,n)

            #st = convert(Vector{Int32},st)

            #colors = st.+offset
            #@show(colors)

            #colors = convert(Vector{Int32},colors)

            Plots.scatter!(p1,t,n,legend=false,markersize=3.5,markerstrokewidth=2)  
            offset+=Int(maximum(st))
      
        end
    end
    Plots.plot(p1)
    savefig("latest_patterns.png")
    =#
    #=
    @time spikeMat,sws,ews,window_size = spike_matrix_divided(spikes_ragged, step_size,number_divisions, maxt,recording_start_time ;displace=true,sliding=false)
    #tps,nps,full_sliding_window_starts,full_sliding_window_ends = spike_matrix_slices(nodes,times,number_divisions,maxt,recording_start_time)

    @time (_,rep,NURS) = label_spikes(ts,sws,ews,times,nodes,spikeMat,step_size;threshold=similarity_threshold)
    @show(NURS)
    spike_jobs,a = cluster_get_jobs(distmat,spikes_ragged)
    @inbounds for (ind,spikes_ragged_packet) in enumerate(spike_jobs)
        div_spike_mat_with_displacement_local = div_spike_mat_with_displacement[a.==ind,:]
        nodes,times = ragged_to_lists(spikes_ragged_packet)
        times_per_slice,nodes_per_slice,full_sliding_window_starts,full_sliding_window_ends = spike_matrix_slices(nodes,times,number_divisions,maxt,recording_start_time)
        (new_distmat,_) = compute_metrics_on_matrix_divisions(div_spike_mat_with_displacement_local,times_per_slice,ews,step_size,metric=:kreuz)
        (_,rep,_,_,NURS,state_frequency_histogram) = label_spikes(new_distmat,sws,ews,times,nodes,div_spike_mat_with_displacement_local;threshold=similarity_threshold)
    end
    =#
    #distance_matrix::AbstractVecOrMat,repeatitive::Real,dict0::Dict,dict1::Dict,NURS::Real
    #(_,rep,_,_,NURS,state_frequency_histogram) = label_spikes(new_distmat,sws,ews,times,nodes,div_spike_mat_with_displacement_local;threshold=similarity_threshold)

end



function doanalysis(d)
    @unpack nodes,times, number_divisions, similarity_threshold = d
    dt = 0.01
    tau = 0.5
    ts = get_ts(nodes,times,dt,tau)#;disk=false)
    #normalize!(ts)
    classes = 10
    R = kmeans(ts', classes; maxiter=2000, display=:iter)
    a = assignments(R) # get the assignments of points to clusters
    sort_idx =  sortperm(assignments(R))
    ts = ts[sort_idx,:]

    #=
    for ind in 1:length(unique(a))
        temp = a.==ind
        ts_ = ts[temp,:]
        Plots.heatmap(ts_)
        savefig("xx$ind.sortedTSHeatmap.png")
    end
    Plots.heatmap(ts)
    savefig("sorted.png")
    =#

    #profile = matrix_profile(ts, 2; showprogress=true)

    #plot(profile) # Should have minima at 21 and 52
    #k = 10
    #mot = motifs(profile, k; r=2, th=5)
    #plot(profile, mot)
    #savefig("matProfile2.png")

    #Plots.heatmap(ts)
    #savefig("heatmap.png")
    sfs=Vector{Any}([])
    sum_of_rep=0.0
    NURS_sum=0.0
    Ncells = length(unique(nodes))
    maxt = maximum(times)
    recording_start_time = minimum(times)
    times_per_slice,nodes_per_slice,full_sliding_window_starts,full_sliding_window_ends = spike_matrix_slices(nodes,times,number_divisions,maxt,recording_start_time)
    @assert length(nodes)>0.0
    @assert length(times)>0.0
    (spikes_ragged,numb_neurons) = create_spikes_ragged(nodes,times) 
    spikes_ragged = spikes_ragged[sort_idx,:]

    step_size = maxt/number_divisions
    div_spike_mat_with_displacement,sws,ews,window_size = spike_matrix_divided(spikes_ragged, step_size,number_divisions, maxt,recording_start_time ;displace=true,sliding=true)
    @assert size(div_spike_mat_with_displacement)[1] > number_divisions

    #=
    @time (tt,nn) = compare_every_time_block_to_all_others(div_spike_mat_with_displacement)
    @show(tt)
    #@infiltrate
    ts = get_ts(nn,tt,dt,tau)#;disk=false)
    normalize!(ts)
    classes = 10
    R = kmeans(ts', classes; maxiter=2000, display=:iter)
    a = assignments(R) # get the assignments of points to clusters

    sort_idx =  sortperm(assignments(R))
    ts = ts[sort_idx,:]
    Plots.heatmap(ts)
    savefig("sorted.png")
    =#
    #time_surface(tt,nn)
        #for (indj,rowj) enumerate(eachcol(div_spike_mat_with_displacement'))

    #@assert size(div_spike_mat_with_displacement)[2] == number_divisions
    #@show(Ncells)
    #@show(Int64(numb_neurons))
    #@show(size(div_spike_mat_with_displacement)[1])
    @assert size(div_spike_mat_with_displacement)[1] == numb_neurons
    #Plots.heatmap(distmat)
    #ylabel!("Number cells: $Ncells")
    #xlabel!("Number of windows: $number_divisions.$window_size")
    #savefig("bug_fixedKreuz.png")
    #(distmat,c_variance) = compute_metrics_on_matrix_divisions(div_spike_mat_with_displacement,times_per_slice,ews,step_size,metric=:count)
    #(distmat,lv_variance) = compute_metrics_on_matrix_divisions(div_spike_mat_with_displacement,times_per_slice,ews,step_size,metric=:LV)
    (distmatk,k_variance) = compute_metrics_on_matrix_divisions(div_spike_mat_with_displacement,times_per_slice,ews,step_size,metric=:kreuz)

    #t  = range(0, stop=1, step=1/10)
    #y0 = sin.(2pi .* t)
    #T  = [randn(20); y0; randn(20); y0; randn(20)]
    #window_length = length(y0)
    #window_length= length()
    #@infiltrate


    #function slicematrix(A::AbstractMatrix)
    #    return [A[i, :] for i in 1:size(A,1)]
    #end
    #distmatk=convert(Matrix{Float64},distmatk)

    #temp1 = slicematrix(distmatk)
    #@show(typeof(temp1))
    #temp1 = [ copy(row)[:] ; for row in eachrow(distmatk)]

    #A = Array{Float64, 2}(undef, size(distmatk)[1], size(distmatk)[2]) 

    #for (ind,col) in enumerate(eachcol(distmatk))
    #    A[:,ind] = col 
    #end

    #for 
    #T  = [randn(20); y0; randn(20); y0; randn(20)]
    #end
    #temp1 = [copy(col)[:] for col in eachcol(absarr)]
    #@infiltrate
    #profile = matrix_profile(A, Int64(trunc(round(step_size))))
    #reshape(A)
    
    #df = DataFrame([c_variance,lv_variance,k_variance], Vector{Any}([:c_variance,:lv_variance,:k_variance]))
    Plots.heatmap(distmatk)
    ylabel!("Number cells: $Ncells")
    xlabel!("Number of windows: $number_divisions.$window_size")
    savefig("bug_fixedKreuz.png")

    (distmat,k_variance) = compute_metrics_on_matrix_divisions(div_spike_mat_with_displacement,ews,step_size,metric=:kreuz)
    (distmatlv,lv_variance) = compute_metrics_on_matrix_divisions(div_spike_mat_with_displacement,ews,step_size,metric=:LV)
    #function compute_metrics_on_matrix_divisions(div_spike_mat_no_displacement::Matrix{Vector{Vector{Float32}}},ews,step_size;metric=:kreuz,disk=false)

    Plots.heatmap(distmatlv)
    ylabel!("Number cells: $Ncells")
    xlabel!("Number of windows: $number_divisions.$window_size")
    savefig("bug_fixedCV.png")
    #spike_jobs = Vector{Any}([])
    classes = 10
    R = kmeans(distmat', classes; maxiter=2000, display=:iter)
    a = assignments(R) # get the assignments of points to clusters
    sort_idx =  sortperm(assignments(R))
    distmat = distmat[sort_idx,:]
    Plots.heatmap(distmat)
    savefig("xxsortedHeatmap.png")
    div_spike_mat_with_displacement = div_spike_mat_with_displacement[sort_idx,:]

    #(_,rep,_,_,NURS,state_frequency_histogram) = label_spikes(distmat,sws,ews,times,nodes,div_spike_mat_with_displacement,times_per_slice,nodes_per_slice;threshold=similarity_threshold)
    #Plots.heatmap(distmat1)
    #ylabel!("Number cells: $Ncells")
    #xlabel!("Number of windows: $number_divisions.$window_size")
    #savefig("bug_fixedLV.png")
    
    spike_jobs,a = cluster_get_jobs(distmat,spikes_ragged)
    @inbounds for (ind,spikes_ragged_packet) in enumerate(spike_jobs)
        div_spike_mat_with_displacement_local = div_spike_mat_with_displacement[a.==ind,:]
        nodes,times = ragged_to_lists(spikes_ragged_packet)
        times_per_slice,nodes_per_slice,full_sliding_window_starts,full_sliding_window_ends = spike_matrix_slices(nodes,times,number_divisions,maxt,recording_start_time)
        (new_distmat,_) = compute_metrics_on_matrix_divisions(div_spike_mat_with_displacement_local,times_per_slice,ews,step_size,metric=:kreuz)
        (_,rep,_,_,NURS,state_frequency_histogram,cat_cnt) = label_spikes(new_distmat,sws,ews,times,nodes,div_spike_mat_with_displacement_local;threshold=similarity_threshold)
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
    @show(lv_variance)
    @show(k_variance)
    @show(c_variance)
    @show(cv_variance)
    (distmat,div_spike_mat_with_displacement,spikes_ragged,NURS_sum,sum_of_rep,sfs,times_per_slice)
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

function more_plotting0(distmat,div_spike_mat_with_displacement,nodes,times,timesList)
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
        spk_countst = sum([length(times) for times in timesList[cnt]])
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

