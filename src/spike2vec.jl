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
#using Makie
#using CairoMakie
using Revise
using UMAP
using Distances
using StaticArrays
#using ColorSchemes
#using LinearAlgebra
using Clustering
#using ProgressMeter
using LoopVectorization
#using RecurrenceAnalysis
using StatsBase


"""
Augment by lengthening with duplication useful for sanity checking algorithm.
"""

function augment(ttt,nnn,scale)
    for s in scale
        maxt = maximum(ttt)
        for (i, t) in enumerate(spikes)
            for tt in t
                if length(t)!=0
                    push!(nnn,i);
                    txt = Float32(tt+maxt)
                    push!(ttt,txt)
                end
            end
        end
    end
    (nnn,ttt)
end
function lvr(time_intervals, R=5*0.001)
    # statistics-Shinomoto2009_e1000433
    # https://github.com/NeuralEnsemble/elephant/blob/master/elephant/statistics.py

    N = length(time_intervals)
    t = time_intervals[1:N-1] .+ time_intervals[2,:]
    frac1 = 4.0 * time_intervals[1:N-1].*time_intervals[2,:] ./ t.^2.0
    frac2 = 4.0 * R ./ t
    lvr = (3.0 / (N-1.0)) * sum((1.0.-frac1) .* (1.0.+frac2))
    lvr
end

function bag_of_isis(nodes::Vector{<:Integer},times::Vector{<:Real})
    spikes_ragged = Vector{Any}([])
    numb_neurons=Int(maximum(nodes))+1 # Julia doesn't index at 0.
    @inbounds for n in 1:numb_neurons
        push!(spikes_ragged,Vector{Float32}[])
    end
    @inbounds for i in 1:numb_neurons
        for (n,t) in zip(nodes,times)
            if i==n
                push!(spikes_ragged[UInt32(i)],t)
            end
        end
    end
    processed_isis = bag_of_isis(spikes_ragged)
    processed_isis::Vector{Any}
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
    #result = 
    std(isi_s)/mean(isi_s)
end


"""
On a RTSP packet, get a bag of ISIs. So we can analyse the temporal structure of RTSPs regardless of spatial structure.
"""
function bag_of_isis(spikes_ragged::AbstractArray)
    bag_of_isis = Vector{Any}([]) # the total lumped population ISI distribution.
    isi_s = Float32[]
    @inbounds for (i, times) in enumerate(spikes_ragged)
        push!(isi_s,[])
    end
    @inbounds for (i, times) in enumerate(spikes_ragged)
        
        for (ind,x) in enumerate(times)
            if ind>1
                isi_current = x-times[ind-1]
                push!(isi_s[i],isi_current)
            end
        end
        append!(bag_of_isis,Vector{Float32}(isi_s[i]))
    end
    bag_of_isis
end


"""
Divide epoch into windows
"""
function divide_epoch(nodes::AbstractVector,times::AbstractVector,start::Real,stop::Real)
    n0=Vector{UInt32}([])
    t0=Vector{Float32}([])
    window_size = stop-start
    @inbounds for (n,t) in zip(nodes,times)
        if start<=t && t<=stop
            append!(t0,t-start)
            append!(n0,n)
        end
    end
    time_raster =  Array{}([Float32[] for i in 1:maximum(nodes)+1])
    for (neuron,t) in zip(n0,t0)
        append!(time_raster[neuron],t)        
    end
    #print("gets here how?")
    time_raster
end


function get_vector_coords_uniform!(uniform::AbstractArray, neuron1::AbstractArray, self_distances::AbstractArray;metric="kreuz")

    @inbounds for (ind,n1_) in enumerate(neuron1)
        if length(n1_) != 0
            pooledspikes = vcat(uniform,n1_)
            maxt = maximum(sort!(unique(pooledspikes)))
            t1_ = sort(unique(n1_))
            if metric=="kreuz"
                _, S = SPIKE_distance_profile(t1_,uniform;t0=0,tf = maxt)
                self_distances[ind]=abs(sum(S))
            elseif metric=="CV"
                if length(t1_)>1
                    self_distances[ind] = CV(t1_)
                else
                    self_distances[ind]=0
                end                
            elseif metric=="autocov"
                if length(t1_)>1
                    self_distances[ind] = autocov( t1_, [length(t1_)-1],demean=true)[1]
                else
                    self_distances[ind]=0
                end

            elseif metric=="LV"
                if length(t1_)>1

                    self_distances[ind]=lvr(t1_,maximum(t1_))
                else
                    self_distances[ind]=0
                end
                    
            end

        else
            self_distances[ind]=0
        end
    end
end
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
"""
Just a helper method to get some locally stored spike data if it exists.
"""
function fromHDF5spikes()
    hf5 = h5open("spikes.h5","r")
    nodes = Vector{Int64}(read(hf5["spikes"]["v1"]["node_ids"]))
    nodes = [n+1 for n in nodes]
    times = Vector{Float64}(read(hf5["spikes"]["v1"]["timestamps"]))
    close(hf5)
    (times,nodes)
end
#=
function final_plots2(mat_of_distances)
    angles0,distances0,angles1,distances1 = post_proc_viz(mat_of_distances)
    plot!(angles1,distances1,marker =:circle, arrow=(:closed, 3.0)) 
    savefig("statemvements_nmn.png")   
    (angles1,distances1)
end
=#
function divide_epoch(times::AbstractVector,start::Real,stop::Real)
    t0=Vector{Float32}([])
    window_size = stop-start
    for t in times
        if start<=t && t<=stop
            append!(t0,t-start)
            #@show(typeof(last(t0)))
        end

    end
    #@show(typeof(t0))

    t0::Vector{Float32}
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
    end_windows = collect(step_size:step_size:step_size*number_divisions_size)
    start_windows = collect(0:step_size:(step_size*number_divisions_size)-step_size)
    #T = 
    mat_of_spikes = array_of_empty_vectors(Vector{Float32},(length(spikes_raster),length(end_windows)))
    spike_matrix_divided!(mat_of_spikes,spikes_raster,step_size,end_windows,start_windows,displace)
    mat_of_spikes#:::Matrix{Vector{Vector{Float32}}}
end
function spike_matrix_divided!(mat_of_spikes::Matrix{Vector{Vector{Float32}}},spikes_raster,step_size,end_windows,start_windows,displace)
    for neuron_id in 1:length(spikes_raster)
        for (windex,final_end_time) in enumerate(end_windows)
            sw = start_windows[windex]
            only_one_neuron_spike_times = copy(spikes_raster[neuron_id])
            epoch_windowed_spikes = divide_epoch(only_one_neuron_spike_times,sw,final_end_time)
            if displace
                push!(mat_of_spikes[neuron_id,windex],epoch_windowed_spikes.+sw)
            else
                push!(mat_of_spikes[neuron_id,windex],epoch_windowed_spikes)
            end
        end
    end
    #mat_of_spikes::Matrix{Vector{Vector{Float32}}}
end
#=
sw: start window a length used to filter spikes.
windex: current window index
times_associated: times (associated with) indices, before stop window length applied
"""
function spike_matrix_divided_nd(spikes_raster::Vector{Any},number_divisions_size::Int,numb_neurons::Int,maxt::Real)
    step_size = maxt/number_divisions_size
    end_windows = collect(step_size:step_size:step_size*number_divisions_size)
    start_windows = collect(0:step_size:(step_size*number_divisions_size)-step_size)
    T = Vector{Float32}
    mat_of_spikes = array_of_empty_vectors(T,(length(spikes_raster),length(end_windows)))
    spike_matrix_divided!(mat_of_spikes,step_size,end_windows,start_windows,T)
    mat_of_spikes::Array{Vector{T}}
end
=#
#for neuron_id in 1:length(spikes_raster)
#    for (windex,times_associated) in enumerate(end_windows) # toi: times of (associated with) indices.
#        sw = start_windows[windex] # sw: start window
#        epoch_length_spikes = divide_epoch(copy(spikes_raster[neuron_id]),sw,times_associated)
#        push!(mat_of_spikes[neuron_id,windex],epoch_length_spikes)
#    end
#end

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

function get_divisions(nodes::Vector{UInt32},times::Vector{Float64},division_size::Int,numb_neurons::Integer,maxt::Real;plot=false,file_name::String="stateTransMat.png",metric="kreuz",disk=false)
    step_size = maxt/division_size
    end_windows = Vector{Float64}(collect(step_size:step_size:step_size*division_size))
    spike_distance_size = length(end_windows)
    start_windows = Vector{Float64}(collect(0:step_size:(step_size*division_size)-step_size))
    # This can be made an MMAP arrray instead.
    if !disk
        mat_of_distances = Array{Float64}(undef, numb_neurons, spike_distance_size)

    else
        io = open("/tmp/mmap.bin", "w+")
        # We'll write the dimensions of the array as the first two Ints in the file
        mat_of_distances = mmap(io, Matrix{Float32}, (numb_neurons,spike_distance_size));
    end
    refspikes = divide_epoch(nodes,times,start_windows[2],end_windows[2])
    segment_length = end_windows[2] - start_windows[2]
    max_spk_counts = Int32(round(mean([length(times) for times in enumerate(refspikes[:])])))
    temp = LinRange(0.0, maximum(times), max_spk_counts)
    linear_uniform_spikes = Vector{Float64}([i for i in temp[:]])
    nlist = Array{Vector{UInt32}}([])
    tlist = Array{Vector{Float64}}([])
    @inbounds @showprogress for (ind,toi) in enumerate(end_windows)
        sw = start_windows[ind]
        observed_spikes = divide_epoch(nodes,times,sw,toi)    
        self_distances = Vector{Float64}(zeros(numb_neurons))
        get_vector_coords_uniform!(linear_uniform_spikes, observed_spikes, self_distances; metric=metric)
        mat_of_distances[:,ind] = self_distances
        if disk
            Mmap.sync!(mat_of_distances) 
        end
        get_window!(nlist,tlist,observed_spikes,sw)
    end
    mat_of_distances[isnan.(mat_of_distances)] .= 0.0
    @inbounds for (ind,col) in enumerate(eachcol(mat_of_distances))
        mat_of_distances[:,ind] .= (col.-mean(col))./std(col)
    end
    mat_of_distances[isnan.(mat_of_distances)] .= 0.0
    if plot
        Plots.heatmap(mat_of_distances)
        savefig("Unormalised_heatmap_$metric.$file_name.png")
    end
    if disk
        Mmap.sync!(mat_of_distances)        
    end
    (mat_of_distances,tlist,nlist,start_windows,end_windows,spike_distance_size)
end


function plot_umap_of_dist_vect(mat_of_distances; file_name::String="stateTransMat.png")
    Q_embedding = umap(mat_of_distances',20,n_neighbors=20)
    Plots.plot(Plots.scatter(Q_embedding[1,:], Q_embedding[2,:], title="Spike Time Distance UMAP, reduced precision,", marker=(1, 1, :auto, stroke(0.05)),legend=true))
    savefig(file_name)
    Q_embedding
end
function label_online_distmat!(mat_of_distances::AbstractVecOrMat,distance_matrix;threshold::Real=5)
    @inbounds @showprogress for (ind,row) in enumerate(eachcol(mat_of_distances))
        stop_at_half = Int(trunc(length(eachcol(mat_of_distances))/2))
        #if ind <= stop_at_half
        @inbounds for (ind2,row2) in enumerate(eachcol(mat_of_distances))
            if ind!=ind2
                distance = evaluate(Euclidean(),row,row2)
                if distance<threshold
                    distance_matrix[ind,ind2] = abs(distance)
                end
            end
        end
    end
end

function label_online_distmat(mat_of_distances::AbstractVecOrMat;threshold::Real=5, disk=false)
    if !disk
        distance_matrix = zeros(length(eachcol(mat_of_distances)),length(eachcol(mat_of_distances)))
    else
        io = open("/tmp/mmap.bin", "w+")
        distance_matrix = mmap(io, Matrix{Float32}, (length(eachcol(mat_of_distances)),length(eachcol(mat_of_distances))))

    end
    label_online_distmat!(mat_of_distances,distance_matrix;threshold)
    distance_matrix::AbstractVecOrMat
end


function cluster_distmat(mat_of_distances::AbstractMatrix)
    R = affinityprop(mat_of_distances)
    sort_idx =  sortperm(assignments(R))
    assign = R.assignments
    R,sort_idx,assign
end

function horizontal_sort_into_tasks(mat_of_distances::AbstractArray)
    R = affinityprop(mat_of_distances')
    sort_idx =  sortperm(assignments(R))
    assign = R.assignments
    R,sort_idx,assign
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
function get_division_scatter_identify_via_recurrence_mat(mat_of_distances,assign,nlist,tlist,start_windows,end_windows,nodes,times,ε::Real=5)
    sss =  StateSpaceSet(hcat(mat_of_distances))
    R = RecurrenceMatrix(sss, ε; metric = Euclidean(), parallel=true)
    xs, ys = RecurrenceAnalysis.coordinates(R)# -> xs, ys
    #network = RecurrenceAnalysis.SimpleGraph(R)
    #graphplot(network)
    #savefig("sanity_check_markov.png")
    #p=Plots.scatter()
    return rqa(R),xs, ys,sss,R
end

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

    @inbounds for row in eachrow(distmat)
        @inbounds for (ii,xx) in enumerate(row)
            #sw_old = -1
            if abs(xx)<threshold
                sw = start_windows[ii]
               # if sw!=sw_old
                push!(assing_progressions,assign[ii])
                push!(assing_progressions_times,sw)
                #end
                #sw_old = sw
    
            end
        end
    end
    assing_progressions,assing_progressions_times
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
                #push!(repeated_windows,ind)
                push!(repeated_windows,y)

            end 
        end
    end
    
    #end
    if plot
        (n,m) = size(stateTransMat)
        
        #cols = columns(stateTransMat)
        Plots.heatmap(cor(stateTransMat), fc=cgrad([:white,:dodgerblue4]), xticks=(1:n,m), xrot=90, yticks=(1:m,n), yflip=true)
        Plots.annotate!([(j, i, (round(stateTransMat[i,j],digits=3), 8,"Computer Modern",:black)) for i in 1:n for j in 1:m])
        #Plots.heatmap(stateTransMat)
        savefig("corr_state_transition_matrix$file_name.png")
        Plots.heatmap(stateTransMat, fc=cgrad([:white,:dodgerblue4]), xticks=(1:n,m), xrot=90, yticks=(1:m,n), yflip=true)
        savefig("state_transition_matrix$file_name.png")

    end
    assing_progressions[unique(i -> assing_progressions[i], 1:length(assing_progressions))].=-1
    assing_progressions[unique(i -> assing_progressions[i], 1:length(assing_progressions))].=-1
    if plot

        p1 = Plots.plot()
        Plots.scatter!(p1,assing_progressions_times,assing_progressions,legend=false)
        #https://github.com/open-risk/transitionMatrix

        #Plots.plot!(p1,assing_progressions,assing_progressions_times,legend=false)
        #display(p1)
        savefig("state_transition_trajectory$file_name.png")
        if false
            g = SimpleWeightedDiGraph(stateTransMat)

            edge_label = Dict((i,j) => string(stateTransMat[i,j]) for i in 1:size(stateTransMat, 1), j in 1:size(stateTransMat, 2))

            graphplot(g; names = 1:length(stateTransMat), weights=stateTransMat)
            savefig("state_transition_graph$file_name.png")
        end
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

"""
Using the windowed spike trains for neuron0: a uniform surrogate spike train reference, versus neuron1: a real spike train in the  target window.
compute the intergrated spike distance quantity in that time frame.

SpikeSynchrony is a Julia package for computing spike train distances just like elephant in python
And in every window I get the population state vector by comparing current window to uniform spiking windows
But it's also a good idea to use the networks most recently past windows as reference windows 

"""
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