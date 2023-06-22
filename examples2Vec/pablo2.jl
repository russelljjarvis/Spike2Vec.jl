using JLD
#using SpikeTime
#using
#using DrWatson
using Plots
using SpikeTime
#ST = SpikeTime.ST
#using ST
#using OnlineStats
#using SparseArrays
#SNN.@load_units
#using Test
using DimensionalData
using Revise
using StatsBase
using ProgressMeter
using ColorSchemes
using PyCall
using LinearAlgebra
using Makie
using JLD
using CairoMakie#,KernelDensity, Distributions
using ProgressMeter
using Distances
#using Gadfly
#using Gadfly
#using SGtSNEpi, Random

using SimpleWeightedGraphs, Graphs
#using GraphPlot
using Plots, GraphRecipes

using Plots
#using PyCall
using UMAP
using Plots
# Other Imports
#import PyPlot: plt
import DelimitedFiles: readdlm
#import Random
#import StatsBase: quantile
using Clustering
using UMAP
# Songbird metadata
#num_neurons = 77
#max_time = 22.2
using DrWatson
# Randomly permute neuron labels.
# (This hides the sequences, to make things interesting.)
#_p = Random.randperm(num_neurons)

# Load spikes.
#spikes = seq.Spike[]
function ragged_to_uniform(nodes,times)
    nnn=Vector{Int32}([])
    ttt=Vector{Float32}([])
    for (i, t) in enumerate(times)
        for tt in t
            if length(t)!=0
                push!(nnn,i);
                push!(ttt,Float32(tt))
            end
        end
    end
    (nnn,ttt)
end
function load_datasets()

    py"""
    import pickle
    def get_pablo_sim():

        temp = pickle.load(open("pablo_conv.p","rb"))
        return (temp[0],temp[1])
    """
    (nodes,times) = py"get_pablo_sim"()
    (nodes,times) = ragged_to_uniform(nodes,times)
    return (nodes,times)
end

function get_plot(times,nodes,division_size)
    step_size = maximum(times)/division_size
    @show(step_size)
    end_window = collect(step_size:step_size:step_size*division_size)
    spike_distance_size = length(end_window)
    start_windows = collect(0:step_size:(step_size*division_size)-step_size)

    mat_of_distances = zeros(spike_distance_size,maximum(unique(nodes))+1)
    
    @show(last(start_windows),last(end_window))
    n0ref = divide_epoch(nodes,times,last(start_windows),last(end_window))
    
    segment_length = end_window[2] - start_windows[2]
    @show(segment_length)
    mean_spk_counts = Int32(round(mean([length(times) for times in enumerate(n0ref)])))
    t0ref = surrogate_to_uniform(n0ref,segment_length,mean_spk_counts)
    PP = []
    temp = LinRange(0.0, maximum(times), mean_spk_counts)
    linear_uniform_spikes = Vector{Float32}([i for i in temp])

    @showprogress for (ind,toi) in enumerate(end_window)
        sw = start_windows[ind]
        neuron0 = divide_epoch(nodes,times,sw,toi)    
        self_distances = Array{Float32}(zeros(maximum(nodes)+1))

        get_vector_coords_uniform!(linear_uniform_spikes, neuron0, self_distances)
        mat_of_distances[ind,:] = self_distances
    end
    
    cs1 = ColorScheme(distinguishable_colors(spike_distance_size, transform=protanopic))
    mat_of_distances[isnan.(mat_of_distances)] .= 0.0
    Plots.heatmap(mat_of_distances)#, axis=(xticks=(1:7, xs), yticks=(1:10, ys), xticklabelrotation = pi/4) ))
    savefig("Unormalised_heatmap_song_bird.png")
    mat_of_distances[isnan.(mat_of_distances)] .= 0.0

    @inbounds @showprogress for (ind,col) in enumerate(eachcol(mat_of_distances))
        mat_of_distances[:,ind] .= (col.-mean(col))./std(col)
    end
    mat_of_distances[isnan.(mat_of_distances)] .= 0.0

    display(mat_of_distances)
    return mat_of_distances
end

function plot_umap(mat_of_distances; file_name::String="empty.png")

    Q_embedding = umap(mat_of_distances',20,n_neighbors=20)#, min_dist=0.01, n_epochs=100)
    display(Plots.plot(Plots.scatter(Q_embedding[1,:], Q_embedding[2,:], title="Spike Time Distance UMAP, reduced precision", marker=(1, 1, :auto, stroke(0.07)),legend=true)))
    #Plots.plot(scatter!(p,model.knns)
    savefig(file_name)
    Q_embedding
end

function label_online_distmat(mat_of_distances)#,nclasses)

    distance_matrix = zeros(length(eachrow(mat_of_distances)),length(eachrow(mat_of_distances)))
    display(distance_matrix)
    all_perm_pairs = []


    @showprogress for (ind,row) in enumerate(eachrow(mat_of_distances))
        push!(all_perm_pairs,[])
        for (ind2,row2) in enumerate(eachrow(mat_of_distances))
            best_distance = 100000.0
            distance = evaluate(Euclidean(),row,row2)
            if distance<7
                push!(all_perm_pairs[ind],ind2)
                distance_matrix[ind,ind2] = distance
            else
                distance_matrix[ind,ind2] = -10.0
            end
        end
    end
    display(Plots.heatmap(distance_matrix))
    #@show(all_perm_pairs)
    distance_matrix
 end
function cluster_distmat(mat_of_distances)
    display(mat_of_distances)
    R = affinityprop(mat_of_distances)
    sort_idx =  sortperm(assignments(R))
    assign = R.assignments
    R,sort_idx,assign
end

(nnn,ttt)= load_datasets()
#display(Plots.scatter(ttt,nnn))
resolution = 90
mat_of_distances = get_plot(ttt,nnn,resolution)
display(mat_of_distances)
distmat = label_online_distmat(mat_of_distances)#,nclasses)
display(distmat)
(R,sort_idx,assign) = cluster_distmat(distmat)
display(Plots.heatmap(distmat))
function get_division_scatter2(times,nodes,division_size,distmat,sort_idx,assign)
    step_size = maximum(times)/division_size
    end_window = collect(step_size:step_size:step_size*division_size)
    spike_distance_size = length(end_window)
    start_windows = collect(0:step_size:(step_size*division_size)-step_size)

    mat_of_distances = zeros(spike_distance_size,maximum(unique(nodes))+1)
    segment_length = end_window[3] - start_windows[3]
    fig = Figure(backgroundcolor=RGBf(0.6, 0.6, 0.96))

    p=Plots.scatter(backgroundcolor=RGBf(0.6, 0.6, 0.96))
    witnessed_unique=[]
    @showprogress for (ind,toi) in enumerate(end_window)
        sw = start_windows[ind]
        spikes = divide_epoch(nodes,times,sw,toi)    
        Nx=Vector{Int32}([])
        Tx=Vector{Float32}([])
        for (i, t) in enumerate(spikes)
            for tt in t
                if length(t)!=0
                    push!(Nx,i)
                    push!(Tx,Float32(sw+tt))
                end
            end
        end
        for (ii,xx) in enumerate(distmat[ind,:])
            if abs(xx)<7
                push!(witnessed_unique,assign[ii])
                
                Plots.scatter!(p,Tx,Nx, markercolor=assign[ii],legend = false, markersize = 2.0,markerstrokewidth=0,alpha=1.0, bgcolor=:snow2, fontcolor=:blue)#, marker=:auto)#,group=categorical(length(unique(sort_idx))))
             end
        end


    end
    nunique = length(unique(witnessed_unique))

    Plots.scatter!(p,xlabel="time (ms)",ylabel="Neuron ID", yguidefontcolor=:black, xguidefontcolor=:black,title = "N observed states $nunique")##,backgroundcolor=RGBf(0.6, 0.6, 0.96))
    p2 = Plots.scatter(times,nodes,legend = false, markersize = 1.9,markerstrokewidth=0,alpha=0.7, bgcolor=:snow2, fontcolor=:blue,thickness_scaling = 1)#,group=categorical(length(unique(sort_idx))))
    Plots.scatter!(p2,xlabel="time (ms)",ylabel="Neuron ID", yguidefontcolor=:black, xguidefontcolor=:black,title = "Un-labeled spike raster")##,backgroundcolor=RGBf(0.6, 0.6, 0.96))

    Plots.plot(p,p2, layout = (2, 1))

    savefig("repeated_pattern_song_bird2.png")
end
get_division_scatter2(ttt,nnn,resolution,distmat,sort_idx,assign)

function get_state_transitions(times,nodes,division_size,distmat,sort_idx,assign)
    step_size = maximum(times)/division_size
    end_window = collect(step_size:step_size:step_size*division_size)
    spike_distance_size = length(end_window)
    start_windows = collect(0:step_size:(step_size*division_size)-step_size)
    nunique = length(unique(assign))
    assing_progressions=[]
    assing_progressions_times=[]

    @showprogress for (ind,toi) in enumerate(end_window)
        sw = start_windows[ind]
        push!(assing_progressions_times,sw)
    end
    for row in eachrow(distmat)
        for (ii,xx) in enumerate(row)
            if abs(xx)<7
                push!(assing_progressions,assign[ii])
 
            end
        end
    end
    assing_progressions,assing_progressions_times
end
get_division_scatter2(ttt,nnn,resolution,distmat,sort_idx,assign)
assing_progressions,assing_progressions_times = get_state_transitions(ttt,nnn,resolution,distmat,sort_idx,assign)


#using SimpleWeightedGraphs, Graphs
nunique = length(unique(assign))
#using Plots, GraphRecipes
empty = zeros(nunique,nunique)
for (ind,x) in enumerate(assing_progressions)
    if ind < length(assing_progressions)
        empty[x,assing_progressions[ind+1]]+=1
    end 
end
repeated_windows = Vector{UInt32}([])
repititions = zeros(size(empty))
for (ind,_) in enumerate(eachrow(empty))
    for (y,val) in enumerate(empty[ind,:])
        if val==1
            #@show(ind,y)
            repititions[ind,y] = 0.0

        elseif val>1
            repititions[ind,y] = val 

            push!(repeated_windows,ind)


        end 

    end
end

#@show(store_non_zero)
display(Plots.heatmap(empty))

assing_progressions[unique(i -> assing_progressions[i], 1:length(assing_progressions))].=-1
p1 =Plots.plot()
Plots.scatter!(p1,assing_progressions,legend=false)
#https://github.com/open-risk/transitionMatrix

Plots.plot!(p1,assing_progressions,legend=false)
display(p1)
savefig("state_transition_trajectory.png")
g = SimpleWeightedDiGraph(empty)

edge_label = Dict((i,j) => string(empty[i,j]) for i in 1:size(empty, 1), j in 1:size(empty, 2))

graphplot(g; names = 1:length(empty), weights=empty)#,line_z=empty)#, edge_label)
savefig("state_transition_matrix.png")

function get_repeated_scatter(times,nodes,division_size,repeated_windows)
    step_size = maximum(times)/division_size
    end_window = collect(step_size:step_size:step_size*division_size)
    spike_distance_size = length(end_window)
    start_windows = collect(0:step_size:(step_size*division_size)-step_size)

    segment_length = end_window[3] - start_windows[3]
    fig = Figure(backgroundcolor=RGBf(0.6, 0.6, 0.96))

    p=Plots.scatter(backgroundcolor=RGBf(0.6, 0.6, 0.96))
    @showprogress for (ind,toi) in enumerate(end_window)

    #for (ind,repeat_window) in enumerate(repeated_windows)
        if repeated_windows[ind]!=-1
            toi = end_window[ind]
            sw = start_windows[ind]
            spikes = divide_epoch(nodes,times,sw,toi)    
            Nx=Vector{Int32}([])
            Tx=Vector{Float32}([])
            for (i, t) in enumerate(spikes)
                for tt in t
                    if length(t)!=0
                        push!(Nx,i)
                        push!(Tx,Float32(sw+tt))
                    end
                end
            end
            Plots.scatter!(p,Tx,Nx,legend = false, markercolor=repeated_windows[ind],markersize = 2.0,markerstrokewidth=0,alpha=1.0, bgcolor=:snow2, fontcolor=:blue, xlims=(0, maximum(times)))
        end

    end
    nunique = length(unique(repeated_windows))

    #
    Plots.scatter!(p,xlabel="time (ms)",ylabel="Neuron ID", yguidefontcolor=:black, xguidefontcolor=:black,title = "N observed states $nunique", xlims=(0, 22.2))##,backgroundcolor=RGBf(0.6, 0.6, 0.96))
    #Plots.plot!(p,
    #titlefont = font(0.017, "Courier")
    #)
    #display(p)R,M,sort_idx
    #end
    p2 = Plots.scatter(times,nodes,legend = false, markersize = 1.9,markerstrokewidth=0,alpha=0.7, bgcolor=:snow2, fontcolor=:blue,thickness_scaling = 1)#,group=categorical(length(unique(sort_idx))))
    Plots.scatter!(p2,xlabel="time (ms)",ylabel="Neuron ID", yguidefontcolor=:black, xguidefontcolor=:black,title = "Un-labeled spike raster")##,backgroundcolor=RGBf(0.6, 0.6, 0.96))

    Plots.plot(p,p2, layout = (2, 1))

    savefig("genuinely_repeated_pattern_song_bird.png")
end
@show(assing_progressions)
get_repeated_scatter(ttt,nnn,resolution,assing_progressions)
#assing_progressions[(i -> assing_progressions[i]==0, 1:length(assing_progressions))].=0

display(Plots.scatter(assing_progressions))

