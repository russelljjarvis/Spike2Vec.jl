using JLD
import JLD.save
#using SpikeTime
using DynamicalSystems
import Attractors#.DiscreteDynamicalSystem
import Attractors#.AttractorsViaRecurrence
using DelayEmbeddings

using Attractors
using StaticArrays
#using
#using DrWatson
#using Plots
using SpikeTime
#import Attractors.DiscreteDynamicalSystem
#import Attractors.AttractorsViaRecurrences
#ST = SpikeTime.ST
#using ST
#using OnlineStats
#using SparseArrays
#SNN.@load_units
#using Test
using Revise
using StatsBase
using ProgressMeter
using ColorSchemes
using PyCall
using LinearAlgebra
using Makie
#using JLD
#using CairoMakie#,KernelDensity, Distributions
using ProgressMeter
using Distances
#using Gadfly
#using Gadfly
#using SGtSNEpi, Random
using Attractors
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
#num_neurons = 75
#max_time = 22.2

# Randomly permute neuron labels.
# (This hides the sequences, to make things interesting.)
#_p = Random.randperm(num_neurons)

# Load spikes.
#spikes = seq.Spike[]
function load_datasets()

    spikes = []
    
    nodes = [n for (n, t) in eachrow(readdlm("../data/songbird_spikes.txt", '\t', Float64, '\n'))]
    for _ in 1:maximum(unique(nodes))+1
        push!(spikes,[])
    end

    for (n, t) in eachrow(readdlm("../data/songbird_spikes.txt", '\t', Float64, '\n'))
        push!(spikes[Int32(n)],t)
    end
    #@show(spikes)
    nnn=Vector{Int32}([])
    ttt=Vector{Float32}([])
    for (i, t) in enumerate(spikes)
        for tt in t
            if length(t)!=0
                push!(nnn,i);
                push!(ttt,Float32(tt))
            end
        end
    end
    maxt = maximum(ttt)
    for (i, t) in enumerate(spikes)
        for tt in t
            if length(t)!=0
                push!(nnn,i);
                txt = Float32(tt+maxt)
                push!(ttt,txt)
                #@show(i,txt)
            end
        end
    end
    maxt = maximum(ttt)
    for (i, t) in enumerate(spikes)
        for tt in t
            if length(t)!=0
                push!(nnn,i);
                txt = Float32(tt+maxt)
                push!(ttt,txt)
                #@show(i,txt)
            end
        end
    end

    (nnn,ttt)
    #(spikes,nnn)
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
    #mean_spk_counts = Int32(round(mean(non_zero_spikings)))
    temp = LinRange(0.0, maximum(times), mean_spk_counts)
    linear_uniform_spikes = Vector{Float32}([i for i in temp])

    @showprogress for (ind,toi) in enumerate(end_window)
        sw = start_windows[ind]
        neuron0 = divide_epoch(nodes,times,sw,toi)    
        self_distances = Array{Float32}(zeros(maximum(nodes)+1))

        get_vector_coords_uniform!(linear_uniform_spikes, neuron0, self_distances)
        #self_distances = get_vector_coords(neuron0,t0ref,self_distances)
        mat_of_distances[ind,:] = self_distances
    end
    cs1 = ColorScheme(distinguishable_colors(spike_distance_size, transform=protanopic))
    mat_of_distances[isnan.(mat_of_distances)] .= 0.0
    Plots.heatmap(mat_of_distances)#, axis=(xticks=(1:5, xs), yticks=(1:10, ys), xticklabelrotation = pi/4) ))
    savefig("Unormalised_heatmap_song_bird.png")
    @inbounds @showprogress for (ind,col) in enumerate(eachcol(mat_of_distances))
        mat_of_distances[:,ind] .= (col.-mean(col))./std(col)
    end
    mat_of_distances[isnan.(mat_of_distances)] .= 0.0
    Plots.heatmap(mat_of_distances)
    savefig("Normalised_heatmap_song_bird.png")
    p=nothing
    p = Plots.plot()
    Plots.plot!(p,mat_of_distances[1,:],label="1")#, fmt = :svg)
    Plots.plot!(p,mat_of_distances[9,:],label="2")#, fmt = :svg)
    savefig("just_two_song_bird_raw_vectors.png")
    #@save "song_bird_matrix.jld" mat_of_distances
    #zeros(spike_distance_size,maximum(unique(nodes))+1)
    return SMatrix{maximum(unique(nodes))+1,spike_distance_size}(mat_of_distances)
end
#=
function post_proc_viz(mat_of_distances)
    # ] add https://github.com/JeffreySarnoff/AngleBetweenVectors.jl
    # ] add Distances
    cs1 = ColorScheme(distinguishable_colors(length(mat_of_distances), transform=protanopic))

    angles0 = []
    distances0 = []
    @inbounds @showprogress for (ind,self_distances) in enumerate(eachrow(mat_of_distances))
        temp0 = mat_of_distances[ind,:]
        temp1 = Vector{Float32}([2.0 for i in 1:length(temp)])
        θ = angle(temp0,temp1)
        r = evaluate(Euclidean(),temp0, temp1)
        append!(angles0,θ)
        append!(distances0,r)        
    end
    @save "distances_angles_song_bird.jld" angles0 distances0#,angles1,distances1
    return angles0,distances0#,angles1,distances1)
end
=##=
function song_bird_plots(mat_of_distances)
    R = kmeans(mat_of_distances, 5; maxiter=100, display=:iter)
    @assert nclusters(R) == 5 # verify the number of clusters
    a = assignments(R) # get the assignments of points to clusters
    c = counts(R) # get the cluster sizes
    M = R.centers # get the cluster centers
    #@show(sizeof(M))
    #savefig("didit_worksong_bird.png")
    return M
end
function final_plots2(distances0,angles0,ll)
    p=nothing
    p = Plots.plot()
    #angles0,distances0 = post_proc_viz(mat_of_distances)
    pp = Gadfly.Plots.plot(distances0,angles0,Geom.point)#marker =:circle, arrow=(:closed, 3.0)))
    img = SVG("iris_plot.svg", 6inch, 4inch)
    draw(img, p)
    #savefig("relative_to_uniform_reference_song_bird.png")   
end
#(nodes,times) = load_datasets()
=#

function plot_umap(mat_of_distances; file_name::String="empty.png")
    #model = UMAP_(mat_of_distances', 10)
    #Q_embedding = transform(model, amatrix')
    #cs1 = ColorScheme(distinguishable_colors(length(ll), transform=protanopic))

    Q_embedding = umap(mat_of_distances',20,n_neighbors=20)#, min_dist=0.01, n_epochs=100)
    display(Plots.plot(Plots.scatter(Q_embedding[1,:], Q_embedding[2,:], title="Spike Time Distance UMAP, reduced precision", marker=(1, 1, :auto, stroke(0.05)),legend=true)))
    #Plots.plot(scatter!(p,model.knns)
    savefig(file_name)
    Q_embedding
end
#@load "ll.jld" ll
#@save "distances_angles_song_bird.jld" angles0 distances0
#@load "song_bird_matrix.jld" mat_of_distances
#mat_of_distances = mat_of_distances[:,1:10000]



function label_online(mat_of_distances)
    classes = 10
    R = kmeans(mat_of_distances', classes; maxiter=2000, display=:iter)
    #a = assignments(R) # get the assignments of points to clusters
    #c = counts(R) # get the cluster sizes
    M = R.centers # get the cluster centers
    #M = copy(M'[:])

    #c = kmeans(M,4)
    sort_idx =  sortperm(assignments(R))
    M = mat_of_distances'[:,sort_idx]
    p1=Plots.heatmap(M')
    savefig("cluster_sort_song_birds.png")

    p2=Plots.heatmap(mat_of_distances)
    Plots.plot(p1,p2)
    savefig("cluster_sort_song_birds.png")

    labelled_mat_of_distances = zeros(size(mat_of_distances))
    ref_mat_of_distances = zeros(size(mat_of_distances))

    scatter_indexs = []
    scatter_class_colors = []
    yes = []
    #mat_of_distances = copy(mat_of_distances'[:])
    @showprogress for (ind,row) in enumerate(eachrow(mat_of_distances))
        flag = false
        for ix in collect(1:classes)
            best_distance = 100000.0
            current_centre = M[:,ix]
            distance = sum(abs.(row .- current_centre))
            if distance<best_distance
                best_distance = distance
                best_index = ind
                best_index_class = ix
            end
            #if best_distance < 6.5
            if best_distance < 27.5

                flag = true
                labelled_mat_of_distances[best_index,:] .= best_index_class
                ref_mat_of_distances[best_index,:] = view(mat_of_distances,best_index,:)
                push!(scatter_indexs,best_index)
                push!(scatter_class_colors,best_index_class)
            else 
                flag = false
            end
        if flag
            push!(yes,last(best_index_class))
        else
            push!(yes,-1.0)
        end
        flag = false
        end
    end
    #@show(length(yes))
    #zerotonan(x) = x == 0 ? NaN : x
    #c = cgrad([:blue,:white,:red])
    p1 = Plots.heatmap(mat_of_distances,legend = :none,c = cgrad([:white,:red,:blue]))
    #savefig("reference_labelled_mat_of_distances_song_bird.png")
    p2 = Plots.heatmap(labelled_mat_of_distances,legend = :none, c = cgrad([:white,:red,:blue]))
    

    p4 = Plots.heatmap(ref_mat_of_distances[sort_idx,:],legend = :none, c = cgrad([:white,:red,:blue]))

    p3 = Plots.plot()
    prev = 0
    for (ix_,iy_) in zip(scatter_indexs,scatter_class_colors)
        
        temp0 = mat_of_distances[ix_,:].-mean(mat_of_distances[ix_,:])
        temp0 = (temp0./std(temp0)).+ix_
        #temp1 = [ix_ for i in collect(1:length(mat_of_distances[ix_,:])) ]  
        #@show(length(temp1))

        #@show(length(temp0))
        Plots.plot!(p3,temp0,legend = :none,color=iy_)
        #prev = 
    end
    #display(p3)
    #Plots.plot(scatter(ll, temp, marker_z=R.assignments, legend=true))

    Plots.plot(p1,p2,p3,p4,legend = :none)
    savefig("both_labelled_mat_of_distances_song_bird.png")
    @show(yes)
    #return M
    (scatter_indexs,yes,sort_idx)
end

(nnn,ttt)= load_datasets()
display(Plots.scatter(ttt,nnn))
resolution = 90
mat_of_distances = get_plot(ttt,nnn,resolution)
sss = StateSpaceSet(mat_of_distances)
embed = embed(sss)
@show(sss)
#N = 2000; dt = 0.05
#tr = trajectory(sss, N*dt; dt = dt, Ttr = 10.0)

#R = RecurrenceMatrix(tr, 5.0; metric = "euclidean")
#recurrenceplot(R; ascii = true)
#using CausalityTools

# Infer a completed partially directed acyclic graph (CPDAG)
#alg = PC(CorrTest(), CorrTest(); α = 0.05)
#est_cpdag_parametric = infer_graph(alg, X; verbose = false)

# Plot the graph
#plotgraph(est_cpdag_parametric)
ds = DiscreteDynamicalSystem(sss)#, [0.1, 0.2], [3.0])

xg = yg = range(0, maximum(sss); length = length(sss))
# Use non-sparse for using `basins_of_attraction`
mapper = AttractorsViaRecurrences(ds, (xg, yg);
    sparse = false, mx_chk_lost = 1000
)

plot_umap(mat_of_distances;file_name="UMAP_song_bird.png")

(scatter_indexs,yes,sort_idx) = label_online(mat_of_distances)
##
# 75 chanels and 25 seconds
##
function get_division_scatter(times,nodes,scatter_indexs,division_size,yes,sort_idx)
    #yes = yes[sort_idx]
    step_size = maximum(times)/division_size
    end_window = collect(step_size:step_size:step_size*division_size)
    spike_distance_size = length(end_window)
    #start_windows = collect(0:step_size:step_size*division_size-1)
    start_windows = collect(0:step_size:(step_size*division_size)-step_size)

    mat_of_distances = zeros(spike_distance_size,maximum(unique(nodes))+1)
    segment_length = end_window[3] - start_windows[3]
    p=Plots.plot(legend = :none)
    #p=Plots.scatter(times,nodes,legend=true)
    end_window = end_window[sort_idx]
    start_windows = start_windows[sort_idx]

    @showprogress for (ind,toi) in enumerate(end_window)
        sw = start_windows[ind]
        spikes = divide_epoch(nodes,times,sw,toi)    
        Nx=Vector{Int32}([])
        Tx=Vector{Float32}([])
        for (i, t) in enumerate(spikes)
            for tt in t
                if length(t)!=0
                    push!(Nx,i)
                    ###
                    ##
                    ##
                    #push!(Tx,Float32(tt))
                    ##
                    ###
                    ##
                    push!(Tx,Float32(sw+tt))
                    ##
                end
            end
        end
        #if yes[ind]>0.0
        p=Plots.scatter!(p,Tx,Nx,marker_z=yes[ind],legend = :none, markersize = 0.65)
            
        #end
    end
    #display(p)
    savefig("repeated_pattern_song_bird.png")
end
get_division_scatter(ttt,nnn,scatter_indexs,resolution,yes,sort_idx)
Plots.scatter!(ttt,nnn, markersize = 0.65)
savefig("normal.png")
#nnn,ttt = load_datasets()
#label_online(mat_of_distances)
