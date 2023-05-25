using JLD
import JLD.save
#using SpikeTime
using DynamicalSystems
import Attractors#.DiscreteDynamicalSystem
import Attractors#.AttractorsViaRecurrence
using DelayEmbeddings
using DynamicalSystemsBase # to simulate Lorenz63

#using Attractors
using StaticArrays
using ChaosTools, CairoMakie
using CairoMakie
using Colors
using Attractors

using DelayEmbeddings: embed, estimate_delay
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

using DelayEmbeddings

# Randomly permute neuron labels.
# (This hides the sequences, to make things interesting.)
#_p = Random.randperm(num_neurons)

# Load spikes.
#spikes = seq.Spike[]
function load_datasets()

    spikes = []
    the_file_contents_as_iterable = readdlm("../data/songbird_spikes.txt", '\t', Float64, '\n')
    nodes = [n for (n, t) in eachrow(the_file_contents_as_iterable)]
    for _ in 1:maximum(unique(nodes))+1
        push!(spikes,[])
    end

    for (n, t) in eachrow(the_file_contents_as_iterable)
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
    #=
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
    for (n, t) in eachrow(readdlm("../data/songbird_spikes.txt", '\t', Float64, '\n'))
        push!(spikes[Int32(n)],t+maxt)
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
    for (n, t) in eachrow(readdlm("../data/songbird_spikes.txt", '\t', Float64, '\n'))
        push!(spikes[Int32(n)],t+maxt)
    end
    =#
    (nnn,ttt,spikes)
    #(spikes,nnn)
end
function get_plot(times,nodes,division_size)
    step_size = maximum(times)/division_size
    #@show(step_size)
    end_window = collect(step_size:step_size:step_size*division_size)
    spike_distance_size = length(end_window)
    start_windows = collect(0:step_size:(step_size*division_size)-step_size)

    mat_of_distances = zeros(spike_distance_size,maximum(unique(nodes))+1)
    
    #@show(last(start_windows),last(end_window))
    n0ref = divide_epoch(nodes,times,last(start_windows),last(end_window))
    
    segment_length = end_window[2] - start_windows[2]
    #@show(segment_length)
    mean_spk_counts = Int32(round(mean([length(times) for times in enumerate(n0ref)])))
    t0ref = surrogate_to_uniform(n0ref,segment_length,mean_spk_counts)
    PP = []
    #mean_spk_counts = Int32(round(mean(non_zero_spikings)))
    temp = LinRange(0.0, maximum(times), mean_spk_counts)
    linear_uniform_spikes = Vector{Float32}([i for i in temp])
    spike_chops = []
    spike_chops_ind = []

    @showprogress for (ind,toi) in enumerate(end_window)
        sw = start_windows[ind]
        neuron0 = divide_epoch(nodes,times,sw,toi)    
        
        #@show(neuron0)
        self_distances = Array{Float32}(zeros(maximum(nodes)+1))

        get_vector_coords_uniform!(linear_uniform_spikes, neuron0, self_distances)
        #for t in neuron0
            
        #end
        for (ind,ttx) in enumerate(neuron0)

            #if length(ttx)!=0
            neuronch=[]
            push!(spike_chops_ind,ind)
            for t in ttx
                

                push!(neuronch,t+sw)
            end
            
            #end
            push!(spike_chops,neuronch)

        end
        #self_distances = get_vector_coords(neuron0,t0ref,self_distances)
        mat_of_distances[ind,:] = self_distances
    end
    #cs1 = ColorScheme(distinguishable_colors(spike_distance_size, transform=protanopic))
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
    return mat_of_distances,spike_chops,spike_chops_ind
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


#plot_umap(mat_of_distances'; file_name="UMAP_transpose.png")
#plot_umap(mat_of_distances; file_name="UMAP_not_transpose.png")

#@load "ll.jld" ll
#@save "distances_angles_song_bird.jld" angles0 distances0
#@load "song_bird_matrix.jld" mat_of_distances
#mat_of_distances = mat_of_distances[:,1:10000]

using SparseArrays
using Plots, StatsPlots
#import Base.find
function label_online(mat_of_distances,spikes,classes)
    R = kmeans(mat_of_distances, classes; maxiter=2000, display=:iter)
    #a = assignments(R) # get the assignments of points to clusters
    c = counts(R) # get the cluster sizes
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
    #bi = []
    #mat_of_distances = copy(mat_of_distances'[:])
    @showprogress for (ind,row) in enumerate(eachrow(mat_of_distances))
        flag = false
        for ix in collect(1:classes)
            best_distance = 100000.0
            current_centre = M[:,ix]
            distance = evaluate(Euclidean(),row,current_centre)
            #istance = sum(abs.(row .- current_centre))
            if distance<best_distance
                best_distance = distance
                best_index = ind
                best_index_class = ix
            end
            #if best_distance < 6.5
            if best_distance < 5.5

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
    p1 = Plots.heatmap(mat_of_distances',legend = :none,c = cgrad([:white,:red,:blue]), ylabel = "Vector due to Cell id", xlabel = "Time slice")
    #savefig("reference_labelled_mat_of_distances_song_bird.png")
    p2 = Plots.heatmap(labelled_mat_of_distances',legend = :none, c = cgrad([:white,:red,:blue]), ylabel = "Vector due to Cell id", xlabel = "Time slice")
    
    #Plots.heatmap(ref_mat_of_distances,legend = :none, c = cgrad([:white,:red,:blue]), ylabel = "Vector due to Cell id", xlabel = "Time slice")
    savefig("un_sorted_heat_bird.png")
    p4 = Plots.heatmap(ref_mat_of_distances',legend = :none, c = cgrad([:white,:red,:blue]), ylabel = "Vector due to Cell id", xlabel = "Time slice")
    #savefig("sorted_heat_bird.png")
    spar=sparse(ref_mat_of_distances)
    #spar = dropzeros(spar)
    #@show(spar)
    #spars = ref_mat_of_distances
    I,J,Z = findnz(spar)
    dense = ref_mat_of_distances'#[I,J]
    display(sparse(dense))
    StatsPlots.plot(
        StatsPlots.heatmap(dense, colorbar=true))
    savefig("unsort_spike_train_sequence_hierarchy.png")
    #matd=filter!(x->x!=0.0,ref_mat_of_distances)
    #theme(:ggplot2)
    #gr(size = (700, 700))
    #R = kmeans(dense, classes; maxiter=2000, display=:iter)#, linkage=:complete)
    R = kmeans(dense, classes; maxiter=2000, display=:iter)#, linkage=:complete)
    sort_idx =  sortperm(assignments(R))
    #hcl = hclust(dismat, linkage=:ward) # ward linkage
    StatsPlots.plot(
        StatsPlots.heatmap(labelled_mat_of_distances'[:,sort_idx], colorbar=true))
    savefig("spike_train_sequence_hierarchy.png")


    StatsPlots.plot(
        StatsPlots.heatmap(ref_mat_of_distances'[:,sort_idx], colorbar=true))
    savefig("ref_pike_train_sequence_hierarchy.png")
    #p5 = Plots.heatmap(spar,legend = :none, c = cgrad([:white,:red,:blue]), ylabel = "Vector due to Cell id", xlabel = "Time slice")

    #=
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
    =#

    #p5 = Plots.scatter()
    #for spc in scatter_indexs
        #@show(spc)

    #end
    #display(p5)
    #for (i,s) in enumerate(spikes)
    #    Plots.scatter!(p5,s,[i],legend=false, alpha =0.65, markersize = 1.0,marker_z=2)
    #end

    Plots.plot(p1,p2,p4,legend = :none)
    savefig("p1_p4_both_labelled_mat_of_distances_song_bird.png")
    #@show(yes)
    #return M
    (scatter_indexs,yes,sort_idx)
end
resolution=100
classes = 10

(nnn,ttt,spikes)= load_datasets()
mat_of_distances,spike_chops,spike_chops_ind = get_plot(ttt,nnn,resolution)

#display(Plots.scatter(ttt,nnn))
(scatter_indexs,yes,sort_idx) = label_online(mat_of_distances,spikes,classes)#,spike_chops,spike_chops_ind)
#@show(scatter_indexs)
#@show(spike_chops[scatter_indexs])
function compile_this_plot(spike_chops_ind,spike_chops,scatter_indexs,ttt,nnn)
    #for zi in scatter_indexs
    #    @show(spike_chops[zi])
    #    @show(spike_chops_ind[zi])
    #end
    #p3 = Plots.scatter()
    #@showprogress for (i,j) in zip(spike_chops_ind[1:120],spike_chops[1:120])
    #    if length(i)!=0
    #       Plots.scatter!(p3,i,j,legend=false, alpha =1.0, markersize = 2.0,marker_z=1,markerstrokewidth=0)
    #    end
    #end

    #display(p3)
    p1 = Plots.scatter()
    #@show(scatter_indexs)
    #for (cnt,zi) in enumerate(scatter_indexs)
       #if length(spike_chops[zi])!=0
     #   @show(spike_chops[zi])
      #  @show(spike_chops_ind[zi])
       # @show(zi)

            #for z in zi
    Plots.scatter!(p1,spike_chops[scatter_indexs],scatter_indexs,legend=false, markersize = 3.0)#marker_z=length(spike_chops[scatter_indexs]))
            #end
    #   end
    #end
    #p2 = Plots.scatter(spike_chops,spike_chops_ind,legend=false, alpha =0.65, markersize = 1.0)

    #for (time,node) in zip(ttt,nnn)
   # for zi in scatter_indexs
        #if zi in node
        #else
        #    Plots.scatter!(p3,[time],[node],legend=false, alpha =0.5, markersize = 1.5,marker_z=1,markerstrokewidth=0)
        #end
    #end
    #end
    #@show(spike_chops_ind[scatter_indexs])
    Plots.plot(p1)
    savefig("blan.png")
end

compile_this_plot(spike_chops_ind,spike_chops,scatter_indexs,ttt,nnn)
function sort_songs(ttt,nnn,resolution)
    mat_of_distances = get_plot(ttt,nnn,resolution)

    classes = 10
    R = kmeans(mat_of_distances, classes; maxiter=20000, display=:iter)
    #a = assignments(R) # get the assignments of points to clusters
    c = counts(R) # get the cluster sizes
    M = R.centers # get the cluster centers
    #M = copy(M'[:])

    #c = kmeans(M,4)
    sort_idx =  sortperm(assignments(R))
    return (sort_idx,c,mat_of_distances)
end

function plot_umap(mat_of_distances,cluster_sizes,sort_idx; file_name::String="empty.png")
    #model = UMAP_(mat_of_distances', 10)
    #Q_embedding = transform(model, amatrix')
    #cs1 = ColorScheme(distinguishable_colors(length(ll), transform=protanopic))

    Q_embedding = umap(mat_of_distances,20,n_neighbors=20)#, min_dist=0.01, n_epochs=100)
    p=Plots.scatter()#p,Q_embedding[1,:], Q_embedding[2,:], title="Spike Time Distance UMAP, reduced precision", marker=(1, 1, :auto, stroke(0.05)),legend=true))
    c_old = 1
    run_tot=0
    cumulative = 1
    for (color,c) in enumerate(cluster_sizes)
        cumulative += c

        #for altitude in collect(c_old:cumulative)
        Plots.scatter!(p,[Q_embedding[1,c_old:cumulative]], [Q_embedding[2,c_old:cumulative]],marker_z=color+1, title="Spike Time Distance UMAP, reduced precision", markersize=3,legend=false,markerstrokewidth=0)
        #end
        c_old = cumulative
    end
    Plots.scatter!(p,[Q_embedding[1,:]], [Q_embedding[2,:]],marker_z=1, alpha = 0.25, title="Spike Time Distance UMAP, reduced precision", markersize=3,legend=false,markerstrokewidth=0)

    savefig(file_name)
    Q_embedding
end
function dostuff(resolution,spikes)
    (sort_idx,cluster_sizes,mat_of_distances) = sort_songs(ttt,nnn,resolution)    
    #plot_umap(mat_of_distances,cluster_sizes,sort_idx; file_name="UMAP_not_transpose.png")

    #=
    c_old = 1
    for (color,c) in enumerate(cluster_sizes)
        Plots.scatter(spikes[sort_idx[c_old:c]],sort_idx[c_old:c],marker_z=color,legend=false)
        c_old = c
        savefig("plots_sort$color.png")

    end
    =#
    #@show(length(spikes))
    #@show(length(sort_idx))

    p2 = Plots.scatter()

    for (i,s) in enumerate(spikes)
        Plots.scatter!(p2,s,[i],legend=false, alpha =0.65, markersize = 1.0,marker_z=2)
    end

    p = Plots.scatter()

    #for (i,s) in enumerate(spikes)
     #   Plots.scatter!(p,s,[i],marker_z=1,legend=false, alpha = 0.25, markersize = 1.2)
    #end
    c_old = 1
    run_tot=0
    cumulative = 1
    cols = distinguishable_colors(10)
    for (color,c) in enumerate(cluster_sizes)

        #Plots.hspan!(p,[c_old,cumulative], alpha = 0.05,cgrad=cols)
        Plots.hline!(p,[c_old])

        for altitude in collect(c_old:cumulative)
            Plots.scatter!(p,spikes[sort_idx[altitude]],[altitude],marker_z=color+1,legend=false, alpha = 0.75, markersize = 1.0,cgrad=cols,markerstrokewidth=0)
        end
        run_tot += length(sort_idx[c_old:cumulative])
        #for (x,i) in enumerate(sort_idx[c_old:c])

            #Plots.scatter!(p,spikes[i],[i],marker_z=color+1,legend=false)
        #end
        c_old = cumulative
        cumulative += c

    end

    p3 = Plots.scatter()
    c_old = 1
    run_tot=0
    cumulative = 1
    cols = distinguishable_colors(10)
    for (color,c) in enumerate(cluster_sizes)
        #Plots.hline(p,[c_old], alpha = 1,cgrad=cols)

        #Plots.hspan!(p3,[c_old,cumulative], alpha = 0.10,cgrad=cols)
        for altitude in collect(c_old:cumulative)
            Plots.scatter!(p3,spikes[sort_idx[altitude]],[sort_idx[altitude]],marker_z=color+1,legend=false, alpha = 0.75, markersize = 1.0,cgrad=cols,markerstrokewidth=0)
        end
        #Plots.scatter!(p3,spikes[sort_idx[c_old:cumulative]],sort_idx[c_old:cumulative],marker_z=color+1,legend=false, alpha = 0.95, markersize = 1.95,cgrad=cols)
        run_tot += length(sort_idx[c_old:cumulative])
        
        c_old = cumulative
        cumulative += c

    end
    @show(length(spikes))
    @show(run_tot)

    p,p2,p3,mat_of_distances,sort_idx
end
p,p2,p3,mat_of_distances,sort_idx = dostuff(resolution,spikes)
Plots.plot(p3)
savefig("sorted_raster_plotsp3.png")
Plots.plot(p2)
savefig("sorted_raster_plotsp2.png")
Plots.plot(p)
savefig("sorted_raster_plotspx.png")


p4 = Plots.plot(p,p2,p3)

savefig("sorted_raster_plots.png")

px1 = Plots.heatmap(mat_of_distances')
px2 = Plots.heatmap(mat_of_distances'[sort_idx,:])
px3 = Plots.plot(px1,px2)#,p3)
savefig("sorted_heat_plots.png")
p4 = Plots.plot(p,p3)

display(p4)

#display(px3)

#mat_of_distances = copy(mat_of_distances)
#Y_mt, τ_vals_mt, ts_vals_mt, Ls_mt, εs_mt = pecuzal_embedding(mat_of_distances;
#    τs = 0:Tmax, L_threshold = 0.2, w = theiler, econ = true
#mat_of_distances = mat_of_distances'
#display(Plots.heatmap(mat_of_distances))

sss = StateSpaceSet([row for (i,row) in enumerate(eachcol(mat_of_distances))])
Tmax = 20
@show(sss)
display(sss)
theiler = estimate_delay(vec(reduce(vcat,mat_of_distances)), "mi_min") # estimate a Theiler window

@show(theiler)

Y, τ_vals, ts_vals, Ls, εs = pecuzal_embedding(sss; τs = 0:Tmax , w = theiler, econ = true)

fig = Figure()
ax = Axis(fig[1,1])
lines!(εs[:,1], label="1st emb. cycle")
scatter!([τ_vals[2]], [εs[τ_vals[2],1]])
lines!(εs[:,2], label="2nd emb. cycle")
scatter!([τ_vals[3]], [εs[τ_vals[3],2]])
lines!(εs[:,3], label="3rd emb. cycle")
ax.title = "Continuity statistics PECUZAL Lorenz"
ax.xlabel = "delay τ"
ax.ylabel = "⟨ε⋆⟩"
axislegend(ax)
fig

U, S = broomhead_king(mat_of_distances, 20)
fig = Figure()
axs = [Axis3(fig[1, i]) for i in 1:2]
lines!(axs[1], U[:, 1], U[:, 2], U[:, 3])
axs[1].title = "Broomhead-King of s"

R = embed(mat_of_distances, 3, estimate_delay(x, "mi_min"))
lines!(axs[2], columns(R)...)
axs[2].title = "2D embedding of s"
display(fig)

#StateSpaceSet(row for (i,row) in enumerate(mat_of_distances))
#sss = StateSpaceSet(mat_of_distances)

#Y_mt, τ_vals_mt, ts_vals_mt, Ls_mt, εs_mt = pecuzal_embedding(sss;
#          τs = 0:Tmax, L_threshold = 0.2, econ = false
#      )
Y_mt, τ_vals_mt, ts_vals_mt, Ls_mt, εs_mt = pecuzal_embedding(sss;
        τs = 0:Tmax, w=theiler, L_threshold = 0.4, econ = true
)
@show(size(Y_mt))
println(τ_vals_mt)
println(ts_vals_mt)
fig = Figure(resolution = (1000,500) )
ax1 = Axis3(fig[1,1], title = "PECUZAL reconstructed")

ts_str = ["x", "y", "z"]

#fig = Figure(resolution = (1000,500) )
#ax1 = Axis3(fig[1,1], title = "PECUZAL reconstructed")
#display(Plots.scatter(Y_mt[:,1], Y_mt[:,2]))
lines!(ax1, Y_mt[:,1], Y_mt[:,2], Y_mt[:,3]; linewidth = 1.0)

#ax1.xlabel = "$(ts_str[ts_vals_mt[1]])(t+$(τ_vals_mt[1]))"
#ax1.ylabel = "$(ts_str[ts_vals_mt[2]])(t+$(τ_vals_mt[2]))"
#ax1.zlabel = "$(ts_str[ts_vals_mt[3]])(t+$(τ_vals_mt[3]))"
ax1.azimuth = 3π/2 + π/4

#ax2 = Axis3(fig[1,2], title = "original")
#lines!(ax2, mat_of_distances[:,1], mat_of_distances[:,2], mat_of_distances[:,3]; linewidth = 1.0, color = Cycled(2))
#ax2.xlabel = "x(t)"
#ax2.ylabel = "y(t)"
##ax2.zlabel = "z(t)"
#a#x2.azimuth = π/2 + π/4
display(fig)
basins_fractions(Y_mt)
#=
using DynamicalSystems
using Attractors
using Basins

function newton_map(u,mat_of_distances)
    temp = mat_of_distances[u,:]
    sized=size(temp)    
    return SVector{sized[1]}(temp)
end


# dummy Jacobian function to keep the initializator happy
function newton_map_J(J,z0, p, n)
   return
end

ds = DiscreteDynamicalSystem(newton_map,[0, 10], [3] , newton_map_J)
integ  = integrator(ds)

xg=range(0),10,length=200)
yg=range(0,10,length=200)

bsn=basin_discrete_map(xg, yg, integ)

basins, attractors = basins_of_attraction(mat_of_distances; show_progress = true)

function henon_rule(u,mat_of_distances)
    temp = mat_of_distances[u,:]
    sized=size(temp)    
    return SVector{sized[1]}(temp)
end
u0 = [0,length(mat_of_distances)]
henon = DeterministicIteratedMap(henon_rule, u0,mat_of_distances)


#embed = embed(sss)
#@show(sss)
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
=#
#=
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
=#