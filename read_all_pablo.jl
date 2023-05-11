using JLD
#using SpikeTime
#using
#using DrWatson
using Plots
using SpikingNeuralNetworks
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
using JLD
using CairoMakie#,KernelDensity, Distributions
using ProgressMeter
using Distances
#using Gadfly
#using Gadfly
using SGtSNEpi, Random

using Plots
#using PyCall
#odes = df.i
#division_size = maximum(times)/10.0
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
#display(Plots.scatter(times[1:Int32(round((length(times)/2000)))],nodes[1:Int32(round((length(times)/2000)))],markersize=0.05))
#savefig("slice_one_window.png")

#display(Plots.scatter(times[Int32(round((length(times)/2000))):2*Int32(round((length(times)/2000)))],nodes[Int32(round((length(times)/2000))):2*Int32(round((length(times)/2000)))],markersize=0.05))
#savefig("slice_two_window.png")

#display(Plots.scatter(times[2*Int32(round((length(times)/2000))):3*Int32(round((length(times)/2000)))],nodes[2*Int32(round((length(times)/2000))):3*Int32(round((length(times)/2000)))],markersize=0.05))
#savefig("slice_three_window.png")

function make_spike_movie(x,y,times,l)
    mymatrix = zeros((359,359))
    cnt=1
    l_old=1
    x_l = []
    y_l = []

    @showprogress for (t_,x_,y_) in zip(times,x,y,l)
        if l==l_old
            push!(x_l,x)
            push!(y_l,y)

        end
        if l!=l_old
            mymatrix[x_l,y_l] .= 100.0

            x_l = []
            y_l = []
        end
        cnt+=1
        l_old=l

    end
end


function get_changes(times,labels)
    time_break = []
    l_old=0
    @showprogress for (t,l) in zip(times,labels)
        if l!=l_old
            push!(time_break,Float32(t))
        end
        l_old = l
    end
    time_break
end

function get_plot(times,nodes,division_size)
    step_size = maximum(times)/division_size
    end_window = collect(step_size:step_size:step_size*division_size)
    spike_distance_size = length(end_window)
    start_windows = collect(0:step_size:step_size*division_size-1)
    mat_of_distances = zeros(spike_distance_size,maximum(unique(nodes))+1)
    n0ref = divide_epoch(nodes,times,last(start_windows),last(end_window))
    segment_length = end_window[3] - start_windows[3]
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
    savefig("Unormalised_heatmap_pablo.png")
    @inbounds @showprogress for (ind,col) in enumerate(eachcol(mat_of_distances))
        mat_of_distances[:,ind] .= (col.-mean(col))./std(col)
    end
    mat_of_distances[isnan.(mat_of_distances)] .= 0.0
    Plots.heatmap(mat_of_distances)
    savefig("Normalised_heatmap_pablo.png")
    p=nothing
    p = Plots.plot()
    Plots.plot!(p,mat_of_distances[1,:],label="1")#, fmt = :svg)
    Plots.plot!(p,mat_of_distances[9,:],label="2")#, fmt = :svg)
    savefig("just_two_pablo_raw_vectors.png")
    @save "pablo_matrix.jld" mat_of_distances
    return mat_of_distances
end
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
    @save "distances_angles_Pablo.jld" angles0 distances0#,angles1,distances1
    return angles0,distances0#,angles1,distances1)
end
using Clustering
using UMAP
function pablo_plots(mat_of_distances)
    R = kmeans(mat_of_distances, 5; maxiter=100, display=:iter)
    @assert nclusters(R) == 5 # verify the number of clusters
    a = assignments(R) # get the assignments of points to clusters
    c = counts(R) # get the cluster sizes
    M = R.centers # get the cluster centers
    #@show(sizeof(M))
    #savefig("didit_workpablo.png")
    return M
end

function final_plots2(distances0,angles0,ll)
    p=nothing
    p = Plots.plot()
    #angles0,distances0 = post_proc_viz(mat_of_distances)
    pp = Gadfly.Plots.plot(distances0,angles0,Geom.point)#marker =:circle, arrow=(:closed, 3.0)))
    img = SVG("iris_plot.svg", 6inch, 4inch)
    draw(img, p)
    #savefig("relative_to_uniform_reference_Pablo.png")   
end
(nodes,times) = load_datasets()
using UMAP
using Plots
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
#@save "distances_angles_Pablo.jld" angles0 distances0
@load "pablo_matrix.jld" mat_of_distances
mat_of_distances = mat_of_distances[:,1:15000]
plot_umap(mat_of_distances;file_name="pablo_umap.png")
#@load "distances_angles_Pablo.jld" angles0 distances0
#mat_of_distances = get_plot(times,nodes,1000)
#pablo_plots(mat_of_distances)
#final_plots2(mat_of_distances,ll)#,label_inventory_size)

#mat_of_distances .= mat_of_distances ./ norm.(eachcol(mat_of_distances))'
#mat_of_distances .= mat_of_distances ./ norm.(eachcol(mat_of_distances))

#mat_of_distances ./ norm.(eachrow(mat_of_distances))

#@showprogress for (ind,_) in enumerate(eachcol(mat_of_distances))
#    mat_of_distances[:,ind] = (mat_of_distances[:,ind].- mean(mat_of_distances))./std(mat_of_distances)

#end
#@showprogress for (ind,row) in enumerate(eachcol(mat_of_distances'))
#    mat_of_distances[ind,:] = (row.- mean(mat_of_distances))./std(mat_of_distances)
#end

#f = Figure()
#Axis(f[1, 1], title = "State visualization",)#yticks = ((1:length(mat_of_distances)) ,String([Char(i) for i in collect(1:length(mat_of_distances))])))

#=
@showprogress for (ind,_) in enumerate(eachrow(mat_of_distances))
    #d = kde(mat_of_distances[ind,:])
    #@show(ind)
    #if ind==1
    if ind>1
        p = Plots.plot!(p,mat_of_distances[ind,:].+prev,label="$ind")#, fmt = :svg)
    else
        p = Plots.plot!(p,mat_of_distances[ind,:],label="$ind")#, fmt = :svg)
    end
    prev += maximum(mat_of_distances[ind,:])
    
    #d = density!(randn(200) .- 2sin((i+3)/6*pi), offset = i / 4,
    #xs = collect(1:length(mat_of_distances[ind,:]))
    #d = Makie.lines(xs,mat_of_distances[ind,:],offset=ind*2, colormap = :thermal, colorrange = (-10, 10),strokewidth = 1, strokecolor = :black,bw=100)#, bandwidth = 0.02)
    # this helps with layering in GLMakie
    #translate!(d, 0, 0, -0.1i)
end
#Plots.plot!(legend=:outerbottom, legendcolumns=length(mat_of_distances))
savefig("pablo_raw_vectors.png")
=#   

#title!("Trigonometric functions")
#xlabel!("x")
#ylabel!("y")
#display(p)
#save("ridgeline_numberss_nmn.png",f)

#division_size=maximum(times)/10.0

#label_inventory_size = length(unique(labels))
#label_inventory_size = 200

