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
#using Ridgeline
using Makie
#using KernelDensity,Distributions
using JLD
using CairoMakie#,KernelDensity, Distributions
using ProgressMeter
using Distances

using Plots

function filter_on_off(x,y,times,p,l,nodes)
    xo=[];yo=[];timeso=[];po=[];lo=[];nodeso=[]
    x1=[];y1=[];times1=[];p1=[];l1=[];nodes1=[]

    for (x_,y_,times_,p_,l_,nodes_) in zip(x,y,times,p,l,nodes)
        if p_==1
            push!(xo,x_)
            push!(yo,y_)
            push!(timeso,times_)
            push!(lo,l_)
            push!(nodeso,nodes_)

        else
            push!(x1,x_)
            push!(y1,y_)
            push!(times1,times_)
            push!(l1,l_)
            push!(nodes1,nodes_)


        end
    end
    (xo,yo,timeso,po,lo,nodeso,x1,y1,times1,p1,l1,nodes1)

end

function load_datasets()
    @load "all_mnmist.jld" storage

    (x,y,times,p,l,nodes) = (storage[1][1],storage[1][2],storage[1][3],storage[1][4],storage[1][5],storage[1][6])

    (x,y,times,p,l,nodes,x1,y1,times1,p1,l1,nodes1) = filter_on_off(x,y,times,p,l,nodes)

    #(xo,yo,timeso,po,lo,nodeso,x1,y1,times1,p1,l1,nodes1) = filter_on_off(x,y,times,p,l,nodes)
    #@show(p)
    labels = [Int32(ll) for ll in l]
    perm = sortperm(labels)
    times = [Float32(t) for t in times]
    nodes = [Int32(t) for t in nodes]
    p = [Int32(p) for t in p]

    x = [UInt32(x_+1) for x_ in x ]
    y = [UInt32(y_+1) for y_ in y ]
    return (x,y,times,p,l,nodes,perm)
end
(x,y,times,p,l,nodes,perm) = load_datasets()
function make_spike_movie(x,y,times,l)
    mymatrix = zeros((359,359))
    cnt=1
    l_old=1
    x_l = []
    y_l = []

    @showprogress for (t_,x_,y_) in zip(times,x,y,l)
        #mymatrix = zeros((359,359))
        #@show(maximum(x))
        #@show(maximum(y))

        #@show(x_,y_,t_)
        #A = zeros((358,358))
        #if cnt%10 == 0
        if l==l_old
            push!(x_l,x)
            push!(y_l,y)

        end
        if l!=l_old
            mymatrix[x_l,y_l] .= 100.0

            x_l = []
            y_l = []

            #mymatrix .= 0
        end

        #    mymatrix = zeros((359,359))

        #end
        cnt+=1
        l_old=l
        #mymatrix[!=0] .= mymatrix[!=0] .-exp(mymatrix[!=0])
        #display(Plots.heatmap(mymatrix)) #which will also give you a color legend

    end
end
make_spike_movie(x,y,times,l)
#a[perm]
#SpikeTime
#=
nodes = nodes[perm]
times = times[perm]
labels = labels[perm]
x = x[perm]
y = y[perm]
=#
#using UnicodePlots

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
time_break = get_changes(times,labels)
#@show(time_break)

#display(Plots.scatter(times,nodes))
display(Plots.scatter(times,nodes,markersize=0.01))

function get_plot(times,nodes,division_size)
    step_size = maximum(times)/division_size
    end_window = collect(step_size:step_size:step_size*division_size)
    spike_distance_size = length(end_window)
    start_windows = collect(0:step_size:step_size*division_size-1)
    mat_of_distances = zeros(spike_distance_size,maximum(unique(nodes))+1)
    n0ref = divide_epoch(nodes,times,start_windows[3],end_window[3])
    segment_length = end_window[3] - start_windows[3]
    t0ref = surrogate_to_uniform(n0ref,segment_length)
    PP = []
    @showprogress for (ind,toi) in enumerate(end_window)
        self_distances = Array{Float32}(zeros(maximum(nodes)+1))
        sw = start_windows[ind]
        neuron0 = divide_epoch(nodes,times,sw,toi)    
        self_distances = get_vector_coords(neuron0,t0ref,self_distances)
        mat_of_distances[ind,:] = self_distances
    end
    cs1 = ColorScheme(distinguishable_colors(spike_distance_size, transform=protanopic))
    p=nothing
    mat_of_distances .= mat_of_distances ./ norm.(eachcol(mat_of_distances))'
    mat_of_distances .= mat_of_distances ./ norm.(eachcol(mat_of_distances))
    
    #mat_of_distances ./ norm.(eachrow(mat_of_distances))

    #@showprogress for (ind,_) in enumerate(eachcol(mat_of_distances))
    #    mat_of_distances[:,ind] = mat_of_distances[:,ind].- mean(mat_of_distances)./std(mat_of_distances)
    #end


    f = Figure()
    Axis(f[1, 1], title = "State visualization",)#yticks = ((1:length(mat_of_distances)) ,String([Char(i) for i in collect(1:length(mat_of_distances))])))
    p = Plots.plot()
    @showprogress for (ind,_) in enumerate(eachrow(mat_of_distances))
        #d = kde(mat_of_distances[ind,:])
        if ind==1
            p = Plots.plot!(p,mat_of_distances[ind,:],label="$ind")
        else
            p = Plots.plot!(p,mat_of_distances[ind,:].+prev*maximum(mat_of_distances[ind-1,:]),label="$ind")

        end
        prev = maximum(mat_of_distances[ind-1,:])
        #d = density!(randn(200) .- 2sin((i+3)/6*pi), offset = i / 4,
        #xs = collect(1:length(mat_of_distances[ind,:]))
        #d = Makie.lines(xs,mat_of_distances[ind,:],offset=ind*2, colormap = :thermal, colorrange = (-10, 10),strokewidth = 1, strokecolor = :black,bw=100)#, bandwidth = 0.02)
        # this helps with layering in GLMakie
        #translate!(d, 0, 0, -0.1i)
    end
    Plots.plot!(legend=:outerbottom, legendcolumns=length(mat_of_distances))
    savefig("blah.png")
    #title!("Trigonometric functions")
    #xlabel!("x")
    #ylabel!("y")
    #display(p)
    #save("ridgeline_numberss_nmn.png",f)

    return mat_of_distances,f
end
#division_size=maximum(times)/10.0

label_inventory_size = length(unique(labels))
mat_of_distances,f = get_plot(times,nodes,label_inventory_size)
function post_proc_viz(mat_of_distances)
    # ] add https://github.com/JeffreySarnoff/AngleBetweenVectors.jl
    # ] add Distances
    angles0 = []
    distances0 = []
    angles1 = []
    distances1 = []
    p=nothing
    for (ind,self_distances) in enumerate(eachrow(mat_of_distances))
        if ind>1
            θ = angle(mat_of_distances[ind,:],mat_of_distances[ind-1,:])
            r = evaluate(Euclidean(),mat_of_distances[ind,:],mat_of_distances[ind-1,:])
            append!(angles1,θ)
            append!(distances1,r)        
        end
        θ = angle(mat_of_distances[ind,:],mat_of_distances[1,:])
        r = evaluate(Euclidean(),mat_of_distances[ind,:],mat_of_distances[1,:])
        append!(angles0,θ)
        append!(distances0,r)        
    end
    return angles0,distances0,angles1,distances1
end

function final_plots2(mat_of_distances,label_inventory_size)
    p=nothing
    p = Plots.plot()
    angles0,distances0,angles1,distances1 = post_proc_viz(mat_of_distances)
    display(Plots.scatter!(p,angles1,distances1,marker =:circle, arrow=(:closed, 3.0)))
    savefig("relative_to_uniform_reference.png")   
    p = Plots.plot()


    features = collect(Matrix([angles1,distances1])'); # features to use for clustering
    result = kmeans(features, label_inventory_size); # run K-means for the 3 clusters

    # plot with the point color mapped to the assigned cluster index
    scat= scatter(angles1, istances1, marker_z=result.assignments, color=:lightrainbow, legend=false)
    display(scat)
    savefig("Clustered_relative_to_uniform_reference.png")   

    display(Plots.scatter!(p,angles0,distances0,marker =:circle, arrow=(:closed, 3.0)))
    savefig("relative_to_each_other.png")   

    (angles1,distances1)
end

final_plots2(mat_of_distances,label_inventory_size)
