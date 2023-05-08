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
#using ProgressMeter
using ColorSchemes
using PyCall
using LinearAlgebra
#using Ridgeline
#using Makie
#using KernelDensity,Distributions
using JLD
#using CairoMakie#,KernelDensity, Distributions
using ProgressMeter
using Distances

using Plots
using StaticArrays

function filter_on_off(x,y,times,p,l,nodes)
    xo=[];yo=[];timeso=[];po=[];lo=[];nodeso=[]
    x1=[];y1=[];times1=[];p1=[];l1=[];nodes1=[]

    for (x_,y_,times_,p_,l_,nodes_) in zip(x,y,times,p,l,nodes)
        if p_==1
            #if x_<UInt32(50) && y_<UInt32(50)
            push!(xo,x_)
            push!(yo,y_)
            push!(timeso,times_)
            push!(lo,l_)
            push!(nodeso,nodes_)
            #end
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
    xx=[]
    yy=[]
    tt = []
    pp = []
    ll = []
    nn = []
    window_size = 0
    @showprogress for (ind,s) in enumerate(storage)

        (x,y,times,p,l,nodes) = (s[1],s[2],s[3],s[4],s[5],s[6])
        if  maximum(times) > window_size
            window_size = maximum(times)
        end
        append!(xx,x)
        append!(yy,y)
        append!(tt,times)
        append!(pp,p)
        append!(ll,l)
        append!(nn,nodes)
    end

    (x,y,times,p,l,nodes,x1,y1,times1,p1,l1,nodes1) = filter_on_off(xx,yy,tt,pp,ll,nn)

    #(xo,yo,timeso,po,lo,nodeso,x1,y1,times1,p1,l1,nodes1) = filter_on_off(x,y,times,p,l,nodes)
    #@show(p)
    labels = [Int32(ll) for ll in l]
    #perm = sortperm(labels)
    times = [Float32(t) for t in times]
    nodes = [Int32(t) for t in nodes]
    p = [Int32(p) for t in p]

    x = [UInt32(x_+1) for x_ in x ]
    y = [UInt32(y_+1) for y_ in y ]
    return (x,y,times,p,l,nodes,window_size)
end
#   (x,y,times,p,labels,nodes) = load_datasets()
#labels = l
#using UnicodePlots
#using SparseArrays
function make_spike_movie(x,y,times,labels)
    cnt=0
    l_old=1
    x_l = []
    y_l = []
    mymatrix = zeros((36,36))
    @showprogress for (t_,x_,y_,l) in zip(times,x,y,labels)    
        if l==l_old
            push!(x_l,x)
            push!(y_l,y)

        end
        if l!=l_old && length(x_l)>1 && length(y_l)>1
            @show(length(x_l))
            @show(length(y_l))
            @show(l)
            @show(l_old)
            @show(mymatrix[x_l,y_l]) #.= 10.0
            #display(UnicodePlots.spy(mymatrix))
            Plots.heatmap(mymatrix)
            Plots.savefig("NMNIST_matrix$l.png")
    
            mymatrix = zeros((359,359))
            x_l = []
            y_l = []
        end
        cnt+=1
        l_old=l
    end
end
#make_spike_movie(x,y,times,labels)

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
    t_old=0
    @showprogress for (t,l) in zip(times,labels)
        if l==l_old
            @show(abs(t-t_old))
        end
        if l!=l_old
            push!(time_break,Float32(t))
        end
        l_old = l
        t_old = t
    end
    time_break
end
#time_break = get_changes(times,labels)
#@show(time_break)

#display(Plots.scatter(times,nodes,markersize=0.01))
#p=Plots.scatter(times,nodes,markersize=0.01)
#Plots.vline!(p,time_break,markersize=0.01)
#savefig("nmist_raster_plot.png")
#nodes = convert(Vector{Int32},nodes)

#dt = 0.1
#tau = 0.4
#plot_umap(nodes,times,dt,tau;file_name="UMAP_OF_NMNIST.png")


function get_plot(storage)
    times,nodes = storage[1][3],storage[1][6]
    spike_distance_size = length(storage)
    n0ref = expected_spike_format(nodes,times)
    non_zero_spikings=[]
    for tx in n0ref
        if length(tx)!=0
            push!(non_zero_spikings,length(tx)) 
        end
    end
    mean_spk_counts = Int32(round(mean(non_zero_spikings)))
    temp = LinRange(0.0, maximum(times), mean_spk_counts)
    linear_uniform_spikes = Vector{Float32}([i for i in temp])

    p = Plots.plot()
    Plots.scatter(linear_uniform_spikes,[i for i in collect(1:length(unique(nodes)))])
    savefig("UniformSpikes.png")
    p=nothing

    return (linear_uniform_spikes,mean_spk_counts,nodes,length(storage))
end

function expected_spike_format(nodes1,times1)
    nodes1 = [i+1 for i in nodes1]
    n0ref =  []
    @inbounds for i in collect(1:length(unique(nodes1)))
        push!(n0ref,[])
    end
    @inbounds for i in collect(1:length(unique(nodes1)))
        for (neuron, t) in zip(nodes1,times1)#nxxx_
            if i == neuron
                push!(n0ref[Int32(i)],Float32(t))
            end            
        end
        @show(length(n0ref[Int32(i)]))
    end
    n0ref
end

function get_plot2(storage)

    (linear_uniform_spikes,mean_spk_counts,nodes,spike_distance_size) = get_plot(storage)
    p = Plots.plot()
    mat_of_distances = Array{Float32}(zeros(spike_distance_size,length(nodes)+1))
    cs1 = ColorScheme(distinguishable_colors(spike_distance_size, transform=protanopic))

    @inbounds @showprogress for (ind,s) in enumerate(storage)
        (times,nodes,labels) = (s[3],s[5],s[6])
        ll=labels[1]
        l = convert(Int32,ll)
        #@show(l,ind)
        #Plots.scatter(times,nodes)
        #savefig("NMIST_LABEL_SCATTER_$l.$ind.png")

        #(x,y,times,p,l,nodes,x1,y1,times1,p1,l1,nodes1) = filter_on_off(xx,yy,tt,pp,ll,nn)
        self_distance_populate = Array{Float32}(zeros(length(nodes)+1))

        if length(times) > 1

            times = expected_spike_format(nodes,times)
            #@show(length(times[1]))
            #@show(length(times))
            # get_vector_coords_uniform!(uniform::Vector{Float32}, neuron1::Vector{Any}, self_distances::Vector{Float32})

            self_distance_populate = Array{Float32}(zeros(length(unique(nodes))+1))

            get_vector_coords_uniform!(linear_uniform_spikes,times,self_distance_populate)
            #@show(self_distance_populate)
            #mat_of_distances[ind,:] = self_distance_populate
            #@show(self_distance_populate)
            #@show(prev)

            #if ind>1
            #    p = Plots.plot!(p,self_distance_populate.+prev,label="$ind")
            #else
            #    p = Plots.plot!(p,self_distance_populate,label="$ind.$l")
            #end
            #prev += maximum(self_distance_populate)
            #Plots.plot!(legend=:outerbottom, legendcolumns=length(mat_of_distances))
            #savefig("NMNIST_codes_$l.$ind.png")
        end
    end
    for (ind,col) in enumerate(eachcol(mat_of_distances))
        mat_of_distances[ind,:] .= (col.-mean(col))./std(col)
    end
    p = Plots.plot()
    prev=0.0

    for (ind,row) in enumerate(eachrow(mat_of_distances))
        if ind>1
            p = Plots.plot!(p,row.+prev,label="$ind")
        else
            p = Plots.plot!(p,row,label="$ind.$l")
        end
        prev += maximum(row)
    end
    savefig("NMNIST_codes.png")

    return mat_of_distances
end
#division_size=maximum(times)/10.0
#label_inventory_size = length(unique(labels))
@load "all_mnmist.jld" storage

mat_of_distances = get_plot2(storage)

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
    for (ind,(a,d)) in enumerate(zip(angles,distances1))
        Plots.scatter!(p,angles1,distances1,label="$ind")#marker =:circle, arrow=(:closed, 3.0)))
    savefig("relative_to_uniform_referenceNMMIST.png")   
    p = Plots.plot()


    #features = collect(Matrix([angles1,distances1])'); # features to use for clustering
    #result = kmeans(features, label_inventory_size); # run K-means for the 3 clusters

    # plot with the point color mapped to the assigned cluster index
    #scat= scatter(angles1, istances1, marker_z=result.assignments, color=:lightrainbow, legend=false)
    #display(scat)
    #savefig("Clustered_relative_to_uniform_reference.png")   

    display(Plots.scatter!(p,angles0,distances0,marker =:circle, arrow=(:closed, 3.0)))
    savefig("relative_to_each_otherNMMIST.png")   

    (angles1,distances1)
end

final_plots2(mat_of_distances,label_inventory_size)
