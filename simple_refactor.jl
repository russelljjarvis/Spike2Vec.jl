using JLD2
using DrWatson
using Revise
using StatsBase
using ColorSchemes
using PyCall
using LinearAlgebra
using ProgressMeter
using Distances
using Plots
using StaticArrays
using OnlineStats
using Heatmap
using Clustering
using SpikingNeuralNetworks

#import OnlineStats
#using Pkg
#Pkg.add("ScikitLearn")
using ScikitLearn


function get_plot_uniform(temp_container_store)
    times,nodes = temp_container_store[3],temp_container_store[6]
    #@show(nodes[1])
    spike_distance_size = length(temp_container_store)
    (refnodes,reftimes)=(nodes[1],times[1])
    n0ref = expected_spike_format(refnodes,reftimes)
    non_zero_spikings=[]
    @inbounds for tx in n0ref
        if length(tx)!=0
            push!(non_zero_spikings,length(tx)) 
        end
    end
    mean_spk_counts = Int32(round(maximum(non_zero_spikings)))
    temp = LinRange(0.0, maximum(reftimes), mean_spk_counts)
    linear_uniform_spikes = Vector{Float32}([i for i in temp])

    # p = Plots.plot()
    #@inbounds for xx in collect(1:1220)
    #    altitude = [xx for i in collect(1:1220)]
        #Plots.scatter!(p,linear_uniform_spikes,altitude, legend = false,markersize=0.1)
    #end
    #savefig("UniformSpikes.png")
    return (linear_uniform_spikes,mean_spk_counts,nodes,length(temp_container_store))
end

function expected_spike_format(nodes1,times1)
    nodes1 = [i+1 for i in nodes1]
    n0ref =  []
    @inbounds for i in collect(1:1220)
        push!(n0ref,[])
    end
    @inbounds for i in collect(1:1220)
        @inbounds for (neuron, t) in zip(nodes1,times1)
            if i == neuron
                push!(n0ref[Int32(i)],Float32(t))
            end            
        end
    end
    n0ref
end

function pre_process_spike_data(temp_container_store)
    (linear_uniform_spikes,mean_spk_counts,nodes,spike_distance_size) = get_plot_uniform(temp_container_store)
    list_lists = Vector{Any}([])
    labels = Vector{Any}([])
    prev = 0.0
    @inbounds @showprogress for (ind,s) in enumerate(temp_container_store)
        (times,labels,nodes) = (s[3],s[5],s[6])
        ll=labels[1]
        l = convert(Int32,ll)
        self_distance_populate = Array{Float32}(zeros(1220))
        if length(times) > 1
            times = expected_spike_format(nodes,times)
            get_vector_coords_uniform!(linear_uniform_spikes,times,self_distance_populate)
            push!(list_lists,self_distance_populate)
            push!(labels,l)

        end
    end

    @save "list_lists.jld" list_lists

    return list_lists,labels
end
function list_to_matrix(list_lists)
    min_ = Int(1000000)
    @inbounds @showprogress for (ind,l) in enumerate(list_lists)
        if Int(length(l)) < min_
            min_ = Int(length(l))   
        end
    end
    mat_of_distances = Array{Float32}(zeros(length(list_lists),1220))
    @inbounds @showprogress for (ind,l) in enumerate(list_lists)
        mat_of_distances[ind,:] = l#[1:min_]
    end
    mat_of_distances[isnan.(mat_of_distances)] .= 0.0
    display(Plots.heatmap(mat_of_distances))#, axis=(xticks=(1:5, xs), yticks=(1:10, ys), xticklabelrotation = pi/4) ))
    return mat_of_distances
end

function train_label_online(mat_of_distances,ll,nclasses)
    #classes = 10
    R = kmeans(mat_of_distances, nclasses; maxiter=12000)#, display=:iter)
    a = assignments(R) # get the assignments of points to clusters
    c = counts(R) # get the cluster sizes
    M = R.centers # get the cluster centers
    sort_idx =  sortperm(assignments(R))
    M_ = mat_of_distances[:,sort_idx]
    Plots.heatmap(M_)
    savefig("clustered_train_model.png")
    return M,sort_idx,M_
end

function test_label_online(M,test_mat_of_distances,ll,nclasses)
    labelled_mat_of_distances = copy(test_mat_of_distances)

    labelled_mat_of_distances = zeros(size(test_mat_of_distances))
    ref_mat_of_distances = zeros(size(test_mat_of_distances))

    scatter_indexs = []
    scatter_class_colors = []
    yes = []

    @show(size(M))
    @show(size(test_mat_of_distances))


    @showprogress for (ind,row) in enumerate(eachcol(test_mat_of_distances))
        flag = false
        for ix in collect(1:nclasses)
            best_distance = 100000.0
            current_centre = M[:,ix]
            @show(length(row))
            @show(length(current_centre))
            distance = evaluate(Euclidean(), row, current_centre)
            @show(distance)
            if distance<best_distance
                best_distance = distance
                best_index = ind
                best_index_class = ix
            end
            #if best_distance < 6.5
            if best_distance < 97584

                flag = true
                labelled_mat_of_distances[best_index,:] .= best_index_class
                ref_mat_of_distances[best_index,:] = view(test_mat_of_distances,best_index,:)
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
    Plots.heatmap(labelled_mat_of_distances)
    savefig("labelled_mat_of_distancesNMINST_test_train.png")
end


@load "all_mnmist_complete.jld" storage
nclasses=10
#@load "every_stuff.jld" mat_of_distances_test list_lists_test labels_test mat_of_distances_train M sort_idx M_ Mtest sort_idxtest M_test

function dont(storage)
    storage = storage[1:Int(length(storage)/20.0)]
    (train,test)=ScikitLearn.CrossValidation.train_test_split(storage)
    #Mtest sort_idxtest M_test
    list_lists_train,labels_train = pre_process_spike_data(train)
    mat_of_distances_train = list_to_matrix(list_lists_train)

    list_lists_test,labels_test = pre_process_spike_data(test)
    mat_of_distances_test = list_to_matrix(list_lists_test)
    @save "one_third_every_stuff.jld" mat_of_distances_test labels_test mat_of_distances_train labels_train 

    mat_of_distances_test, labels_test, mat_of_distances_train, labels_train, labels_test
end
mat_of_distances_test, labels_test, mat_of_distances_train, labels_train, labels_test = dont(storage)
#@load "every_stuff.jld" mat_of_distances_test labels_test mat_of_distances_train labels_train 

(Mtrain,sort_idx,sorted_M_train) = train_label_online(mat_of_distances_train,labels_train,nclasses)
(Mtest,sort_idxtest,sorted_M_test) = train_label_online(mat_of_distances_test,labels_test,nclasses)

distances=[]
for (row1,row2) in zip(eachrow(Mtrain),eachrow(Mtest))
    #distance = colwise(Euclidean(), Mtrain[ix], Mtest[ix])
    distance = evaluate(Euclidean(),row1, row2)
    @show(row1, row2,distance)
    push!(distances,distance)
end

p1 = Plots.heatmap(sorted_M_train)
#savefig("test_map.png")
p2 = Plots.heatmap(sorted_M_test)
Plots.plot(p1,p2)
savefig("sorted_train_map.png")


p1 = Plots.heatmap(Mtest)
#savefig("test_map.png")
p2 = Plots.heatmap(Mtrain)
Plots.plot(p1,p2)
savefig("train_map.png")



#test_label_online(Mtrain,mat_of_distances_test,labels_test,nclasses)

#get_plot_uniform(temp_container_store)
