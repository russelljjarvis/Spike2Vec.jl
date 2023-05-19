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
using ScikitLearn


function get_plot_uniform(temp_container_store)
    times,nodes = temp_container_store[1],temp_container_store[3]
    spike_distance_size = length(temp_container_store)
    (refnodes,reftimes)=(nodes[1],times[1])
    n0ref = expected_spike_format(refnodes,reftimes)
    non_zero_spikings=[]
    @inbounds for tx in n0ref
        if length(tx)!=0
            push!(non_zero_spikings,length(tx)) 
        end
    end
    mean_spk_counts = Int32(round(sum(non_zero_spikings)))
    temp = LinRange(0.0, maximum(reftimes), mean_spk_counts)
    linear_uniform_spikes = Vector{Float32}([i for i in temp])
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
    labelsl = Vector{Any}([])
    @inbounds @showprogress for (ind,s) in enumerate(temp_container_store)
        (times,labels,nodes) = (s[1],s[2],s[3]) # (tts,label,pop_stimulation)
        self_distance_populate = Array{Float32}(zeros(1220))
        if length(times) > 1
            times = expected_spike_format(nodes,times)
            get_vector_coords_uniform!(linear_uniform_spikes,times,self_distance_populate)
            push!(list_lists,self_distance_populate)
            push!(labelsl,labels)
        end
    end
    return list_lists,labelsl
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
        mat_of_distances[ind,:] = l[:]
    end
    @inbounds @showprogress for (ind,col) in enumerate(eachcol(mat_of_distances))
        mat_of_distances[:,ind] .= (col.-mean(col))./std(col)
    end
    mat_of_distances[isnan.(mat_of_distances)] .= 0.0
    return mat_of_distances
end

function train_label_online(mat_of_distances,ll,nclasses)
    R = kmeans(mat_of_distances', nclasses; maxiter=22000)#, display=:iter)
    a = assignments(R) # get the assignments of points to clusters
    c = counts(R) # get the cluster sizes
    M = R.centers # get the cluster centers
    sort_idx =  sortperm(assignments(R))
    #for (i,row) in enumerate(eachrow(mat_of_distances'))
    #    @show(i,sort_idx[i])
    #end
    M_ = mat_of_distances'[:,sort_idx]
    Plots.heatmap(M_)
    savefig("clustered_train_model.png")
    return M,sort_idx,M_
end

function test_label_online(M,test_mat_of_distances,nclasses,labels_test)
    labelled_mat_of_distances = zeros(size(test_mat_of_distances))
    ref_mat_of_distances = zeros(size(test_mat_of_distances))
    gtruth = zeros(size(test_mat_of_distances))

    @showprogress for (ind,row) in enumerate(eachrow(test_mat_of_distances))
        best_distance = 100000.0
        gtruth[ind,:] .= labels_test[ind][1]
        for best_index in collect(1:nclasses)
            current_centre = M[:,best_index]
            distance = evaluate(Euclidean(), row, current_centre)
            if distance<best_distance
                best_distance = distance
                labelled_mat_of_distances[ind,:] .= best_index
            end

        end
    end
    p1=Plots.heatmap(gtruth)
    #savefig("labelled_mat_of_distancesNMINST_test_train.png")

    p2 = Plots.heatmap(labelled_mat_of_distances)
    Plots.plot(p1,p2)
    savefig("labelled_mat_of_distancesNMINST_test_train.png")
    (labelled_mat_of_distances,gtruth)
end
function accuracy_test_label_online(labelled_mat_of_distances,gtruth)
    collections_ = []
    temp_list = []
    old=-1
    @showprogress for (ind,row) in enumerate(eachrow(gtruth))
        if row[1]!=old
            push!(collections_,temp_list)
            new_class_ind = ind
            temp_list = []
        end
        old=row[1]
        push!(temp_list,labelled_mat_of_distances[ind])
    end
    for c in collections_
        if length(c)>1
            
            @show(round(mean(c);sigdigits=1))
        end
    end
    return collections_
end
(labelled_mat_of_distances,gtruth) = test_label_online(Mtrain,train,nclasses,labels_train)

collections_ = accuracy_test_label_online(labelled_mat_of_distances,gtruth)


@load "all_mnmist_complete.jld" storage
nclasses=10
#@save "not_complete_every_stuff.jld" mat_of_distances_test labels_test mat_of_distances_train labels_train 

function dont(storage)
    #storage = storage[1:Int(length(storage))]
    #train = storage[1:Int(length(storage)/2)]
    #test = storage[Int(length(storage)/2)+1:Int(length(storage))]
    train = storage
    test = storage

    #@show(train)
    #@show(test)

    #Mtest sort_idxtest M_test
    list_lists_train,labels_train = pre_process_spike_data(train)
    
    #@list_lists_train = list_lists_train[1:Int(length(list_lists_train)/2)]

    mat_of_distances_train = list_to_matrix(list_lists_train)
    mat_of_distances_test = copy(mat_of_distances_train)
    labels_test = copy(labels_train)
    #list_lists_test,labels_test = pre_process_spike_data(test)
    #list_lists_test = list_lists_train[Int(length(list_lists_train/2))+1:Int(length(list_lists_train))]

    #mat_of_distances_test = list_to_matrix(list_lists_train)
    @save "not_complete_every_stuff.jld" mat_of_distances_test labels_train mat_of_distances_train labels_test  

    mat_of_distances_test, labels_test, mat_of_distances_train, labels_train
end
#@load "not_complete_every_stuff.jld" test labels_test train labels_train

test, labels_test, train, labels_train = dont(storage)
#@load "every_stuff.jld" mat_of_distances_test labels_test mat_of_distances_train labels_train 
#@load "not_complete_every_stuff.jld" test labels_test train labels_train  

#=
matl = vcat(labels_test,labels_train)
labels_test = matl[1:length(labels_test),:]
labels_train = matl[length(labels_test)+1:2*length(labels_test),:]

mat = vcat(mat_of_distances_test,mat_of_distances_train)
test = mat[1:length(mat)/2,:]
train = mat[length(mat)/2+1:2*length(mat)/2,:]
#matl = vcat(labels_test,labels_train)
=#
#(Mtrain,sort_idxtrain,sorted_M_train) = train_label_online(train,labels_train,nclasses)
#(Mtest,sort_idxtest,sorted_M_test) = train_label_online(test,labels_test,nclasses)
#(trainM,trainL,testM,testL)=ScikitLearn.CrossValidation.train_test_split(train,labels_train, test_size=0.5)


test_label_online(Mtrain,train,nclasses,labels_train)

function dont_again()
    distancesx=[]


    for (row1,row2) in zip(eachcol(Mtrain),eachcol(test))
        #distance = colwise(Euclidean(), Mtrain[ix], Mtest[ix])
        distance = evaluate(Euclidean(),row1, row2)
        #@show(row1, row2,distance)
        push!(distances,distance)
    end
    distancesy=[]

    for (row1,row2) in zip(eachrow(train),eachrow(test))
        #distance = colwise(Euclidean(), Mtrain[ix], Mtest[ix])
        distance = evaluate(Euclidean(),row1, row2)
        #@show(distance)
        push!(distances,distance)
    end


    p1 = Plots.heatmap(Mtrain)
    #savefig("test_map.png")
    p2 = Plots.heatmap(Mtest)
    Plots.plot(p1,p2)
    savefig("cluster_centres_map.png")



    p1 = Plots.heatmap(sorted_M_train)
    #savefig("test_map.png")
    p2 = Plots.heatmap(sorted_M_test)
    Plots.plot(p1,p2)
    savefig("sorted_train_map.png")


    p1 = Plots.heatmap(test)
    #savefig("test_map.png")
    p2 = Plots.heatmap(train)
    Plots.plot(p1,p2)
    savefig("train_map.png")

    for col in eachcol(Mtrain)
        @show(col)
    end
end

#get_plot_uniform(temp_container_store)
