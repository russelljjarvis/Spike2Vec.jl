

using Plots
using MAT
using StatsBase
using JLD2
using OnlineStats
using SparseArrays
using DelimitedFiles
using DataFrames
using Revise
#using ProgressMeter

"""
Augment by lengthening with duplication useful for sanity checking algorithm.
Augmentation just concatonates recordings by whole recording durations.
"""

function augment_by_time(tt,nn,scale)
    ttt = Vector{Float32}([])
    nnn = Vector{UInt32}([])
    append!(nnn,nn)
    append!(ttt,tt)
    @inbounds @showprogress for s in 1:scale
        maxt = maximum(ttt)
        @inbounds for (t,n) in zip(tt,nn)
            push!(nnn,n);
            aug_spike = Float32(t+maxt)
            push!(ttt,aug_spike)
        end

    end
    (nnn,ttt)
end

"""
Augment by lengthening with duplication useful for sanity checking algorithm.
Augmentation just concatonates recordings by total number of neurons.
"""
function augment_by_neuron_count(tt,nn,scale)
    ttt = Vector{Float32}([])
    nnn = Vector{UInt32}([])
    append!(nnn,nn)
    append!(ttt,tt)
    @inbounds @showprogress for s in 1:scale
        maxn = maximum(nnn)
        @inbounds for (t,n) in zip(tt,nn)
            aug_neuron = n+maxn
            push!(nnn,aug_neuron)
            push!(ttt,t)
        end

    end
    (nnn,ttt)
end
"""
Augment by lengthening with duplication useful for sanity checking algorithm. 
"""
 
function augment(spikes,scale)
     ttt = Vector{Float32}([])
     nnn = Vector{UInt32}([])
    for s in 1:scale
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

#=
    for s in scale
        maxn = maximum(nnn)
        for (i, t) in enumerate(spikes)
            for tt in t
                if length(t)!=0
                    aug_neuron = i+maxn
                    push!(nnn,aug_neuron);
                    push!(ttt,tt)
                end
            end
        end
    end
    (nnn,ttt)
end
=#
"""
A method to re-represent dense boolean vectors as a two dense vectors of spikes, and times.
spikes is a matrix with regularly sampled windows, populated by spikes, with calcium spikes.
"""
function convert_bool_matrice_to_raster(read_spike_dense::Matrix{Bool}, frame_width::Real)
    nodes = UInt32[]
    times = Float32[]
    @inbounds for (indy,row) in enumerate(eachrow(read_spike_dense))
        for (indx,x) in enumerate(row)
            if x
                push!(nodes,indy)
                push!(times,indx*frame_width)                
            end
        end
    end
    whole_duration = length(read_spike_dense[1,:])*frame_width
    (nodes::Vector{UInt32},times::Vector{Float32},whole_duration::Real)
end

function load_datasets_pfc()
    spikes = []
    file_read_list =  readdlm("../data2/150628_SpikeData.dat", '\t', Float64, '\n')
    nodes = [n for (t, n) in eachrow(file_read_list)]
    numb_neurons=Int(maximum(nodes))+1
    @inbounds for (t, n) in eachrow(file_read_list)
        if length(t)!=0
            push!(spikes,[])

        end
    end
    @inbounds for (t, n) in eachrow(file_read_list)
        if length(t)!=0
            push!(spikes[UInt32(n)],t)
        end
    end
    nnn_scatter=Vector{UInt32}([])
    ttt_scatter=Vector{Float32}([])
    @inbounds for (i, t) in enumerate(spikes)
        if length(t)!=0    
            @inbounds for tt in t
                        push!(nnn_scatter,i)
                        push!(ttt_scatter,Float32(tt))
                    end
        end

    end
    maxt = (maximum(ttt_scatter))    
    (nnn_scatter,ttt_scatter,spikes,numb_neurons,maxt)
    
end


function pablo_load_datasets()
    df=  CSV.read("output_spikes.csv",DataFrame)
    nodes = Vector{UInt32}(df.id)
    nodes = [UInt32(n+1) for n in nodes]
    times = Vector{Float32}(df.time_ms)
    (nodes,times)
end

function load_zebra_finche_nmc_dataset()
    spikes = []    
    file_path = "../data/songbird_spikes.txt"
    nodes = [n for (n, _) in eachrow(readdlm(file_path, '\t', Float64, '\n'))]
    @inbounds for _ in 1:maximum(unique(nodes))+1
        push!(spikes,[])
    end
    @inbounds for (n, t) in eachrow(readdlm(file_path, '\t', Float64, '\n'))
        push!(spikes[Int32(n)],t)
    end
    nodesdense=Vector{UInt32}([])
    timesdense=Vector{Float32}([])
    @inbounds for (i, timeslist) in enumerate(spikes)
        @inbounds for times in timeslist
            if length(times)!=0
                push!(nodesdense,i);
                push!(timesdense,Float32(times))
            end
        end
    end
    (nodesdense,timesdense)
end

"""
Just a helper method to get some locally stored spike data if it exists.
"""
function fromHDF5spikes()
    hf5 = h5open("spikes.h5","r")
    nodes = Vector{UInt32}(read(hf5["spikes"]["v1"]["node_ids"]))
    nodes = [n+1 for n in nodes]
    times = Vector{Float64}(read(hf5["spikes"]["v1"]["timestamps"]))
    close(hf5)
    (times,nodes)
end

"""
Of course the absolute paths below will need to be wrangled to match your directory tree.
"""

function load_datasets_calcium_v1()
    (nodes,times,whole_duration) = get_all_exempler_days()

    spikes_ragged,numb_neurons = create_spikes_ragged(nodes,times)
    (times::Vector{Float32},nodes::Vector{UInt32},whole_duration::Real,spikes_ragged::Vector{Any},numb_neurons::Int) 
    
end
#(times::Vector{Float32},nodes::Vector{UInt32},whole_duration::Float32,spikes_ragged::Vector{Any},numb_neurons) 
function get_105_neurons(nn,tt)
    times=Vector{Float32}([])
    nodes=Vector{UInt32}([])

    for (t,n) in zip(tt,nn)
        if n<105
            push!(times,t)
            push!(nodes,n)
        end
    end
    current_max_t = maximum(times)

    @save "105_neurons.jld" times nodes current_max_t
    times::Vector{Float32},nodes::Vector{UInt32},current_max_t
end

function get_105_neurons()
    (nn,tt,whole_duration) = get_all_exempler_days()

    times=Vector{Float32}([])
    nodes=Vector{UInt32}([])

    for (t,n) in zip(tt,nn)
        if n<105
            push!(times,t)
            push!(nodes,n)
        end
    end
    current_max_t = maximum(times)

    @save "105_neurons.jld" times nodes current_max_t
    times::Vector{Float32},nodes::Vector{UInt32},current_max_t
end

function get_250_neurons(nn,tt)
    times=Vector{Float32}([])
    nodes=Vector{UInt32}([])
    current_max_t = 1150
    for (t,n) in zip(tt,nn)
        if t<current_max_t
            if n<250
                push!(times,t)
                push!(nodes,n)
            end
        end
    end
    current_max_t = maximum(times)
    @save "250_neurons.jld" times nodes current_max_t
    times::Vector{Float32},nodes::Vector{UInt32},current_max_t
end

function get_all_exempler_days()
    FPS = matread("../JPMatlabFiles/M4 analyzed2DaysV.mat")["dataIntersected"][1]["Movie"]["FPS"]
    frame_width = 1.0/FPS #0.08099986230023408 #second, sample_rate =  12.3457#Hz
    length_of_spike_mat0 = length(matread("../JPMatlabFiles/M4 analyzed2DaysV.mat")["dataIntersected"])
    length_of_spike_mat1 = 1:6


    current_max_t = 0.0
    tt = Vector{Float32}([])
    nn = Vector{UInt32}([])
    init_mat = Vector{Any}([])
    @inbounds for i in 1:length_of_spike_mat0
        #if "Transients" in keys(matread("../JesusMatlabFiles/M1 analyzed2DaysV.mat")["dataIntersected"][i])
        #    init_mat = matread("../JesusMatlabFiles/M1 analyzed2DaysV.mat")["dataIntersected"][i]["Transients"]["Raster"]
        # end
        @inbounds for j in 1:6

           if "Transients" in keys(matread("../JPMatlabFiles/M$j analyzed2DaysV.mat")["dataIntersected"][i])

               temp = matread("../JPMatlabFiles/M$j analyzed2DaysV.mat")["dataIntersected"][1]["Transients"]["Raster"]
               if j==1 
                    init_mat = copy(matread("../JPMatlabFiles/M$j analyzed2DaysV.mat")["dataIntersected"][i]["Transients"]["Raster"])
               else
                    init_mat = vcat(init_mat,copy(temp))
               end
           end
        end
        #@show(init_mat)
        (nodes,times,whole_duration) = convert_bool_matrice_to_raster(init_mat,frame_width)
        Plots.scatter(times,nodes)
        savefig("plot_scatter.png")
        times = [t+current_max_t for t in times ]
        current_max_t = maximum(times)
        append!(tt,times)
        append!(nn,nodes)

    end
    Plots.scatter(tt,nn,legend = false, markersize = 0.5,markerstrokewidth=0,alpha=0.8, bgcolor=:snow2, fontcolor=:blue,xlabel="Time (ms)", ylabel="Neuron ID")
    savefig("long.png")
    (nn::Vector{UInt32},tt::Vector{Float32},current_max_t::Real)
end

