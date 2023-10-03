

using Plots
using MAT
using StatsBase
using JLD2
using OnlineStats
using SparseArrays
using DelimitedFiles
using DataFrames
using Revise

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

function load_datasets_calcium_jesus()
    (nodes,times,whole_duration) = get_all_exempler_of_days()

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
function get_250_neurons(nn,tt)
    times=Vector{Float32}([])
    nodes=Vector{UInt32}([])
    current_max_t = 1750
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

function get_all_exempler_of_days()
    FPS = matread("../datasets/M4 analyzed2DaysV.mat")["dataIntersected"][1]["Movie"]["FPS"]
    frame_width = 1.0/FPS #0.08099986230023408 #second, sample_rate =  12.3457#Hz
    length_of_spike_mat0 = length(matread("../datasets/M4 analyzed2DaysV.mat")["dataIntersected"])
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

           if "Transients" in keys(matread("../JesusMatlabFiles/M$j analyzed2DaysV.mat")["dataIntersected"][i])

               temp = matread("../JesusMatlabFiles/M$j analyzed2DaysV.mat")["dataIntersected"][i]["Transients"]["Raster"]
               if j==1 
                    init_mat = copy(matread("../JesusMatlabFiles/M$j analyzed2DaysV.mat")["dataIntersected"][i]["Transients"]["Raster"])
               else
                    init_mat = vcat(init_mat,copy(temp))
               end
           end
        end
        (nodes,times,whole_duration) = convert_bool_matrice_to_raster(init_mat,frame_width)
        times = [t+current_max_t for t in times ]
        current_max_t = maximum(times)
        append!(tt,times)
        append!(nn,nodes)

    end
    #tt,nn,current_max_t = get_105_neurons(copy(nn),copy(tt))
    tt,nn,current_max_t = get_250_neurons(copy(nn),copy(tt))
    Plots.scatter(tt,nn,legend = false, markersize = 0.5,markerstrokewidth=0,alpha=0.8, bgcolor=:snow2, fontcolor=:blue)
    savefig("longmysterious_scatter_plot.png")
    (nn::Vector{UInt32},tt::Vector{Float32},current_max_t::Real)

end

