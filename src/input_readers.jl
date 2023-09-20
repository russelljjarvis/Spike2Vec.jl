

using Plots
using MAT
using StatsBase
using JLD2
#using Plots
#using SpikeTime
#using DrWatson
#using ProgressMeter
using OnlineStats
using SparseArrays
#using CSV, Tables

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

"""
Just a helper method to get some locally stored spike data if it exists.
"""
function fromHDF5spikes()
    hf5 = h5open("spikes.h5","r")
    nodes = Vector{Int64}(read(hf5["spikes"]["v1"]["node_ids"]))
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
    #(nn::Vector{UInt32},tt::Vector{Float32},current_max_t::Real)

    global_isis,spikes_ragged,numb_neurons = create_ISI_histogram(nodes,times)
    #(nodes,times,whole_duration,global_isis,spikes_ragged,numb_neurons)
    (nodes,times,whole_duration,global_isis,spikes_ragged,numb_neurons) 
end
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
function get_280_neurons(nn,tt)
    times=Vector{Float32}([])
    nodes=Vector{Float32}([])
    current_max_t = 1820
    for (t,n) in zip(tt,nn)
        if t<current_max_t
            if n<280
                push!(times,t)
                push!(nodes,n)
            end
        end
    end
    current_max_t = maximum(times)
    @save "280_neurons.jld" times nodes current_max_t
    times,nodes,current_max_t
end

function get_all_exempler_of_days()
    FPS = matread("../JesusMatlabFiles/M4 analyzed2DaysV.mat")["dataIntersected"][1]["Movie"]["FPS"]
    frame_width = 1.0/FPS #0.08099986230023408 #second, sample_rate =  12.3457#Hz
    length_of_spike_mat0 = length(matread("../JesusMatlabFiles/M4 analyzed2DaysV.mat")["dataIntersected"])
    length_of_spike_mat1 = 1:6

    init_mat = matread("../JesusMatlabFiles/M1 analyzed2DaysV.mat")["dataIntersected"][1]["Transients"]["Raster"]

    current_max_t = 0.0
    tt = Vector{Float32}([])
    nn = Vector{UInt32}([])
    @inbounds for i in 1:length_of_spike_mat0
        #if "Transients" in keys(matread("../JesusMatlabFiles/M1 analyzed2DaysV.mat")["dataIntersected"][i])
        #    init_mat = matread("../JesusMatlabFiles/M1 analyzed2DaysV.mat")["dataIntersected"][i]["Transients"]["Raster"]
        # end
        @inbounds for j in 1:6
           if "Transients" in keys(matread("../JesusMatlabFiles/M$j analyzed2DaysV.mat")["dataIntersected"][i])
               temp = matread("../JesusMatlabFiles/M$j analyzed2DaysV.mat")["dataIntersected"][i]["Transients"]["Raster"]
               init_mat = vcat(init_mat,temp)
           end
        end
        (nodes,times,whole_duration) = convert_bool_matrice_to_raster(init_mat,frame_width)
        times = [t+current_max_t for t in times ]
        current_max_t = maximum(times)
        append!(tt,times)
        append!(nn,nodes)

    end
    #_,_,_ = get_280_neurons(copy(nn),copy(tt))
    tt,nn,current_max_t = get_105_neurons(copy(nn),copy(tt))
    #display(Plots.scatter(tt,nn,legend = false, markersize = 0.8,markerstrokewidth=0,alpha=0.8, bgcolor=:snow2, fontcolor=:blue,))
    #savefig("mysterious_scatter_plot.png")
    (nn::Vector{UInt32},tt::Vector{Float32},current_max_t::Real)

end

