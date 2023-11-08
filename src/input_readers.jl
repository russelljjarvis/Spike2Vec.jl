

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
    @inbounds for s in 1:scale
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

#using HDF5
#=
function fromHDF5spikesSleep()
    hf5 = h5open("../data2/neuralData/hippocampus/PreprocessedDatasets/dataset_2017_08_24_sleep/2017-08-24_09-36-44.hdf5","r")
    spike_raster = []
    for (ind,(k,v)) in enumerate(pairs(read(hf5["ephys"])))
        push!(spike_raster,[])
        push!(spike_raster[ind],v["spikes"]["times"])
    end
    nodes,times = ragged_to_lists(spike_raster)
    spike_raster,nodes,times
end
#spike_raster = fromHDF5spikes()
function load_datasets_pfc150629()
    spikes = []
    file_read_list =  readdlm("../data2/neuralData/mPFC_Data/150629/150629_SpikeData.dat", '\t', Float64, '\n')
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
=#

function read_path_collectionPFC()
    paths=[]
    #paths=readdir("../data2/neuralData/mPFC_Data/")
    push!(paths,"../data2/neuralData/mPFC_Data/150630/150630_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/150628/150628_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/150629/150629_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/150701/150701_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/150704/150704_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/150705/150705_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/150706/150706_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/150707/150707_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/150708/150708_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/150711/150711_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/150712/150712_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/150713/150713_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/150714/150714_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/150715/150715_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/181011/181011_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/181012/181012_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/181014/181014_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/181017/181017_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/181018/181018_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/181019/181019_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/200223/200223_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/200224/200224_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/200227/200227_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/200228/200228_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/200301/200301_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/200302/200302_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/200303/200303_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/200308/200308_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/200309/200309_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/201219/201219_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/201220/201220_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/201221/201221_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/201222/201222_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/201223/201223_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/201226/201226_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/201227/201227_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/201228/201228_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/201229/201229_SpikeData.dat")
    push!(paths,"../data2/neuralData/mPFC_Data/201230/201230_SpikeData.dat")
   
    #massive_replay_data_set=[]

    nodes=[]
    times=[]
    @inbounds @showprogress for (ind,p) in enumerate(paths)
        p1=Plots.plot()
 
        (nnn_scatter,ttt_scatter,spikes,numb_neurons,maxt) = loadDatasetsPfcGeneric(p)
        append!(nodes,nnn_scatter)
        append!(times,ttt_scatter)
        if ind==18|ind==19|ind==20
            @save "PFC$ind" nnn_scatter ttt_scatter
        end
        
        #push!(massive_replay_data_set,(spike_raster,Nx,Tx))
        Plots.scatter!(p1,ttt_scatter,nnn_scatter,legend=false,markersize=0.6,markerstrokewidth=0.2,markershape =:vline,markercolor = :black,yticks = 1:1:maximum(nodes))
        savefig("PFCReplay_$ind.png")

    end
    #savefig("MassiveReplay.png")
    p2=Plots.plot()

    Plots.scatter!(p2,times,nodes,legend=false,markersize=0.6,markerstrokewidth=0.2,markershape =:vline,markercolor = :black,yticks = 1:1:maximum(nodes))
    ylabel!(p2,"Neuron Id")
    xlabel!(p2,"Time (us)")
    title!(p2,"PFC Replay")
    savefig("PFCReplay.png")

    (times,nodes)
end


#=
function load_datasets_pfc150628()
    spikes = []
    file_read_list =  readdlm("../data2/neuralData/mPFC_Data/150628/150628_SpikeData.dat", '\t', Float64, '\n')
    nodes = [n for (_, n) in eachrow(file_read_list)]
    numb_neurons=Int(maximum(nodes))+1
    @inbounds for (t, _) in eachrow(file_read_list)
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


function load_datasets_pfc150630()
    spikes = []
    file_read_list =  readdlm("../data2/neuralData/mPFC_Data/150630/150630_SpikeData.dat", '\t', Float64, '\n')
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
=#
function fromHDF5spikesHippGen(file_path)
    hf5 = h5open(file_path,"r")
    spike_raster = []
    for (ind,(k,v)) in enumerate(pairs(read(hf5["ephys"])))
        push!(spike_raster,[])
        push!(spike_raster[ind],v["spikes"]["times"])
    end
    Nx=Vector{Any}([])
    Tx=Vector{Any}([])
    @inbounds for (i, t) in enumerate(spike_raster)
        @inbounds for tt in t
            for x in tt
                push!(Nx,i)
                push!(Tx,x)
            end
        end
    end
    (spike_raster,Nx,Tx)
    #,spike_raster1,Nx1,Tx1,spike_raster2,Nx2,Tx2)
end
function read_path_collectionHIPPOCAMPUS()
    paths=[]
    push!(paths,"../data2/neuralData/hippocampus/PreprocessedDatasets/dataset_2017_08_23_prerun/2017-08-23_09-42-01.hdf5")
    push!(paths,"../data2/neuralData/hippocampus/PreprocessedDatasets/dataset_2017_08_24_prerun/2017-08-24_09-36-44.hdf5")
    push!(paths,"../data2/neuralData/hippocampus/PreprocessedDatasets/dataset_2017_08_25_prerun/2017-08-25_09-50-43.hdf5")
    push!(paths,"../data2/neuralData/hippocampus/PreprocessedDatasets/dataset_2017_08_23_postrun/2017-08-23_09-42-01.hdf5")
    push!(paths,"../data2/neuralData/hippocampus/PreprocessedDatasets/dataset_2017_08_24_postrun/2017-08-24_09-36-44.hdf5")
    push!(paths,"../data2/neuralData/hippocampus/PreprocessedDatasets/dataset_2017_08_25_postrun/2017-08-25_09-50-43.hdf5")
    push!(paths,"../data2/neuralData/hippocampus/PreprocessedDatasets/dataset_2017_08_24_sleep/2017-08-24_09-36-44.hdf5")
    push!(paths,"../data2/neuralData/hippocampus/PreprocessedDatasets/dataset_2017_08_25_sleep/2017-08-25_09-50-43.hdf5")
    push!(paths,"../data2/neuralData/hippocampus/PreprocessedDatasets/dataset_2017_08_23_sleep/2017-08-23_09-42-01.hdf5")
    #massive_replay_data_set=[]

    p1=Plots.plot()
    nodes=[]
    times=[]
    @inbounds @showprogress for p in paths

        (spike_raster,Nx,Tx) = fromHDF5spikesHippGen(p)
        append!(nodes,Nx)
        append!(times,Tx)
        
        #push!(massive_replay_data_set,(spike_raster,Nx,Tx))
        #Plots.scatter!(p1,Tx,Nx,legend=false,markersize=0.6,markerstrokewidth=0.2,markershape =:vline,markercolor = :black)
    end
    #savefig("MassiveReplay.png")
    p2=Plots.plot()

    Plots.scatter!(p2,times,nodes,legend=false,markersize=0.6,markerstrokewidth=0.2,markershape =:vline,markercolor = :black,yticks = 1:1:14)
    ylabel!(p2,"Neuron Id")
    xlabel!(p2,"Time (us)")
    title!(p2,"Hippocampus Replay: PreRun,PostRun, Sleep")
    savefig("HippocampusReplay.png")

    (times,nodes)
end

function loadDatasetsPfcGeneric(path)
    spikes = []
    file_read_list =  readdlm(path, '\t', Float64, '\n')
    nodes = [n for (_, n) in eachrow(file_read_list)]
    numb_neurons=Int(maximum(nodes))+1
    @inbounds for (t, _) in eachrow(file_read_list)
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
#=
function fromHDF5spikesSleep()
    hf5 = h5open("../data2/neuralData/hippocampus/PreprocessedDatasets/dataset_2017_08_24_sleep/2017-08-24_09-36-44.hdf5","r")
    spike_raster = []
    for (ind,(k,v)) in enumerate(pairs(read(hf5["ephys"])))
        push!(spike_raster,[])
        push!(spike_raster[ind],v["spikes"]["times"])
    end
    Nx=Vector{Any}([])
    Tx=Vector{Any}([])
    @inbounds for (i, t) in enumerate(spike_raster)
        @inbounds for tt in t
            for x in tt
                push!(Nx,i)
                push!(Tx,x)
            end
        end
    end

    hf50 = h5open("../data2/neuralData/hippocampus/PreprocessedDatasets/dataset_2017_08_25_sleep/2017-08-25_09-50-43.hdf5","r")  
    spike_raster1 = []
    for (ind,(k,v)) in enumerate(pairs(read(hf50["ephys"])))
        push!(spike_raster1,[])
        push!(spike_raster1[ind],v["spikes"]["times"])
    end
    Nx1=Vector{Any}([])
    Tx1=Vector{Any}([])
    @inbounds for (i, t) in enumerate(spike_raster1)
        @inbounds for tt in t
            for x in tt
                push!(Nx1,i)
                push!(Tx1,x)
            end
        end
    end

    hf51 = h5open("../data2/neuralData/hippocampus/PreprocessedDatasets/dataset_2017_08_23_sleep/2017-08-23_09-42-01.hdf5","r")  
    spike_raster2 = []
    for (ind,(k,v)) in enumerate(pairs(read(hf51["ephys"])))
        push!(spike_raster2,[])
        push!(spike_raster2[ind],v["spikes"]["times"])
    end
    Nx2=Vector{Any}([])
    Tx2=Vector{Any}([])
    @inbounds for (i, t) in enumerate(spike_raster2)
        @inbounds for tt in t
            for x in tt
                push!(Nx2,i)
                push!(Tx2,x)
            end
        end
    end
    (spike_raster,Nx,Tx,spike_raster1,Nx1,Tx1,spike_raster2,Nx2,Tx2)
end
=#
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
#=
function get_all_exempler_days()
    FPS = matread("../JPMatlabFiles/M4 analyzed2DaysV.mat")["dataIntersected"][1]["Movie"]["FPS"]
    frame_width = 1.0/FPS #0.08099986230023408 #second, sample_rate =  12.3457#Hz
    length_of_spike_mat0 = length(matread("../JPMatlabFiles/M4 analyzed2DaysV.mat")["dataIntersected"])
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
=#
function get_all_exempler_days_revamped()
    FPS = matread("../JPMatlabFiles/M4 analyzed2DaysV.mat")["dataIntersected"][1]["Movie"]["FPS"]
    frame_width = 1.0/FPS #0.08099986230023408 #second, sample_rate =  12.3457#Hz
    #@infiltrate
    current_max_t = 0.0
    init_mat = Vector{Any}([])
    #if "Transients" in keys(matread("../JesusMatlabFiles/M1 analyzed2DaysV.mat")["dataIntersected"][i])
    #    init_mat = matread("../JesusMatlabFiles/M1 analyzed2DaysV.mat")["dataIntersected"][i]["Transients"]["Raster"]
    # end
    @inbounds @showprogress for j in 1:6
        tt = Vector{Float32}([])
        nn = Vector{UInt32}([])
        i = 1
        #length_of_spike_mat0 = length(matread("../JPMatlabFiles/M$j analyzed2DaysV.mat")["dataIntersected"])
        #@inbounds for i in 1:length_of_spike_mat0
        if "Transients" in keys(matread("../JPMatlabFiles/M$j analyzed2DaysV.mat")["dataIntersected"][i])
            if j==1 
                init_mat = copy(matread("../JPMatlabFiles/M$j analyzed2DaysV.mat")["dataIntersected"][i]["Transients"]["Raster"])
                (nodes,times,_) = convert_bool_matrice_to_raster(init_mat,frame_width)
                times = [t+current_max_t for t in times ]
                current_max_t = maximum(times)
                append!(tt,times)
                append!(nn,nodes)
        
                @save "v1_jesus_day$j.jld" nn tt
                Plots.scatter(tt,nn,legend=false,markersize=0.6,markerstrokewidth=0.2,markershape =:vline,markercolor = :black)
                savefig("v1_Month$j.plot_scatter.png")

            else
                init_mat = copy(matread("../JPMatlabFiles/M$j analyzed2DaysV.mat")["dataIntersected"][i]["Transients"]["Raster"])

                (nodes,times,_) = convert_bool_matrice_to_raster(init_mat,frame_width)
                times = [t+current_max_t for t in times ]
                current_max_t = maximum(times)
                append!(tt,times)
                append!(nn,nodes)            
                @save "v1_jesus_day$j.jld" nn tt
                Plots.scatter(tt,nn,legend=false,markersize=0.6,markerstrokewidth=0.2,markershape =:vline,markercolor = :black)
                savefig("v1_Month$j.plot_scatter.png")

            end
        end
        #end
        #(nodes,times,_) = convert_bool_matrice_to_raster(init_mat,frame_width)
        
    end
    #Plots.scatter(tt,nn,legend = false, markersize = 0.5,markerstrokewidth=0,alpha=0.8, bgcolor=:snow2, fontcolor=:blue,xlabel="Time (ms)", ylabel="Neuron ID")
    #savefig("long.png")
    #(nn::Vector{UInt32},tt::Vector{Float32},current_max_t::Real)
end


