
using HDF5
using SpikeTime
using Plots
using JLD2
using DrWatson
import DrWatson.dict_list
using BenchmarkTools
using UMAP

function get_data_set_dict()
    DataSetDict = Dict()
    (nodes,times) = load_zebra_finche_nmc_dataset()
    DataSetDict["zebrafinch"] = (nodes,times,"Zebra Finch")
    (times,nodes) = read_path_collectionHIPPOCAMPUS()
    DataSetDict["Hippocampus"] = (nodes,times,"Hippocampus")
    (nodes,times) = read_path_collectionPFC()
    #nodes = [n+1 for n in nodes]
    #DataSetDict["PFC"] = (nodes,times,"PFC")
    
    @load "v1_jesus_day1.jld" nn tt
    nodes,times = nn,tt
    #maxt = maximum(times)
    DataSetDict["v1_jesus_day1"] = (nodes,times,"V1 Day 1")
    @load "v1_jesus_day2.jld" nn tt
    nodes,times = nn,tt
    #maxt = maximum(times)
    DataSetDict["v1_jesus_day2"] = (nodes,times,"V1 Day 2")
    @load "v1_jesus_day3.jld" nn tt
    nodes,times = nn,tt
    #maxt = maximum(times)
    DataSetDict["v1_jesus_day3"] = (nodes,times,"V1 Day 3")
    
    DataSetDict
end



function do_UMAP_times(nodes,times,figname,figtitle)
    step_size = dt = 0.25
    tau = 0.5
    ts = get_ts(nodes,times,step_size,tau)#;disk=false)
    ts1 = ts[:, vec(mapslices(col -> any(col .!= 0), ts, dims = 1))]
    Q_embedding = umap(ts1,5,n_neighbors=3)
    Plots.plot(Plots.scatter(Q_embedding[1,:], Q_embedding[2,:],zcolor=1:size(Q_embedding, 2), title=figtitle, marker=(1, 1, :auto, stroke(1.5)),legend=false))
    savefig(figname)
end
function do_UMAP_nodes(nodes,times,figname,figtitle)
    step_size = dt = 0.25
    tau = 0.5
    ts = get_ts(nodes,times,step_size,tau)#;disk=false)
    ts1 = ts[:, vec(mapslices(col -> any(col .!= 0), ts, dims = 1))]
    Q_embedding = umap(ts1',5,n_neighbors=3)
    Plots.plot(Plots.scatter(Q_embedding[1,:], Q_embedding[2,:],zcolor=1:size(Q_embedding, 2), title=figtitle, marker=(1, 1, :auto, stroke(1.5)),legend=false))
    savefig(figname)
end

DataSetDict = get_data_set_dict()
for (key,value) in pairs(DataSetDict)
    (nodes,times,figtitle) = value
    @show(value[3])
    do_UMAP_times(nodes,times,"UMAP_on_Time_Surface_of_"*value[3]*".png","UMAP_on_Nodes_of_"*value[3])
    do_UMAP_nodes(nodes,times,"UMAP_on_Nodes_of_"*value[3]*".png","UMAP_on_Nodes_of_"*value[3])
end