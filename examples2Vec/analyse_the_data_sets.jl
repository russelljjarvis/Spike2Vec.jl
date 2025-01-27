

using HDF5
using SpikeTime
using Plots
using JLD2
using DrWatson
import DrWatson.dict_list
using BenchmarkTools
using UMAP

using Infiltrator
(nodes,times) = load_zebra_finche_nmc_dataset()
result = maximum(times)/22.0
#@show(result)
param_dict = Dict()
param_dict["number_divisions"] = 44# same

# ideal bin size is half a second
# or 500 milliseconds


#22.2 * 54.05810810810811 = v1_jesus_day1
#22.2 seconds

#22.2 / 1200.3137f0

#
# maximum(tt) = 
# 22.2
# 1200.094f0 1,200 seconds
# 0.9998169f0
# either seconds or milliseconds.

#get_ze
#get_zebra
#@@infiltrate
#@load "v1_jesus_day2.jld" nn tt
#param_dict = Dict()
#param_dict["number_divisions"] = 500# same

#nodes,times = nn,tt
#@show(maximum(tt))
#(times,nodes) = read_path_collectionHIPPOCAMPUS()

#p3=Plots.scatter(times5,nodes5,legend=false,markersize=0.3,markerstrokewidth=0.1,markershape =:vline,markercolor = :black,yticks = 1:1:maximum(nodes5))

#ylabel!(p3,"Neuron Id")
#xlabel!(p3,"Time (ms)")
#savefig("whatOfTimes.png")
param_dict["similarity_threshold"] = 31.5 #9548088f0 # single element inside vector; no expansion

param_dict["nodes"] = nodes
param_dict["times"] = times
#param_dict = dict_list(param_dict)[1]
param_struct = (; (Symbol(k) => v for (k,v) in pairs(param_dict))...)
#@show(param_struct)
#doanalysCV(param_struct)

#distmat,div_spike_mat_with_displacement,spikes_ragged,NURS_sum,sum_of_rep,sfs,timesList = 
#@benchmark 
doanalysisrev(param_struct)

#more_plotting1(spikes_ragged,sort_idx,IStateI,p1)
#=
p1 = Plots.plot()
for state_frequency_histogram in sfs 
    Plots.histogram!(p1,state_frequency_histogram)
end
savefig("state_frequency_histogram.png")
more_plotting0(distmat,div_spike_mat_with_displacement,nodes5,times5,timesList)
=#
#=
(timesHPC,nodesHPC) = read_path_collectionHIPPOCAMPUS()
p1=Plots.scatter(timesHPC,nodesHPC,legend=false,markersize=0.3,markerstrokewidth=0.1,markershape =:vline,markercolor = :black,yticks = 1:1:maximum(nodesHPC))

ylabel!(p1,"Neuron Id")
xlabel!(p1,"Time (sec)")
display(Plots.plot(p1))
#get_all_exempler_days_revamped()

@load "v1_jesus_day1.jld" nn tt
nodes0,times0 = nn,tt
p2=Plots.scatter(times0,nodes0,legend=false,markersize=0.3,markerstrokewidth=0.1,markershape =:vline,markercolor = :black,yticks = 1:1:maximum(nodes0))

ylabel!(p1,"Neuron Id")
xlabel!(p1,"Time (ms)")
display(Plots.plot(p2))

display(Plots.plot(p1,p2))

@load "v1_jesus_day2.jld" nn tt
nodes1,times1 = nn,tt

@load "v1_jesus_day3.jld" nn tt
nodes2,times2 = nn,tt

@load "v1_jesus_day4.jld" nn tt
nodes3,times3 = nn,tt

@load "v1_jesus_day5.jld" nn tt
nodes4,times4 = nn,tt
=#


#(nodesPFC,timesPFC) = read_path_collectionPFC()


#(nnn_scatterpfc1,ttt_scatterpfc1,spikes,numb_neurons,maxt) = load_datasets_pfc150628()
#(nnn_scatterpfc2,ttt_scatterpfc2,spikes,numb_neurons,maxt) = load_datasets_pfc150629()
#(nnn_scatterpfc3,ttt_scatterpfc3,spikes,numb_neurons,maxt) = load_datasets_pfc150630()

#(spike_raster,Nx,Tx,spike_raster1,Nx1,Tx1,spike_raster2,Nx2,Tx2) = fromHDF5spikesSleep()

#p2=Plots.scatter(Tx1,Nx1,legend=false,markersize=0.6,markerstrokewidth=0.2,markershape =:vline,markercolor = :black)
#p3=Plots.scatter(Tx2,Nx2,legend=false,markersize=0.6,markerstrokewidth=0.2,markershape =:vline,markercolor = :black)

#p4=Plots.scatter(ttt_scatterpfc1,nnn_scatterpfc1,legend=false,markersize=0.6,markerstrokewidth=0.2,markershape =:vline,markercolor = :black)
#p5=Plots.scatter(ttt_scatterpfc2,nnn_scatterpfc2,legend=false,markersize=0.6,markerstrokewidth=0.2,markershape =:vline,markercolor = :black)
#p6=Plots.scatter(ttt_scatterpfc3,nnn_scatterpfc3,legend=false,markersize=0.6,markerstrokewidth=0.2,markershape =:vline,markercolor = :black)


#ylabel!(p2,"Neuron Id")
#xlabel!(p2,"Time (ms)")


#ylabel!(p3,"Neuron Id")
#xlabel!(p3,"Time (us)")
#title!(p3,"pfc150628")

#ylabel!(p4,"Neuron Id")
#xlabel!(p4,"Time (us)")
#title!(p4,"pfc150629")


#ylabel!(p5,"Neuron Id")
#xlabel!(p5,"Time (us)")
#title!(p5,"pfc150630")


#px = Plots.plot(p1,p2,p3)
#Plots.plot(px,title="hippocampus in sleep")
#savefig("HippocampusInSleep.png")
#py = Plots.plot(p4,p5,p6)
#Plots.plot(py,title="PFC replay in sleep")
#savefig("PFCInSleep.png")
#(times,nodes) = read_path_collection()