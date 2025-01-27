#"dataset" => ["calcium_v1_ensemble", "zebra_finche", "pfc","hippocampus"],         # it is inside vector. It is expanded.


using HDF5
using SpikeTime
using Plots
using JLD2

(timesHPC,nodesHPC) = read_path_collectionHIPPOCAMPUS()
p1=Plots.scatter(timesHPC,nodesHPC,legend=false,markersize=0.3,markerstrokewidth=0.1,markershape =:vline,markercolor = :black,yticks = 1:1:maximum(nodesHPC))

ylabel!(p1,"Neuron Id")
xlabel!(p1,"Time (sec)")
display(Plots.plot(p1))
#get_all_exempler_days_revamped()

@load "v1_jesus_day1.jld" nn tt
nodes0,times0 = nn,tt
p1=Plots.scatter(times0,nodes0,legend=false,markersize=0.3,markerstrokewidth=0.1,markershape =:vline,markercolor = :black,yticks = 1:1:maximum(nodes0))

ylabel!(p1,"Neuron Id")
xlabel!(p1,"Time (ms)")
display(Plots.plot(p1))

@load "v1_jesus_day2.jld" nn tt
nodes1,times1 = nn,tt

@load "v1_jesus_day3.jld" nn tt
nodes2,times2 = nn,tt

@load "v1_jesus_day4.jld" nn tt
nodes3,times3 = nn,tt

@load "v1_jesus_day5.jld" nn tt
nodes4,times4 = nn,tt

@load "v1_jesus_day6.jld" nn tt
nodes5,times5 = nn,tt

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