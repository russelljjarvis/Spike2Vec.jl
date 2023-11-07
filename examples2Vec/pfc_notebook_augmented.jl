using DrWatson
import DrWatson.dict_list
#using JuliaSyntax
#using JET;
using SpikeTime
using JLD2
using Plots
using JLD2
#(nodes,times,spikes,numb_neurons,maxt) = load_datasets_pfc()
(times,nodes,whole_duration,spikes_ragged,numb_neurons)  = load_datasets_calcium_v1()
times,nodes,current_max_t = get_250_neurons(nn,tt)
#p2 = Plots.scatter!(px,Txag,Nxag,markercolor=Int(l),markersize = 1.2,markerstrokewidth=0,alpha=0.8, fontcolor=:blue,legend=false)

Plots.scatter(times,nodes,markersize = 0.2,markerstrokewidth=0,alpha=0.8, fontcolor=:blue,legend=false, xlabel="Time (ms)",ylabel="Neuron Id")
savefig("the_scatter_plot0.png")
scale = 20
#(nodes,times) = augment_by_time(times,nodes,scale)

(nodes,times) = augment_by_neuron_count(times,nodes,scale)
Plots.scatter(times,nodes,markersize = 0.2,markerstrokewidth=0,alpha=0.8, fontcolor=:blue,legend=false, xlabel="Time (ms)",ylabel="Neuron Id")

savefig("the_scatter_plot1000pfc.png")

scale = 5
(nodes,times) = augment_by_time(times,nodes,scale)

Plots.scatter(times,nodes,markersize = 0.2,markerstrokewidth=0,alpha=0.8, fontcolor=:blue,legend=false, xlabel="Time (ms)",ylabel="Neuron Id")
savefig("the_scatter_plot12000pfc.png")
@save "large_augmented_data_set.jld2" nodes times

#end
#for (i, d) in enumerate(dicts)
#f = doanalysis(param_dict)
#@tagsave(param_dict["dataset"], f)
#end
#SNN.@load_units
#sim_duration = 3.0second

#config = @dict(pop_size,sim_duration,division_size)

#produce_or_load(datadir("simulation"),
#                config,
#                run_simulation)