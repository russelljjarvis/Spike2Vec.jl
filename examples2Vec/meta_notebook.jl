using DrWatson
import DrWatson.dict_list
using JuliaSyntax
using JET;
using SpikeTime
using JLD2
#JET
#report_and_watch_file("meta_notebook.jl",annotate_types=true)

##
## TODO 
## * Online Clustering of jobs
## * Sliding window
## * Consistent time window size in milliseconds.
## * 
#@quickactivate "SpikeThroughput.jl" #

function get_window_size()
    return
end

allparams = Dict(
    "dataset" => ["calcium_v1_ensemble", "zebra_finche", "pfc","hippocampus"],         # it is inside vector. It is expanded.
    "window_size" => [100, 200],         # same
    "similarity_threshold" => [5,10], # single element inside vector; no expansion
)

if !isfile("param_dict.jld")
    dicts = dict_list(allparams)
    @save "param_dict.jld" dicts
else
    @load "param_dict.jld" dicts

end
function preparesim(d::Dict)
    @unpack dataset, window_size, similarity_threshold = d
    local expanding_param
    expanding_param = copy(d)
    if dataset=="calcium_v1_ensemble"
        (times,nodes,whole_duration,spikes_ragged,numb_neurons)  = load_datasets_calcium_v1()
    elseif dataset=="zebra_finche"
        (nodes,times) = load_zebra_finche_nmc_dataset()
    elseif dataset=="pfc"
        (nodes,times,spikes,numb_neurons,maxt)= load_datasets_pfc()

    end
    expanding_param["spikes_ragged"] = spikes_ragged
    expanding_param["numb_neurons"] = numb_neurons
    expanding_param["times"] = times
    expanding_param["nodes"] = nodes

    return expanding_param
end

if !isfile("preparesim.jld")
    d = dicts[1]
    param_dict = preparesim(d)
    @save "preparesim.jld" param_dict
else
    @load "preparesim.jld" param_dict

end
#for (i, d) in enumerate(dicts)
f = doanalysis(param_dict)
@tagsave(param_dict["dataset"], f)
#end
#SNN.@load_units
#sim_duration = 3.0second

config = @dict(pop_size,sim_duration,division_size)

produce_or_load(datadir("simulation"),
                config,
                run_simulation)