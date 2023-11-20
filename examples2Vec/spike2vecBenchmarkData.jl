using PyCall

using HDF5
using SpikeTime
using Plots
using JLD2
import PPSeq
const seq = PPSeq

# Other Imports
#import PyPlot: plt
import DelimitedFiles: readdlm
import Random
import StatsBase: quantile
using Infiltrator

@load "v1_jesus_day5.jld" nn tt
# Load data.
#if !isfile("HippocampusInSleep.jld")
#    (times,nodes) = read_path_collectionHIPPOCAMPUS()
#    @save "HippocampusInSleep.jld" times nodes
#else
#    @load "HippocampusInSleep.jld" times nodes
#end
#(nodes,times) = load_zebra_finche_nmc_dataset()
###
# PPSEQ takes 38 seconds to run on this dataset
# 38.347733 seconds (13.81 M allocations: 1.213 GiB, 0.31% gc time)
##

nodes = convert(Vector{Int64},nn)

times = convert(Vector{Float64},tt)

# metadata
num_neurons = maximum(nodes)
max_time = maximum(times)

@show(num_neurons)
@show(max_time)
@show(length(times))


# Randomly permute neuron labels.
# (This hides the sequences, to make things interesting.)
_p = Random.randperm(num_neurons)

# Load spikes.
spikes = seq.Spike[]
@inbounds for (n, t) in zip(nodes,times)
    push!(spikes, seq.Spike(_p[n], t))
end
#@show(spikes)
     


# Initialize all spikes to background process.


function get_results_PPSEQ(spikes,_p,max_time,num_neurons)


    config = Dict(

    # Model hyperparameters
    :num_sequence_types =>  2,
    :seq_type_conc_param => 1.0,
    :seq_event_rate => 1.0,

    :mean_event_amplitude => 100.0,
    :var_event_amplitude => 1000.0,

    :neuron_response_conc_param => 0.1,
    :neuron_offset_pseudo_obs => 1.0,
    :neuron_width_pseudo_obs => 1.0,
    :neuron_width_prior => 0.5,

    :num_warp_values => 1,
    :max_warp => 1.0,
    :warp_variance => 1.0,

    :mean_bkgd_spike_rate => 30.0,
    :var_bkgd_spike_rate => 30.0,
    :bkgd_spikes_conc_param => 0.3,
    :max_sequence_length => Inf,

    # MCMC Sampling parameters.
    :num_anneals => 2,
    :samples_per_anneal => 100,
    :max_temperature => 40.0,
    :save_every_during_anneal => 10,
    :samples_after_anneal => 2000,
    :save_every_after_anneal => 10,
    :split_merge_moves_during_anneal => 10,
    :split_merge_moves_after_anneal => 10,
    :split_merge_window => 1.0,

    # Masking specific parameters
    :mask_lengths => 2,
    :percent_masked => 10,
    :num_spike_resamples_per_anneal => 20,
    :num_spike_resamples => 200,
    :samples_per_resample => 10
    );
    

    # Create the masks
    init_assignments = fill(-1, length(spikes))

    @time masks = seq.create_random_mask(
        num_neurons,
        max_time+0.000001,
        config[:mask_lengths]+0.000000001, # we add a small number so no mask time is identical to a spike time, which can otherwise cause difficulty
        config[:percent_masked]
    )

    # Clean them up, glueing neighbouring masks together, avoids problems later
    @time masks = seq.merge_contiguous_masks(masks, num_neurons)

    # Construct model struct (PPSeq instance).
    @time model = seq.construct_model(config, max_time, num_neurons)

    # Run Gibbs sampling with an initial annealing period.
    #@infiltrate
    results = seq.easy_sample_masked!(model, spikes, masks, init_assignments, config);

    (results,model)
end     
@time results,model = get_results_PPSEQ(spikes,_p,max_time,num_neurons)


# Grab the final MCMC sample
final_globals = results[:globals_hist][end]
final_events = results[:latent_event_hist][end]
final_assignments = results[:assignment_hist][:, end]

# Helpful utility function that sorts the neurons to reveal sequences.
neuron_ordering = seq.sortperm_neurons(final_globals)

# Plot model-annotated raster.
fig = seq.plot_raster(
    spikes,
    final_events,
    final_assignments,
    neuron_ordering;
    color_cycle=["red", "blue"] # colors for each sequence type can be modified.
)
fig.set_size_inches([7, 3]);
     


#@load "v1_jesus_day5.jld" nn tt
#times = convert(Vector{Float32},tt)
#nodes = convert(Vector{UInt32},nn)
maxt = maximum(times)

println("number of spikes")
@show(length(times))
@show(length(unique(nodes)))
(spikes_ragged,numb_neurons) = create_spikes_ragged(nodes,times)
#=
if !isfile("PFC.jld")
    (timesPFC,nodesPFC) = read_path_collectionPFC()
    timesPFC = convert(Vector{Float32},timesPFC)
    nodesPFC = convert(Vector{UInt32},nodesPFC)
    (spikes_ragged,numb_neurons) = create_spikes_ragged(nodesPFC,timesPFC)

    @save "PFC.jld" spikes_ragged timesPFC nodesPFC
else
    @load "PFC.jld" spikes_ragged timesPFC nodesPFC
end
println("number of spikes")
@show(length(timesPFC))
@show(length(unique(nodesPFC)))

#nodes5,times5 = nn,tt
#(spike_raster,Nx,Tx,spike_raster1,Nx1,Tx1,spike_raster2,Nx2,Tx2) = fromHDF5spikesSleep()
=#


py"""
#from functools import wraps
from time import time
import numpy as np
import quantities as pq
import neo
import elephant
import time
from quantities import s,ms
from elephant.spade import spade
import timeit
import pickle
#import viziphant
def analyse_spikes_spade(spikeTrains,maxt):
    wrangle_start=timeit.timeit()
    list_of_trains=[]
    for t in spikeTrains:
        if len(t)>0:
            spk = neo.SpikeTrain(t*ms,t_stop=maxt)
            list_of_trains.append(spk)
    wrangle_end=timeit.timeit()
    delta1 = wrangle_end - wrangle_start
    print("wrangle time")
    print(delta1)
    start = timeit.timeit()
    patterns = spade(list_of_trains, bin_size=10 * pq.ms, winlen=1,
                      dither=5 * pq.ms, min_spikes=3, n_surr=3,
                      psr_param=[0, 0, 3])['patterns']  # doctest:+ELLIPSIS
    end = timeit.timeit()
    delta = end - start
    print(delta,end,start)
    with open("zebra_spade.p","wb") as f:
        pickle.dump([patterns,list_of_trains],f)
    #viziphant.patterns.plot_patterns(spiketrains, patterns)
    return (patterns,delta)
"""

(patterns,delta) = py"analyse_spikes_spade"(spikes_ragged,maxt)
@show(delta)
@show(patterns)