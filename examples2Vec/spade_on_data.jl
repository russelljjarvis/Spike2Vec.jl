using PyCall

using HDF5
using SpikeTime
using Plots
using JLD2

#(times,nodes) = read_path_collectionHIPPOCAMPUS()
(nodes,times) = load_zebra_finche_nmc_dataset()

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