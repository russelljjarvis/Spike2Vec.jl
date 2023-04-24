using PyCall
using StaticArrays
using ProgressMeter
using JLD2
py"""
from os.path import exists
import os
import tonic
import torch
dataset = tonic.datasets.SMNIST(os.getcwd()+'../datasets',train=False) #,train=True,num_neurons=999,dt=1.0)

def get_info(dataset):
    return (len(dataset),dataset.sensor_size)

def get_dataset_item(dataset, index):
    events, label = dataset[index]
    return (events,label,len(dataset),dataset.sensor_size)
"""

#self._dataset = tonic.datasets.SMNIST(os.getcwd()+"../datasets",train=False,num_neurons=999,dt=1.0,train=train, first_saccade_only=first_saccade_only,transform=transform))

#dataset = py"tonic.datasets.SMNIST(os.getcwd()+'../datasets',train=False,num_neurons=999,dt=1.0,train=train, first_saccade_only=first_saccade_only,transform=transform))"

#SM = py"SMNIST()"   

function julia_tonic_spike_cache()
    (iter_num,dim) = py"get_info"(py"dataset")
    @show(typeof(iter_num))
    spikes = []#Array{Int32(iter_num),Tuple}
    labels = []#Array{Int32(iter_num),Tuple}

    @inbounds @showprogress for i in 1:100#iter_num
        events,label = py"get_dataset_item"(py"dataset", i)
        #@show(events)        
        midway_events = convert(Array{Array{Int32}}, events)
        append!(spikes,midway_events)
        midway_label = convert(Int8, label)
        append!(labels,midway_label)

        #midway = convert(SArray{lenght(midway),Array{Int32}}, midway)
        #final = Tuple{Int32,length(midway),3}(midway)
    end
    @save "SMNISTspikey.jld" spikes labels

    (spikes,labels)

end
spikes,labels = julia_tonic_spike_cache()

#final = convert(Array{length(midway),{Tuple{Int32,3}}},midway)
#pre_final = SArray{NTuple{3,Int32}}
#@show(typeof(events))
#@show(typeof(label))
#events::Array{NTuple{5,Int64},1}
#=
assert(len(indices) <= 100)
all_events = []

for id,index in enumerate(indices):
    (grid_x,grid_y) = np.unravel_index(id,(10,10))
    
    events, label = dataset[index]
    label_array = np.full(events['x'].shape[0],label,dtype=[('label','i8')])
    event_array = merge_arrays((events,label_array),flatten=True)
    event_array['x'] = grid_x*36 + event_array['x'] + 1
    event_array['y'] = grid_y*36 + event_array['y'] + 1
    # event_array[:,3] -= event_array[0,3]
    all_events.append(event_array)
super_events = np.hstack(all_events)
super_events = super_events[super_events['t'].argsort()]
return super_events
=#