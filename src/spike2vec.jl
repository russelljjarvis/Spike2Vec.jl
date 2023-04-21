using HDF5
using Plots
using OnlineStats
#using ThreadsX
using Plots
using JLD2
using SpikeSynchrony
using LinearAlgebra
using ColorSchemes
using AngleBetweenVectors
using Revise
using OnlineStats, Plots#, Random
#using DataFrames
using UMAP
using Distances

"""
Divide epoch into windows.
"""
function divide_epoch(nodes,times,sw,toi)
    t1=[]
    n1=[]
    t0=[]
    n0=[]
    @assert sw< toi
    third = toi-sw
    @assert third==300
    for (n,t) in zip(nodes,times)
        if sw<=t && t<toi
            append!(t0,t-sw)
            append!(n0,n)            
        elseif t>=toi && t<=toi+third
            append!(t1,abs(t-toi))
            @assert t-toi>=0
            append!(n1,n)
        end
    end
    neuron0 =  Array{}([Float32[] for i in 1:maximum(nodes)+1])
    for (neuron,t) in zip(n0,t0)
        append!(neuron0[neuron],t)        
    end
    neuron0
end
"""
Using the windowed spike trains for neuron0: a uniform surrogate spike train reference, versus neuron1: a real spike train in the  target window.
compute the intergrated spike distance quantity in that time frame.

SpikeSynchrony is a Julia package for computing spike train distances just like elephant in python
And in every window I get the population state vector by comparing current window to uniform spiking windows
But it's also a good idea to use the networks most recently past windows as reference windows 

"""
function get_vector_coords(neuron0::Vector{Vector{Float32}}, neuron1::Vector{Vector{Float32}}, self_distances::Vector{Float32})
    for (ind,(n0_,n1_)) in enumerate(zip(neuron0,neuron1))        
        if length(n0_) != 0 && length(n1_) != 0
            pooledspikes = vcat(n0_,n1_)
            maxt = maximum(sort!(unique(pooledspikes)))
            t1_ = sort(unique(n0_))
            t0_ = sort(unique(n1_))
            t, S = SPIKE_distance_profile(t0_,t1_;t0=0,tf = maxt)
            self_distances[ind]=sum(S)
        else
            self_distances[ind]=0
        end
    end
    self_distances
end
"""
Just a helper method to get some locally stored data if it exists.
"""
function get_()
    hf5 = h5open("spikes.h5","r")
    nodes = Vector{Int64}(read(hf5["spikes"]["v1"]["node_ids"]))
    nodes = [n+1 for n in nodes]
    times = Vector{Float64}(read(hf5["spikes"]["v1"]["timestamps"]))
    close(hf5)
    (times,nodes)
end

"""
For polar visualizations of spike distance vectors
"""
function looped!(times,t0,spk_counts,segment_length,temp)
    doonce = LinRange(0.0, segment_length, temp)[:]
    for (neuron, t) in enumerate(t0)
        times[neuron] = doonce
    end
end
"""
Generate uniform surrogate spike trains that fire at the networks mean firing rate
"""
function surrogate_to_uniform(times_,segment_length)
    times =  Array{}([Float32[] for i in 1:length(times_)])
    spk_counts = []
    for (neuron, t) in enumerate(times_)
        append!(spk_counts,length(t))
    end
    temp = 4
    looped!(times,times_,spk_counts,segment_length,temp)
    times
end

function get_plot()
    times,nodes = get_()
    division_size = 10
    step_size = maximum(times)/division_size
    end_window = collect(step_size:step_size:step_size*division_size)
    spike_distance_size = length(end_window)
    start_windows = collect(0:step_size:step_size*division_size-1)
    mat_of_distances = zeros(spike_distance_size,maximum(unique(nodes))+1)
    n0ref = divide_epoch(nodes,times,start_windows[3],end_window[3])
    segment_length = end_window[3] - start_windows[3]
    t0ref = surrogate_to_uniform(n0ref,segment_length)
    PP = []
    for (ind,toi) in enumerate(end_window)
        self_distances = Array{Float32}(zeros(maximum(nodes)+1))
        sw = start_windows[ind]
        neuron0 = divide_epoch(nodes,times,sw,toi)    
        self_distances = get_vector_coords(neuron0,t0ref,self_distances)
        mat_of_distances[ind,:] = self_distances
        #@show(self_distances)
    end
    cs1 = ColorScheme(distinguishable_colors(spike_distance_size, transform=protanopic))
    p=nothing
    for (ind,_) in enumerate(eachrow(mat_of_distances))
        temp = (mat_of_distances[ind,:].- mean(mat_of_distances[ind,:]))./std(mat_of_distances[ind,:])
        n = length(temp)
        θ = LinRange(0, 2pi, n)
        if ind==1
            p = plot(θ, temp, proj=:polar,color=cs1[ind])#, layout = length(mat_of_distances))
        else 
            plot!(p,θ,mat_of_distances[ind,:], proj=:polar,color=cs1[ind])  |>display
        end

    end
    return mat_of_distances
end
#mat_of_distances = get_plot()

function post_proc_viz(mat_of_distances)
    angles0 = []
    distances0 = []
    angles1 = []
    distances1 = []
    p=nothing
    for (ind,self_distances) in enumerate(eachrow(mat_of_distances))
        if ind>1
            θ = angle(mat_of_distances[ind,:],mat_of_distances[ind-1,:])
            r = evaluate(Euclidean(),mat_of_distances[ind,:],mat_of_distances[ind-1,:])
            append!(angles1,θ)
            append!(distances1,r)        

            #scatter!(p,θ,r,color=cs1[ind])  |>display
        end
        θ = angle(mat_of_distances[ind,:],mat_of_distances[1,:])
        r = evaluate(Euclidean(),mat_of_distances[ind,:],mat_of_distances[1,:])
        append!(angles0,θ)
        append!(distances0,r)        
    end
    return angles0,distances0,angles1,distances1
end

function final_plots(mat_of_distances)
    normalize!(mat_of_distances[:,:])

    for (ind,row) in enumerate(eachcol(mat_of_distances))
        mat_of_distances[:,ind].- mean(mat_of_distances[:,ind])./std(mat_of_distances[:,ind])
    end
    angles0,distances0,angles1,distances1 = post_proc_viz(mat_of_distances)
    display(scatter(angles0,distances0))#,color=cs1))
    display(scatter(angles1,distances1))#,color=cs1))
    cs1 = ColorScheme(distinguishable_colors(size(mat_of_distances)[1], transform=protanopic))
    plot!(angles1,distances1, proj=:polar)  |>display   
end

#=
function get_ts(nodes,times)
    #num_spikes = length(nodes)
    # The temporal resolution of the final timesurface
    dt = 10
    num_neurons = Int(length(unique(nodes)))+1#int(df.max(axis=0)['x1'])
    total_time =  Int(maximum(times))
    time_resolution = Int(round(total_time/dt))
    # Final output. 
    final_timesurf = zeros((num_neurons, time_resolution+1))
    # Timestamp and membrane voltage store for generating time surface
    timestamps = zeros((num_neurons)) .- Inf
    mv = zeros((num_neurons))
    tau = 200
    last_t = 0
    for (tt,nn) in zip(times,nodes)
        #Get the current spike
        neuron = Int(nn) 
        time = Int(tt)        
        # If time of the next spikes leaps over, make sure to generate 
        # timesurfaces for all the intermediate dt time intervals and fill the 
        # final_timesurface.
        if time > last_t
            timesurf = similar(final_timesurf[:,1])
            for t in collect(last_t:dt:time)
                @. timesurf = mv*exp((timestamps-t)/tau)
                final_timesurf[:,1+Int(round(t/dt))] = timesurf
            end
            last_t = time
        end
        # Update the membrane voltage of the time surface based on the last value and time elapsed
        mv[neuron] =mv[neuron]*exp((timestamps[neuron]-time)/tau) +1
        timestamps[neuron] = time
        # Update the latest timestamp at the channel. 
    end
    # Generate the time surface for the rest of the time if there exists no other spikes. 
    timesurf = similar(final_timesurf[:,1])
    for t in collect(last_t:dt:total_time)
        @. timesurf = mv*exp((timestamps-t)/tau)
        final_timesurf[:,1+Int(round(t/dt))] = timesurf
    end
    return final_timesurf

end
=#