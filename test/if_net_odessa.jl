using Plots
using SpikingNeuralNetworks
using OnlineStats
using SparseArrays
SNN.@load_units

using Test
using Revise
##
# Since Odesa is not an officially Julia registered package, running this example would require:
# using Pkg
# Pkg.add(url=https://github.com/russelljjarvis/Odesa.jl-1")
# ] resolve
# ] activate
# ] instantiate
##
using Odesa
using StatsBase

pop_size::Int32=100
sim_type = Vector{Float32}(zeros(1))
sim_duration = 1.0second
u1 = Float32[10.0*abs(4.0*rand()) for i in 0:1ms:sim_duration]
E = SNN.IFNF(pop_size,sim_type)
I = SNN.IFNF(pop_size,sim_type)
EE = SNN.SpikingSynapse(E, E,sim_type; σ = 160*0.27/1, p = 0.025)
EI = SNN.SpikingSynapse(E, I,sim_type; σ = 160*0.27/1, p = 0.055)
IE = SNN.SpikingSynapse(I, E,sim_type; σ = -160*0.27/1, p = 0.250)
II = SNN.SpikingSynapse(I, I,sim_type; σ = -160*0.27/1, p = 0.15)
P = [I, E]
C = [EE, EI, IE, II]

##
# ToDO make a real interface that uses block arrays.
## 
SNN.monitor([C], [:g])
SNN.monitor([E, I], [:fire])
inh_connection_map=[(E,EE,1,E),(E,EI,1,I)]
exc_connection_map=[(I,IE,-1,E),(I,II,-1,I)]
connection_map = [exc_connection_map,inh_connection_map]
SNN.sim!(P, C;conn_map= connection_map, current_stim = u1, duration = sim_duration)
print("simulation done !")
(times,nodes) = SNN.get_trains([E,I])#,Gx,Gy])
#display(SNN.raster([E,I]))
function get_mean_isis(times,nodes)
    spike_dict = Dict()
    for n in unique(nodes)
        spike_dict[n] = []
    end


    for (st,n) in zip(times,nodes)
        append!(spike_dict[n],st)
    end

    all_isis = []
    for (k,v) in pairs(spike_dict)
        time_old = 0
        for time in spike_dict[k][1:end-1]
            isi = time - time_old
            append!(all_isis,isi)
            time_old = time
        end
    end
    #@show(StatsBase.mean(all_isis))
    mean_isi = StatsBase.mean(all_isis)
end
mean_isi = get_mean_isis(times,nodes)

feast_layer_nNeurons::Int32 = 20# pop_size*2
feast_layer_eta::Float32 = 0.001
feast_layer_threshEta::Float32 = 0.001
feast_layer_thresholdOpen::Float32 = 0.01
feast_layer_tau::Float32 =  1.0/mean_isi #/2.0)/2.0#0.464
# This doesn't matter, it is used in ODESA but not in FEAST 
feast_layer_traceTau::Float32 = 0.81
precision::UInt16 = convert(UInt32,0)  

feast_layer = Odesa.Feast.FC(precision,Int32(1),Int32(pop_size*2),feast_layer_nNeurons,feast_layer_eta,feast_layer_threshEta,feast_layer_thresholdOpen,feast_layer_tau,feast_layer_traceTau)

perm = sortperm(times)
nodes = nodes[perm]
times = times[perm]
winners = []
p1=plot(feast_layer.thresh)
display(SNN.raster([E,I]))
function collect_distances(feast_layer,nodes,times)
    distances = feast_layer.dot_prod

    for i in 1:2
        Odesa.Feast.reset_time(feast_layer)
        for (y,ts) in zip(nodes,times)
            winner = Odesa.Feast.forward(feast_layer, Int16(1), Int16(y), Float16(ts))    
            distances = feast_layer.dot_prod
            
        end
        #display(plot!(p1,feast_layer.thresh,legend=false))
    end
    distances
end
distances = collect_distances(feast_layer,nodes,times)

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
final_timesurf = get_ts(nodes,times);
@show(sum(final_timesurf))
display(Plots.heatmap(final_timesurf))

#using CSV, Tables, DataFrames

#neurons_and_times = zeros((length(times),2))
#neurons_and_times[:,1] = nodes
#neurons_and_times[:,2] = times

# write out a DataFrame to csv file
#df = DataFrame(neurons_and_times, :auto)

#CSV.write("times_for_yesh.csv", df)

# write a matrix to an in-memory IOBuffer
#io = IOBuffer()
#mat = rand(10, 10)
#CSV.write(io, Tables.table(mat))
