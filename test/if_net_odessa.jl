using Plots
using SpikingNeuralNetworks
using OnlineStats
using SparseArrays
SNN.@load_units
#using CUDA
#CUDA.allowscalar(false)
using Test
using Revise
using Odesa
#import Odesa.Feast



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

feast_layer_nNeurons::Int32 = 20# pop_size*2
feast_layer_eta::Float32 = 0.001
feast_layer_threshEta::Float32 = 0.001
feast_layer_thresholdOpen::Float32 = 0.01
feast_layer_tau::Float32 =  1.0/Int(round(sum(unique(times))/(pop_size*2)))#/2.0)/2.0#0.464
# This doesn't matter, it is used in ODESA but not in FEAST 
feast_layer_traceTau::Float32 = 0.81
# Create a Feast layer with the above parameters
feast_layer = Odesa.Feast.FC(Int32(1),Int32(pop_size*2),feast_layer_nNeurons,feast_layer_eta,feast_layer_threshEta,feast_layer_thresholdOpen,feast_layer_tau,feast_layer_traceTau)

perm = sortperm(times)
nodes = nodes[perm]
times = times[perm]
winners = []
p1=plot(feast_layer.thresh)
display(SNN.raster([E,I]))
for i in 1:325
    Odesa.Feast.reset_time(feast_layer)
    for (y,ts) in zip(nodes,times)
        winner = Odesa.Feast.forward(feast_layer, Int32(1), Int32(y), ts)    
        if i==125
            append!(winners,winner)
        end
        
    end
    display(plot!(p1,feast_layer.thresh,legend=false))
end
