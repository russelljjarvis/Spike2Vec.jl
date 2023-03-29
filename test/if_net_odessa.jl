using Plots
using SpikingNeuralNetworks
using OnlineStats
using SparseArrays
SNN.@load_units
using CUDA
CUDA.allowscalar(false)
using Test
using Revise
using Odesa
#import Odesa.Feast



pop_size::Int32=100
sim_type = Vector{Float32}(zeros(1))
sim_duration = 1.0second
u1 = Float32[10.0*abs(4.0*rand()) for i in 0:0.01ms:sim_duration]
E = SNN.IFNF(pop_size,sim_type)
I = SNN.IFNF(pop_size,sim_type)
EE = SNN.SpikingSynapse(E, E,sim_type; σ = 160*0.27/1, p = 0.025)
EI = SNN.SpikingSynapse(E, I,sim_type; σ = 160*0.27/1, p = 0.025)
IE = SNN.SpikingSynapse(I, E,sim_type; σ = -160*0.27/1, p = 0.25)
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

feast_layer_nNeurons::Int32 = pop_size*2
feast_layer_eta::Float32 = 0.001
feast_layer_threshEta::Float32 = 0.001
feast_layer_thresholdOpen::Float32 = 0.1
feast_layer_tau::Float32 = 0.464
# This doesn't matter, it is used in ODESA but not in FEAST 
feast_layer_traceTau::Float32 = 0.81
# Create a Feast layer with the above parameters
feast_layer = Odesa.Feast.FC(Int32(1),Int32(pop_size*2),feast_layer_nNeurons,feast_layer_eta,feast_layer_threshEta,feast_layer_thresholdOpen,feast_layer_tau,feast_layer_traceTau)

for (y,ts) in zip(nodes,times)
    Odesa.Feast.forward(feast_layer, Int32(1), Int32(y), ts)    
end

Odesa.Feast.reset_time(feast_layer)