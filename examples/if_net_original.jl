using Plots
using SpikingNeuralNetworks
using OnlineStats

SNN.@load_units

E = SNN.IF(;N = 200000, param = SNN.IFParameter())#;El = -49mV))
I = SNN.IF(;N = 60000, param = SNN.IFParameter())#;El = -49mV))
EE = SNN.SpikingSynapse(E, E, :ge; σ = 60*0.27/10, p = 0.02)
EI = SNN.SpikingSynapse(E, I, :ge; σ = 60*0.27/10, p = 0.02)
IE = SNN.SpikingSynapse(I, E, :gi; σ = -20*4.5/10, p = 0.02)
II = SNN.SpikingSynapse(I, I, :gi; σ = -20*4.5/10, p = 0.02)
P = [E, I]
C = [EE, EI, IE, II]

cnt_synapses=0
for sparse_connections in C
    cnt_synapses=+length(C.W)
end
@show(cnt_synapses)
SNN.monitor([E, I], [:fire])
@time SNN.sim!(P, C; duration = 2second)
print("simulation done !")
(times,nodes) = SNN.get_trains([E,I])
@time o1 = HeatMap(zip(minimum(times):maximum(times)/100.0:maximum(times),minimum(nodes):maximum(nodes/100.0):maximum(nodes)) )
@time fit!(o1,zip(times,convert(Vector{Float64},nodes)))
plot(o1, marginals=false, legend=true) #|>display 
Plots.savefig("default_heatmap.png")

SNN.raster(P)
SNN.train!(P, C; duration = 1second)
print("training done !")

(times,nodes) = SNN.get_trains([E,I])
@time o1 = HeatMap(zip(minimum(times):maximum(times)/100.0:maximum(times),minimum(nodes):maximum(nodes/100.0):maximum(nodes)) )
@time fit!(o1,zip(times,convert(Vector{Float64},nodes)))
plot(o1, marginals=false, legend=true) #|>display 
Plots.savefig("default_heatmap.png")
