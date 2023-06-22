using Plots
using SpikingNeuralNetworks
SNN.@load_units
u1 = Float32[0.052929259 for i in 50:0.1ms:150ms]
#u2 = Float32[0 for i in 0:0.1ms:500ms]
#u3 = Float32[0 for i in 250ms:0.1ms:3500ms]
#append!(u2,u1)
#append!(u2,u3)
#@show(u2)
sim_type = Vector{Float32}(zeros(1))
#E = SNN.IFNF(pop_size,sim_type)
E = SNN.IFNF(Int32(1),sim_type,u1)#, I=Float32[0])#;El = -49mV))
SNN.monitor(E, [:v, :fire])
SNN.sim!([E], []; duration = 150ms)
(times,nodes) = SNN.get_trains([E])

#@show(current)
SNN.raster(E) |> display
SNN.vecplot(E, :v) |>display
