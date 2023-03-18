using Plots
using SpikingNeuralNetworks
SNN.@load_units
current = Float32[0.001 for i in 0:0.1ms:1300ms]
E = SNN.IFNF(;N = 1, param = SNN.IFParameter(), I=current)#, I=Float32[0])#;El = -49mV))
SNN.monitor(E, [:v, :fire])
SNN.sim!([E], []; duration = 1300ms)
(times,nodes) = SNN.get_trains([E])
SNN.raster(E) |> display
SNN.vecplot(E, :v) |>display

#@show(current)
