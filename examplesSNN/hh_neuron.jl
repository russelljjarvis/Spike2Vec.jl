using Plots
using SpikingNeuralNetworks
SNN.@load_units

E = SNN.HH(;N = 1)
E.I = [0.001]

SNN.monitor(E, [:v])
SNN.sim!([E], []; dt = 0.01ms, duration = 100ms)
SNN.vecplot(E, :v)
