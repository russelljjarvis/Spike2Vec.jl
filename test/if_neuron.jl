using Revise
using SpikingNeuralNetworks
SNN.@load_units

sim_type = Vector{Float32}(zeros(1))
#u1 = Float32[0.052929259 for i in 50:0.1ms:150ms]

E = SNN.IFNF(1,sim_type)

#E = SNN.IFNF(1)
#E.I = [11]
SNN.monitor(E, [:v, :fire])

for t = 1:1000
    #E.I .= 5randn(Ne)
    #$I.I .= 2randn(Ni)
    E.gi .= 10.4*randn(E.N)
    #E.ge .= 10.4*randn(E.N)
    E.gi = -abs.(E.gi)
    #E.ge = abs.(E.ge)
    SNN.integrate_neuron!(E.N, E.v,1ms, E.ge, E.gi, E.fire, E.u, E.tr)
    #SNN.integrate_neuron!(E, 1ms)
    SNN.record!(E)
    #SNN.sim!([E], 1ms)
end


#SNN.sim!([E], []; duration = 300ms)
#SNN.vecplot(E.v) |> display
SNN.vecplot([E], :v) |> display

E = SNN.IFNF(1,sim_type)
SNN.monitor(E, [:v, :fire])


for t = 1:1000
    #E.I .= 5randn(Ne)
    #$I.I .= 2randn(Ni)
    #E.gi .= 10.4*randn(E.N)
    E.ge .= 1.4*randn(E.N)
    #E.gi = -abs.(E.u)
    E.ge = 10*abs.(E.ge)
    SNN.integrate_neuron!(E.N, E.v,1ms, E.ge, E.gi, E.fire, E.u, E.tr)
    #SNN.integrate_neuron!(E, 1ms)
    SNN.record!(E)
    
    #SNN.sim!([E], 1ms)
end
#SNN.vecplot([E], :v) |> display
#@show(sum(E.fire))

#SNN.raster([E]) |> display
