using Plots
using SpikingNeuralNetworks
SNN.@load_units


#>>> size_w_one = 50000*50000
#>>> size_w_two = 100000*50000
#>>> size_w_three = 50000*100000
#>>> size_w_four = 100000*100000
#>>> size_w_one+size_w_two+size_w_three+size_w_four


# 22 500 000 000
# 22 billion plus (half a billion)

E = SNN.IF(;N = 120000, param = SNN.IFParameter())#;El = -49mV))
@time I = SNN.IF(;N = 60000, param = SNN.IFParameter())#;El = -49mV))
EE = SNN.SpikingSynapse(E, E, :ge; σ = 60*0.27/10, p = 0.02)
EI = SNN.SpikingSynapse(E, I, :ge; σ = 60*0.27/10, p = 0.02)
IE = SNN.SpikingSynapse(I, E, :gi; σ = -20*4.5/10, p = 0.02)
II = SNN.SpikingSynapse(I, I, :gi; σ = -20*4.5/10, p = 0.02)
P = [E, I]
C = [EE, EI, IE, II]

SNN.monitor([E, I], [:fire])
@time SNN.sim!(P, C; duration = 1second)
SNN.raster(P)
SNN.train!(P, C; duration = 1second)