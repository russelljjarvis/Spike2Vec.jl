using Plots
using SpikingNeuralNetworks
using OnlineStats
using SparseArrays
SNN.@load_units
using CUDA
CUDA.allowscalar(false)
#using ProfileView
#unicodeplots()
function main()
    pop_size::UInt64=120
    sim_type = Vector{Float16}(zeros(1))
    E = SNN.IFNF(pop_size,sim_type)
    I = SNN.IFNF(pop_size,sim_type)
    #G = SNN.Poisson(pop_size, 1Hz)
    #G1 = SNN.Poisson(pop_size)
    #G0 = SNN.SpikingSynapse(G, E,sim_type; σ = 0.27, p = 0.00125)
    #G1 = SNN.SpikingSynapse(G, I,sim_type; σ = 0.27, p = 0.00125)
    EE = SNN.SpikingSynapse(E, E,sim_type; σ = 60*0.27/1, p = 0.015)
    EI = SNN.SpikingSynapse(E, I,sim_type; σ = 60*0.27/1, p = 0.015)
    IE = SNN.SpikingSynapse(I, E,sim_type; σ = -20*4.5/1, p = 0.015)
    II = SNN.SpikingSynapse(I, I,sim_type; σ = -20*4.5/1, p = 0.015)
    P = [E, I]
    #C = [EE, EI, IE, II]#,G0,G1]
    
    C = [EE, EI, IE, II]#,G0]
    SNN.monitor([E,I], [:fire])
    SNN.sim!(P, C; duration = 2.5second)
    print("simulation done !")
    (times,nodes) = SNN.get_trains(P)
    o1 = HeatMap(zip(0:5ms:2.5second,minimum(nodes):1:maximum(nodes)) )
    fit!(o1,zip(times,convert(Vector{Float64},nodes)))
    plot(o1, marginals=true, legend=true) |>display 
    display(SNN.raster([E,I]))
    times,nodes,E,I,C
end
times,nodes,E,I,C = main();
#current0 = Vector{Float16}([0.008 for i in 0:1ms:500ms])
#current1 = Vector{Float16}([0.011 for i in 0:1ms:500ms])
#o1 = HeatMap(zip(0:5ms:2.5second,minimum(nodes):1:maximum(nodes)) )
#fit!(o1,zip(times,convert(Vector{Float64},nodes)))
#plot(o1, marginals=true, legend=true) |>display 
#SNN.raster([E,I])|>display
#SNN.raster([E,I])#, [:fire])

