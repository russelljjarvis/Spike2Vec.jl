using Plots
using SpikingNeuralNetworks
using OnlineStats
using SparseArrays
SNN.@load_units
using CUDA
CUDA.allowscalar(false)
using ProfileView

function main()
    pop_size::UInt64=1200
    sim_type = Vector{Float16}(zeros(1))
    E = SNN.IFNF(pop_size,sim_type)
    I = SNN.IFNF(pop_size,sim_type)
    G = SNN.Rate(pop_size)
    GG = SNN.RateSynapse(G, E; σ = 10.2, p = 0.5)
    GG = SNN.RateSynapse(G, I; σ = 10.2, p = 0.5)
    EE = SNN.SpikingSynapse(E, E,sim_type; σ = 60*0.27/1, p = 0.05)
    EI = SNN.SpikingSynapse(E, I,sim_type; σ = 60*0.27/1, p = 0.05)
    IE = SNN.SpikingSynapse(I, E,sim_type; σ = -20*4.5/1, p = 0.05)
    II = SNN.SpikingSynapse(I, I,sim_type; σ = -20*4.5/1, p = 0.05)
    P = [E, I]
    C = [EE, EI, IE, II]
    SNN.monitor([E,I], [:fire])
    SNN.sim!(P, C; duration = 2.5second)
    print("simulation done !")
    (times,nodes) = SNN.get_trains([E,I])
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

