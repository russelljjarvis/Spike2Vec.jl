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
    pop_size::UInt64=2000
    sim_type = Vector{Float16}(zeros(1))
    u1 = Float32[0.052929259 for i in 50:0.1ms:150ms]

    E = SNN.IFNF(pop_size,sim_type)
    I = SNN.IFNF(pop_size,sim_type)
    Gx = SNN.Poisson(pop_size, 20Hz)
    Gy = SNN.Poisson(pop_size, 20Hz)

    G0 = SNN.SpikingSynapse(Gx, E,sim_type; σ = 2.9927, p = 0.9925)
    #G0 = SNN.SpikingSynapse(E,Gx,sim_type; σ = 0.27, p = 0.925)

    G1 = SNN.SpikingSynapse(Gy, I,sim_type; σ = 1.827, p = 0.990125)

    EE = SNN.SpikingSynapse(E, E,sim_type; σ = 60*0.27/1, p = 0.015)
    EI = SNN.SpikingSynapse(E, I,sim_type; σ = 60*0.27/1, p = 0.015)
    IE = SNN.SpikingSynapse(I, E,sim_type; σ = -20*4.5/1, p = 0.015)
    II = SNN.SpikingSynapse(I, I,sim_type; σ = -20*4.5/1, p = 0.015)
    P = [E, I,Gx,Gy]
    #C = [EE, EI, IE, II]#,G0,G1]
    
    C = [EE, EI, IE, II,G0,G1]
    SNN.monitor(P, [:fire])#,:v,:ge,:gi])
    #SNN.monitor([C], [:g])
    
    SNN.sim!(P, C; duration = 3.5second)
    print("simulation done !")
    #(times,nodes) = SNN.get_trains([E,I])#,Gx,Gy])
    display(SNN.raster([E,I]))
    #,
    EE,II,C#$#,times,nodes
end


#times,nodes,
#$times,nodes,
EE,II,C = main()

#@show(unique(nodes))

#o1 = HeatMap(zip(0:5ms:2.5second,minimum(nodes):1:maximum(nodes)) )
#fit!(o1,zip(times,convert(Vector{Float64},nodes)))
#plot(o1, marginals=true, legend=true) |>display 

#SNN.vecplot(EE, :ge);# plot!(f.(ts))
#SNN.vecplot(EE, :v);# plot!(f.(ts))
#SNN.vecplot(C, :g)

#current0 = Vector{Float16}([0.008 for i in 0:1ms:500ms])
#current1 = Vector{Float16}([0.011 for i in 0:1ms:500ms])
#o1 = HeatMap(zip(0:5ms:2.5second,minimum(nodes):1:maximum(nodes)) )
#fit!(o1,zip(times,convert(Vector{Float64},nodes)))
#plot(o1, marginals=true, legend=true) |>display 
#SNN.raster([E,I])|>display
#SNN.raster([E,I])#, [:fire])

