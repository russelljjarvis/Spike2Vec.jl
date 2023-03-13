
using SpikingNeuralNetworks
SNN.@load_units
using SparseArrays
import LinearAlgebra.normalize!
using OnlineStats
using Plots

function makeNetGetTimes()
    scale = 1.0/10.0
    @time (_,Lee,Lie,Lei,Lii),Ne,Ni = SNN.potjans_layer(scale)
    @time (NoisyInputSynInh,NoisyInputSyn,LeeSyn,LeiSyn,LiiSyn,LieSyn,E,I,Noisy) = SNN.SpikingSynapse(Lee,Lei,Lii,Lie)
    print("wiring done")    
    P = [E,I,Noisy] # populations     
    C = [NoisyInputSynInh,NoisyInputSyn,LeeSyn,LeiSyn,LiiSyn,LieSyn] # connections
    SNN.monitor([E,I], [:fire])
    @time SNN.sim!(P, C; duration = 2.0second)
    print("simulation done !")
    SNN.raster([E,I]) #|> display
    Plots.savefig("cheap_dirty_plot.png")


    (times,nodes) = SNN.get_trains([E,I])
    
    @time o1 = HeatMap(zip(minimum(times):maximum(times)/100.0:maximum(times),minimum(nodes):maximum(nodes/100.0):maximum(nodes)) )
    @time fit!(o1,zip(times,convert(Vector{Float64},nodes)))
    plot(o1, marginals=false, legend=true) #|>display 
    Plots.savefig("default_heatmap.png")

    return (P,C,times,nodes)
end
#=
function plot_results(times,nodes)
    nbins = 525
    data = SNN.bespoke_2dhist(nbins,times,nodes)
    foreach(normalize!, eachcol(data))
    Plots.plot(heatmap(data),legend = false, normalize=:pdf)
    Plots.savefig("untrainedHeatMap_raster_trained.png")
    SNN.train!(P, C; duration = 3second)
    SNN.raster(P)
    Plots.savefig("default_raster_trained.png")
    (times,nodes) = SNN.get_trains(P)
    nbins = 525
    data = SNN.bespoke_2dhist(nbins,times,nodes)
    foreach(normalize!, eachcol(data))
    Plots.plot(heatmap(data),legend = false, normalize=:pdf)
    Plots.savefig("trainingHeatMap_raster_trained.png")
    print("simulation results plotted")
end
=#
function main()
    (P,C,times,nodes) = makeNetGetTimes()
    #plot_results(times,nodes,P)
end
main()