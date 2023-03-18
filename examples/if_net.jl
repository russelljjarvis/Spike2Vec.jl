
using SpikingNeuralNetworks
SNN.@load_units
using SparseArrays
#import LinearAlgebra.normalize!
using OnlineStats
using Plots
using UnicodePlots

function makeNetGetTimes()
    scale = 1.0/500.0
    @time (_,Lee,Lie,Lei,Lii),Ne,Ni = SNN.potjans_layer(scale)
    @time (NoisyInputSynInh,NoisyInputSyn,LeeSyn,LeiSyn,LiiSyn,LieSyn,E,I,Noisy) = SNN.SpikingSynapse(Lee,Lei,Lii,Lie)
    print("wiring done")    
    P = [E,I,Noisy] # populations     
    C = [NoisyInputSynInh,NoisyInputSyn,LeeSyn,LeiSyn,LiiSyn,LieSyn] # connections
    cnt_synapses=0

    #cnt_synapses=0
    for sparse_connections in C
        cnt_synapses+=length(sparse_connections.W)
        #UnicodePlots.spy(C.W) |> display

    end

    @show(cnt_synapses)

    SNN.monitor([E,I], [:fire])
    @time SNN.sim!(P, C; duration = 0.25second)
    print("first simulation done !")
    #(times,nodes) = SNN.get_trains([E,I])
    #@time o1 = HeatMap(zip(minimum(times):maximum(times)/100.0:maximum(times),minimum(nodes):maximum(nodes/100.0):maximum(nodes)) )
    #@time fit!(o1,zip(times,convert(Vector{Float64},nodes)))
    #plot(o1, marginals=false, legend=true) #|>display 
    #Plots.savefig("default_heatmap.png")

    #SNN.monitor([E,I], [:fire])

    #@time SNN.train!(P, C; duration = 0.25second)
    
    
    (times,nodes) = SNN.get_trains([E,I])
    @time o1 = HeatMap(zip(minimum(times):maximum(times)/100.0:maximum(times),minimum(nodes):maximum(nodes/100.0):maximum(nodes)) )
    @time fit!(o1,zip(times,convert(Vector{Float64},nodes)))
    plot(o1, marginals=false, legend=true) #|>display 
    Plots.savefig("train_default_heatmap.png")

    return (P,C,times,nodes)
end

(P,C,times,nodes) = makeNetGetTimes()
