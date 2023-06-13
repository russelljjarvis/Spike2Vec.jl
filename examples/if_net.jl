
using SpikeTime
SpikeTime.@load_units
using SparseArrays
#import LinearAlgebra.normalize!
using OnlineStats
using Plots
using UnicodePlots
using SparseArrays
Ne = 800;      
Ni = 200
total_cnt = Ne+Ni
final_connectome = spzeros(total_cnt,total_cnt)
p = 0.25
σ = 0.2
wee = σ * sprand(Ne, Ne, p) 
final_connectome[1:Ne,1:Ne] = wee
wei = σ * sprand(Ne, Ni, p) 
final_connectome[Ni+1:total_cnt,1:Ni] = wei



σ = -0.2

wii = σ * sprand(Ni, Ni, p) 
final_connectome[1:Ni,1:Ni] = wii

wie = σ * sprand(Ni, Ne, p) 
final_connectome[1:Ni,Ni+1:total_cnt] = wie


ragged_array_targets = []
for (x,row) in enumerate(eachrow(final_connectome))
    push!(ragged_array_targets,[])
end
for (x,row) in enumerate(eachrow(final_connectome))
    for (y,i) in enumerate(row)
        if i!=0
            push!(ragged_array_targets[x],y)
            #@show(x,i)   
        end 
    end
end

ragged_array_weights = []
for (x,row) in enumerate(eachrow(final_connectome))
    push!(ragged_array_weights,[])
end
for (x,row) in enumerate(eachrow(final_connectome))
    for (y,i) in enumerate(row)
        if i!=0
            push!(ragged_array_weights[x],i)
        end 
    end
end

sim_type = Vector{Float32}([])
pop = IFNF(total_cnt,sim_type,ragged_array_targets)

#@time SpikeTime.sim!(P, C; duration = 0.25second)

# Now Construct a population.

function makeNetGetTimes()
    #scale = 1.0/500.0
    #@time (_,Lee,Lie,Lei,Lii),Ne,Ni = SNN.potjans_layer(scale)
    #@time (NoisyInputSynInh,NoisyInputSyn,LeeSyn,LeiSyn,LiiSyn,LieSyn,E,I,Noisy) = SpikeTime.SpikingSynapse(Lee,Lei,Lii,Lie)
    #print("wiring done")    
    #P = [E,I,Noisy] # populations     
    #C = [NoisyInputSynInh,NoisyInputSyn,LeeSyn,LeiSyn,LiiSyn,LieSyn] # connections
    #cnt_synapses=0

    #cnt_synapses=0
    for sparse_connections in C
        cnt_synapses+=length(sparse_connections.W)
        #UnicodePlots.spy(C.W) |> display

    end

    @show(cnt_synapses)

    SpikeTime.monitor([E,I], [:fire])
    @time SpikeTime.sim!(P, C; duration = 0.25second)
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
