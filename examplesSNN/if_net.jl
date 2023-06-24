
using SpikeTime
#SpikeTime.@load_units
using SparseArrays
#import LinearAlgebra.normalize!
using OnlineStats
using Plots
#using UnicodePlots
Ne = 1800     
Ni = 1800
total_cnt = Ne+Ni
final_connectome = spzeros(total_cnt,total_cnt)
σ = 0.21
p = 0.55
wee = σ * sprand(Ne, Ne, p) 
final_connectome[1:Ne,1:Ne] = wee
σ = 0.15
p = 0.85

wei = σ * sprand(Ne, Ni, p) 
final_connectome[1:Ni,Ni+1:total_cnt] = wei

σ = -0.25
p = 0.75
wii = σ * sprand(Ni, Ni, p) 
final_connectome[Ni:total_cnt-1,Ni:total_cnt-1] = wei

σ = -0.05
p = 0.125
wie = σ * sprand(Ni, Ne, p) 
final_connectome[Ni+1:total_cnt,1:Ni] = wii
# = wie


Plots.heatmap(final_connectome)
savefig("balanced_if_net_structure.png")
#=
ragged_array_targets = []
for (x,row) in enumerate(eachrow(final_connectome))
    push!(ragged_array_targets,[])
end
for (x,row) in enumerate(eachrow(final_connectome))
    for (y,i) in enumerate(row)
        if i!=0
            push!(ragged_array_targets[x],y)
        end 
    end
end
=#
ragged_array_weights = []
for (x,row) in enumerate(eachrow(final_connectome))
    push!(ragged_array_weights,[])
end
for (x,row) in enumerate(eachrow(final_connectome))
    for (y,i) in enumerate(row)
        push!(ragged_array_weights[x],i)
    end
end
ragged_array_weights = [ i for i in ragged_array_weights ]
total_cnt = length(ragged_array_weights)
sim_type = Vector{Float32}([])

pop = SpikeTime.IFNF(total_cnt,sim_type,ragged_array_weights)
#current_stim=9.79#125

pop.u = Vector{Float32}([rand(4.1:11.92222) for i in 1:Int(round(length(pop.fire)))])
@show(pop.u)
SpikeTime.monitor([pop], [:fire])
#dt::Real = 1ms, duration::Real = 10ms
#sim!(P::IFNF{Int64, Vector{Bool}, Vector{Float32}}; dt::Real = 1ms, duration::Real = 10ms)#;current_stim=nothing)

simx!(pop; dt=0.1, duration=6000.0)
(Tx,Nx) = SpikeTime.get_trains([pop])
xlimits = maximum(Tx)
#p= Plots.scatter(,legend = false,markersize = 0.5,markerstrokewidth=0,alpha=0.8, bgcolor=:snow2, fontcolor=:black,xlabel="Time (ms)",ylabel="Neuron Index")
display(Plots.scatter(Tx,Nx,legend = false,markersize = 0.35,markerstrokewidth=0,alpha=0.8, bgcolor=:snow2, fontcolor=:black,xlabel="Time (ms)",ylabel="Neuron Index"))
savefig("balanced_random_spikes.png")
#display(Plots.scatter(Tx,Nx,legend = false,markersize = 0.8,markerstrokewidth=0,alpha=0.8, bgcolor=:snow2, fontcolor=:blue, xlims=(0.0, xlimits)))

#display(Plots.scatter(times,nodes))

#@show(times)
#@show(nodes)
#function sim!(P, dt = 1ms, duration = 10ms,current_stim=nothing)

#@time SpikeTime.sim!(P, C; duration = 0.25second)

# Now Construct a population.
#=
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
=#
#(P,C,times,nodes) = makeNetGetTimes()
