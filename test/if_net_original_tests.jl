using Plots
using SpikingNeuralNetworks
using OnlineStats
using SparseArrays
SNN.@load_units
using CUDA
CUDA.allowscalar(false)
using Test
#using ProfileView
#unicodeplots()
using Revise

function assign_gids(list_of_pops)
    gid = 1
    offset = 0
    for p in list_of_pops
        if offset == 0
            p.gid = collect(1:p.pop_size)
            offset += p.pop_size
        else
            p.gid = collect(offset:offset+p.pop_size)
        end
    end

end



function forwards_here!(colptr::Vector{<:Real}, I, W,fireJ::Vector{Bool},g::Vector)
    fill!(g, zero(Float32))

    @inbounds for j in 1:(length(colptr) - 1)
        #if fireJ[j]
        for s in colptr[j]:(colptr[j+1] - 1)
            g[I[s]] += W[s]
        end
        #end
    end
    replace!(g, Inf=>0.0)
    replace!(g, NaN=>0.0)   
    replace!(g,-Inf16=>0.0)
    g
    #@show(sum(g))

end
function main()
    pop_size::UInt64=2000
    sim_type = Vector{Float32}(zeros(1))
    u1 = Float32[0.052929259 for i in 50:0.1ms:150ms]

    E = SNN.IFNF(pop_size,sim_type)
    I = SNN.IFNF(pop_size,sim_type)
    #Gx = SNN.Poisson(pop_size, 20Hz)
    #Gy = SNN.Poisson(pop_size, 20Hz)

    #G0 = SNN.SpikingSynapse(Gx, E,sim_type; σ = 2.9927, p = 0.9925)

    #G1 = SNN.SpikingSynapse(Gy, I,sim_type; σ = 1.827, p = 0.990125)

    EE = SNN.SpikingSynapse(E, E,sim_type; σ = 60*0.27/1, p = 0.15)
    EI = SNN.SpikingSynapse(E, I,sim_type; σ = 11600*0.27/1, p = 0.99)
    IE = SNN.SpikingSynapse(I, E,sim_type; σ = -20*4.5/1, p = 0.125)
    II = SNN.SpikingSynapse(I, I,sim_type; σ = -20*4.5/1, p = 0.15)
    P = [E, I]#,Gx,Gy]
    #C = [EE, EI, IE, II]#,G0,G1]
    
    C = [EE, EI, IE, II]#$,G0,G1]
    #SNN.monitor(P, [:fire])#,:v,:ge,:gi])
    SNN.monitor([C], [:g])
    SNN.monitor([E, I], [:fire])

    #SNN.raster(P)

    inh_connection_map=[(E,EE,1,E),(E,EI,1,I)]
    
    exc_connection_map=[(I,IE,-1,E),(I,II,-1,I)]
    connection_map = [exc_connection_map,inh_connection_map]
    SNN.sim!(P, C;conn_map= connection_map, duration = 5.5second)
    print("simulation done !")
    (times,nodes) = SNN.get_trains([E,I])#,Gx,Gy])
    #@show(length(nodes))
    #@show(times)
    #,
    (EE,II,C,times,nodes,E,I)
end


#times,nodes,
(times,nodes,EE,II,C,E,I) = main();
display(SNN.raster([E,I]))

#@show(times)
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

