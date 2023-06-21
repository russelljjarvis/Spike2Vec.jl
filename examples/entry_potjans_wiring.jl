
using SGtSNEpi, Random
using Revise
using CairoMakie, Colors, LinearAlgebra
using GLMakie
using Graphs
using SpikeTime

import StatsBase.mean
using Plots
include("../src/models/genPotjansWiring.jl")
using JLD2
function grab_connectome(scale)
   

    pot_conn = potjans_layer(scale)

    Plots.heatmap(pot_conn,xlabel="post synaptic",ylabel="pre synaptic")
    savefig("Potjans_connectome_no_input_layer.png")

    if false
        # display(pot_conn)
        Lx = Vector{Int64}(zeros(size(pot_conn[1,:])))
        Lx = convert(Vector{Int64},Lx)
        # The Graph Network analysis can't handle negative weight values so upset every weight to make weights net positive.
        stored_min = abs(minimum(pot_conn))
        for (ind,row) in enumerate(eachrow(pot_conn))
            for (j,colind) in enumerate(row)
                if pot_conn[ind,j] < 0.0
                    pot_conn[ind,j] = pot_conn[ind,j]+stored_min+1.0
                end
            end
            @assert mean(pot_conn[ind,:]) >= 0.0
        end
        savefig("connection_matrix.png")
        dim = 2

        Y0 = 0.1 * randn( size(pot_conn,1), dim);
        cmap_out = distinguishable_colors(
            length(Lx),
            [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

        #display(SGtSNEpi.show_embedding( Y, Lx ,A=pot_conn;edge_alpha=0.035,lwd_in=0.035,lwd_out=0.009,clr_out=cmap))
        #fig = 
        display(SGtSNEpi.show_embedding( Y0, Lx ,A=pot_conn;edge_alpha=0.05,lwd_in=0.05,lwd_out=0.013,clr_out=cmap_out))
        #display(fig)
        #savefig("SGtSNEpi_connection.png")#
        save("SGtSNEpi_connection.png")
    end
    pot_conn
end


if isfile("potjans_wiring.jld")
    #
    # TODO delete following  two lines, just grabbing a plot!
    #scale = 0.07225
    #pot_conn = grab_connectome(scale)

    @load "potjans_wiring.jld" pot_conn ragged_array_weights

else
    scale = 0.07225
    pot_conn = grab_connectome(scale)
    ragged_array_weights = []
    for (x,row) in enumerate(eachrow(pot_conn))
        push!(ragged_array_weights,[])
    end
    for (x,row) in enumerate(eachrow(pot_conn))
        for (y,i) in enumerate(row)
            if i!=0
                push!(ragged_array_weights[x],i)
            end 
        end
    end
    @save "potjans_wiring.jld" pot_conn ragged_array_weights


end





function dont(pop)

    sim_type = Vector{Float32}([])
    total_cnt = length(ragged_array_weights)
    pop = SpikeTime.IFNF(total_cnt,sim_type,ragged_array_weights)
    current_stim=10.0125

    pop.u = Vector{Float32}([current_stim for i in 1:length(pop.fire)])
    SpikeTime.monitor([pop], [:fire])
    @time sim!(pop; dt=0.1, duration=1000.0)
    @time (Tx,Nx) = SpikeTime.get_trains([pop])
    xlimits = maximum(Tx)

    @time display(Plots.scatter(Tx,Nx,legend = false,markersize = 0.8,markerstrokewidth=0,alpha=0.8, bgcolor=:snow2, fontcolor=:blue, xlims=(0.0, xlimits)))
end
if !isfile("NMNIST_spike_packet_conc_v.jld")

    @load "../data2/all_mnmist_complete.jld" storage

    empty_spike_cont, labelsl, packet_window_boundaries= NMNIST_pre_process_spike_data(storage)
    @save "NMNIST_spike_packet_conc_v.jld"  empty_spike_cont labelsl packet_window_boundaries
#@show(empty_spike_cont)
else
    @load "NMNIST_spike_packet_conc_v.jld"  empty_spike_cont labelsl packet_window_boundaries
end
#@show(packet_window_boundaries[1])
#@show(packet_window_boundaries[2])
#@show(last(packet_window_boundaries))

if !isfile("final_topology.jld")

    Nextra = 1220     
    total_cnt_prev = length(ragged_array_weights)
    total_cnt_final = total_cnt_prev + Nextra
    final_connectome = spzeros(total_cnt_final,total_cnt_final)
    p = 1.0
    σ = 0.2
    wexternal_stim = ones(total_cnt_prev,Nextra)
    final_connectome[1:total_cnt_prev,total_cnt_prev+1:total_cnt_final] = wexternal_stim
    Plots.heatmap(final_connectome,xlabel="post synaptic",ylabel="pre synaptic")
    savefig("Potjans_connectome_input_layer.png")

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
    total_cnt = length(ragged_array_weights)

    @save "final_topology.jld" ragged_array_weights total_cnt

else
    @load "final_topology.jld" ragged_array_weights total_cnt
end 

sim_type = Vector{Float32}([])
pop = SpikeTime.IFNF(total_cnt,sim_type,ragged_array_weights)
#current_stim=10.0125

#pop.u = Vector{Float32}([current_stim for i in 1:length(pop.fire)])
SpikeTime.monitor([pop], [:fire])
external_layer_indexs=total_cnt_prev+1:total_cnt_final

duration = maximum(empty_spike_cont)+maximum(empty_spike_cont)/4.0
sim!(pop; dt=0.1, duration=duration,spike_stim=empty_spike_cont,external_layer_indexs=external_layer_indexs)
@time (Tx,Nx) = SpikeTime.get_trains([pop])
xlimits = maximum(Tx)

@time display(Plots.scatter(Tx,Nx,legend = false,markersize = 0.8,markerstrokewidth=0,alpha=0.8, bgcolor=:snow2, fontcolor=:blue, xlims=(0.0, xlimits)))

#1:1220
#@load "../data2/all_mnmist_complete.jld" storage

#pot_conn[diagind(pot_conn)] .= 0.0
#=
G = DiGraph(pot_conn)
results = label_propagation(G)#, maxiter=1000; rng=nothing, seed=nothing)
@show(results)
γ = 0.25
result = Leiden.leiden(Symmetric(pot_conn), resolution = γ)
@show(unique(result[2]))
=#

#=
function scoped_fix(ccu,Lx,scale)
    v_old=1
    K = length(ccu)
    cum_array = Array{Array{UInt32}}(undef,K)
    for i in 1:K
        cum_array[i] = Array{UInt32}([])
    end
    for (k,v) in pairs(ccu)
        push!(cum_array,collect(v_old:v+v_old)[:])
        v_old=v+v_old
    end
    start = 1
    for (ind_,val) in enumerate(cum_array)
        Lx[val] .= ind_ 
    end
end
=#

#Y = sgtsnepi(pot_conn; d=dim, Y0 = Y0, max_iter = 500);


#- `lwd_in=0.5`: line width for internal edges
#- `lwd_out=0.3`: line width for external edges
#- `edge_alpha=0.2`: the alpha channel for the edges
#- `clr_in=nothing`: set color for all intra-cluster edges (if nothing, color by `cmap`)
#- `clr_out=colorant"#aabbbbbb"`: the color of inter-cluster edges
#figure=Plots.plot()
#display(SGtSNEpi.show_embedding( Y, Lx ,clr_out=cmap,clr_in=cmap,edge_alpha=0.5,lwd_in=0.5,lwd_out=0.3))#; A = pot_conn, res = (5000, 5000) )
#savefig("Potjans_SG_EMB.png")
#A = pot_conn
#Y0 = 0.01 * randn( size(A,1), 3 );
#Y = sgtsnepi(A; d = 3, Y0 = Y0, max_iter = 500);
#neighbor_recall(pot_conn, Y)
#sc = scatter( Y[:,1], Y[:,2], Y[:,3], color = L, colormap = cmap, markersize = 5)
#scene.center = false

#save("potjans_static_wiring_network_embedding.png")
#show_embedding( Y, L ; A = - Deliver to person
#Y0 = 0.01 * randn( size(pot_conn[1],1), dim);
#pot_conn = abs.(pot_conn[1])

#=

cmap = distinguishable_colors(
           maximum(L) - minimum(L) + 1,
           [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

Y = sgtsnepi(pot_conn; d=dim, Y0 = Y0, max_iter = 600);
#@show(length(Y))
scene = show_embedding( Y, L )#; A = pot_conn, res = (5000, 5000) )
scene.center = false
save("potjans_static_wiring_network_embedding.png")
scene = show_embedding( Y, L ; A = pot_conn, res = (5000, 5000) )
#scene.center = false
#savefig("potjans_st
#atic_wiring_network_embedding_wires.png")
save("potjans_static_wiring_network_embedding_wires.png")
=#
#=
L = zeros(size(pot_conn[1])[1])
@show(sizeof(L))
labelsx = [ind for (ind,row) in enumerate(eachrow(pot_conn[2])) if sum(row)!=0 ] 
L[labelsx] .= 1
labelsy = [ind for (ind,row) in enumerate(eachrow(pot_conn[3])) if sum(row)!=0 ] 
L[labelsy] .= 2

labelsz = [ind for (ind,row) in enumerate(eachrow(pot_conn[4])) if sum(row)!=0 ] 
L[labelsz] .= 3

labelsa = [ind for (ind,row) in enumerate(eachrow(pot_conn[5])) if sum(row)!=0 ] 
L[labelsa] .= 4

=#
#L = [1+sign(StatsBase.mean(row)) for row in eachrow(pot_conn)]


#using Colors

#Random.seed!(0);
#display(plot(sc))
#record(sc, "sgtsnepi-animation.gif", range(0, 1, length = 24*8); framerate = 24) do ang
#  rotate_cam!( sc.figure.scene.children[1], 2*π/(24*8), 0, 0 )
#end
