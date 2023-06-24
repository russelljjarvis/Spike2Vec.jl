
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
   

    pot_conn,cell_index_to_layer = potjans_layer(scale)

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

        display(SGtSNEpi.show_embedding( Y0, Lx ,A=pot_conn;edge_alpha=0.05,lwd_in=0.05,lwd_out=0.013,clr_out=cmap_out))
        save("SGtSNEpi_connection.png")
    end
    pot_conn,cell_index_to_layer
end


if isfile("potjans_wiring.jld")
    #
    # TODO delete following  two lines, just grabbing a plot!
    #scale = 0.07225
    #pot_conn = grab_connectome(scale)

    @load "potjans_wiring.jld" pot_conn ragged_array_weights cell_index_to_layer

else
    scale = 0.05225
    pot_conn,cell_index_to_layer = grab_connectome(scale)
    #pot_conn *.100
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
    @save "potjans_wiring.jld" pot_conn ragged_array_weights cell_index_to_layer


end






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
