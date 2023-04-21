using Plots
using SpikingNeuralNetworks
using OnlineStats
using SparseArrays
SNN.@load_units
using Test
using Revise
using StatsBase
using ProgressMeter
using ColorSchemes
import StatsBase.ZScoreTransform
using LinearAlgebra
using DrWatson
initialize_project("DrWatson Example"; authors="RJarvis", force=true)
#import StatsBase.fit!
#include("genPotjansWiring.jl")
#import SpikingNeuralNetworks.models.potjans_layer
#include("../models/genPotjansWiring.jl")

function protect_variable()
    scale = 1.0
    (pot_conn,x,y,ccu) = build_neurons_connections(scale)
    Lx = Vector{Int64}(zeros(size(pot_conn)))
    return pot_conn,x,y,ccu,scale,Lx
end

function potjans_weights(args)
    Ncells, g_strengths, ccu, scale = args
    (cumvalues,_,_,conn_probs,syn_pol) = potjans_params(ccu,scale)        
    Lxx = spzeros(Float32, (Ncells, Ncells))
    (jee,_,jei,_) = g_strengths 
    wig = Float32(-20*4.5) 
    return build_neurons_connections!(jee,jei,wig,Lxx,cumvalues,conn_probs,UInt32(Ncells),syn_pol,g_strengths)
end

#pot_conn,x,y,ccu,scale,Lx = protect_variable()

function hide_scope()
    pop_size::UInt64=10000
    sim_type = Vector{Float32}(zeros(1))
    #pop_size::Int32=100
    sim_duration = 3.0second
    u1 = Float32[10.0*abs(4.0*rand()) for i in 0:1ms:sim_duration]
    
    post_synaptic_targets = Array{Array{UInt64}}(undef,pop_size)
    for i in 1:pop_size
        post_synaptic_targets[i] = Array{UInt64}([])
    end

    E = SNN.IFNF(pop_size,sim_type,post_synaptic_targets)

    I = SNN.IFNF(pop_size,sim_type,post_synaptic_targets)


    EE = SNN.SpikingSynapse(E, E,sim_type; σ = 160*0.27/1, p = 0.025)
    EI = SNN.SpikingSynapse(E, I,sim_type; σ = 160*0.27/1, p = 0.055)
    IE = SNN.SpikingSynapse(I, E,sim_type; σ = -160*0.27/1, p = 0.250)
    II = SNN.SpikingSynapse(I, I,sim_type; σ = -160*0.27/1, p = 0.15)
    P = [I, E]
    C = [EE, EI, IE, II]

    ##
    # ToDO make a real interface that uses block arrays.
    ## 
    SNN.monitor([C], [:g])
    SNN.monitor([E, I], [:fire])
    inh_connection_map=[(E,EE,1,E),(E,EI,1,I)]
    exc_connection_map=[(I,IE,-1,E),(I,II,-1,I)]
    connection_map = [exc_connection_map,inh_connection_map]
    SNN.sim!(P, C;conn_map= connection_map, current_stim = u1, duration = sim_duration)

    print("simulation done !")
    (times,nodes) = SNN.get_trains([E,I])
    (times,nodes,E,I)
end
(times,nodes,E,I) = hide_scope()

function get_plot(times,nodes)
    division_size = 20
    step_size = maximum(times)/division_size
    end_window = collect(step_size:step_size:step_size*division_size)
    spike_distance_size = length(end_window)
    start_windows = collect(0:step_size:step_size*division_size-1)
    mat_of_distances = zeros(spike_distance_size,maximum(unique(nodes))+1)
    n0ref = divide_epoch(nodes,times,start_windows[3],end_window[3])
    segment_length = end_window[3] - start_windows[3]
    t0ref = surrogate_to_uniform(n0ref,segment_length)
    PP = []
    for (ind,toi) in enumerate(end_window)
        self_distances = Array{Float32}(zeros(maximum(nodes)+1))
        sw = start_windows[ind]
        neuron0 = divide_epoch(nodes,times,sw,toi)    
        self_distances = get_vector_coords(neuron0,t0ref,self_distances)
        mat_of_distances[ind,:] = self_distances
    end
    cs1 = ColorScheme(distinguishable_colors(spike_distance_size, transform=protanopic))
    p=nothing
    mat_of_distances ./ norm.(eachcol(mat_of_distances))'

    for (ind,_) in enumerate(eachcol(mat_of_distances))
        mat_of_distances[:,ind] = mat_of_distances[:,ind].- mean(mat_of_distances)./std(mat_of_distances)
    end
    for (ind,_) in enumerate(eachrow(mat_of_distances))
        n = length(mat_of_distances[1,:])
        θ = LinRange(0, 2pi, n)
        if ind==1
            p = plot(θ, mat_of_distances[ind,:], proj=:polar,color=cs1[ind])#, layout = length(mat_of_distances))
        else 
            plot!(p,θ,mat_of_distances[ind,:], proj=:polar,color=cs1[ind])  |>display
        end

    end
    savefig("vectors_wrapped.png")
    return mat_of_distances
end

@time mat_of_distances = get_plot(times,nodes)
@time (angles1,distances1) = final_plots(mat_of_distances)
#angles0,distances0,angles1,distances1 = post_proc_viz(mat_of_distances)
display(SNN.raster([E,I]))
savefig("raster.png")
#println("hodld up b")

#display(plot_umap(nodes,times))
#println("hodld up c")
#final_timesurf = get_ts(nodes,times);
#@show(sum(final_timesurf))
#display(Plots.heatmap(final_timesurf))

