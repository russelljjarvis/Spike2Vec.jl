#using
using DrWatson
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
using PyCall
using LinearAlgebra
using Ridgeline
using Makie
using KernelDensity,Distributions
using JLD
using CairoMakie#,KernelDensity, Distributions
#@quickactivate "spike2vec"
#@load "all_mnmist.jld" storage

initialize_project("spike2vec"; authors="Dr Russell Jarvis", force=true)

#=
try
    py"""
    import tonic;
    import os;
    tonic.datasets.SMNIST(os.getcwd(),train=False,num_neurons=999,dt=1.0)
    """
catch

end
=#

# Spike2vec and SpikeThroughPut.jl

#Lets say we have a `simulate` function that we came up with through
#experimentation in the `_research` folder.

function protect_variable()
    scale = 1.0
    (pot_conn,x,y,ccu) = build_neurons_connections(scale)
    Lx = Vector{Int64}(zeros(size(pot_conn)))
    return E,I
include("../src/models/genPotjansWiring.jl")
#potjans_weights(args)

#=
function potjans_weights(args)
    Ncells, g_strengths, ccu, scale = args
    (cumvalues,_,_,conn_probs,syn_pol) = potjans_params(ccu,scale)        
    Lxx = spzeros(Float32, (Ncells, Ncells))
    (jee,_,jei,_) = g_strengths 
    wig = Float32(-20*4.5) 
    Lxx = spzeros(Float32, (Ncells, Ncells))
    #Lie = spzeros(Float32, (Ncells, Ncells))
    #Lei = spzeros(Float32, (Ncells, Ncells))
    #Lii = spzeros(Float32, (Ncells, Ncells))
    return build_neurons_connections!(jee,jei,wig,Lxx,cumvalues,conn_probs,UInt32(Ncells),syn_pol,g_strengths)
end;
=#

#pot_conn,x,y,ccu,scale,Lx = protect_variable()

function hide_scope(config)
    @unpack pop_size,sim_duration = config
    E,I = protect_variable()


    #EE = SNN.SpikingSynapse(E, E,sim_type; σ = 160*0.27/1, p = 0.025)
    #EI = SNN.SpikingSynapse(E, I,sim_type; σ = 160*0.27/1, p = 0.055)
    #IE = SNN.SpikingSynapse(I, E,sim_type; σ = -160*0.27/1, p = 0.250)
    #II = SNN.SpikingSynapse(I, I,sim_type; σ = -160*0.27/1, p = 0.15)
    #P = [I, E]
    #C = [EE, EI, IE, II]

    ##
    # ToDO make a real interface that uses block arrays.
    ## 
    #SNN.monitor([C], [:g])
    #SNN.monitor([E, I], [:fire])
    #inh_connection_map=[(E,EE,1,E),(E,EI,1,I)]
    #exc_connection_map=[(I,IE,-1,E),(I,II,-1,I)]
    #connection_map = [exc_connection_map,inh_connection_map]
    SNN.sim!(P, duration = sim_duration)

    (times,nodes) = SNN.get_trains([E,I])
    (times,nodes,E,I)
end
#tern_it(x,y)= [x + 0.5 * y,sqrt(0.75) *  y]' # convert proportions to cartisian coords on a triangle
#closest(a,b) = findmin(abs.(a .-b))[2]
#set_dens(a,b,k) = k.density[closest(a,collect(k.x)),closest(b,collect(k.y))]

function get_plot(times,nodes,division_size)
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
    #=
    fig = nothing
    for (ind,_) in enumerate(eachrow(mat_of_distances))
        n = length(mat_of_distances[1,:])
        θ = LinRange(0, 2pi, n)
        if ind==1
            fig = Plots.plot(θ, mat_of_distances[ind,:], proj=:polar,color=cs1[ind])#, layout = length(mat_of_distances))
        else 
            Plots.plot!(fig,θ,mat_of_distances[ind,:], proj=:polar,color=cs1[ind])  |>display
        end

    end
    =#
    #nb %% A slide [code] {"slideshow": {"slide_type": "subslide"}}
    #display(fig)

    f = Figure()
    Axis(f[1, 1], title = "State visualization",)#yticks = ((1:length(mat_of_distances)) ,String([Char(i) for i in collect(1:length(mat_of_distances))])))

    for (ind,_) in enumerate(eachrow(mat_of_distances))
        #d = kde(mat_of_distances[ind,:])

        #d = density!(randn(200) .- 2sin((i+3)/6*pi), offset = i / 4,
        d = Makie.density!(mat_of_distances[ind,:],offset=ind*2,color = :x, colormap = :thermal, colorrange = (-10, 10),strokewidth = 1, strokecolor = :black, bandwidth = 0.02)
        # this helps with layering in GLMakie
        #translate!(d, 0, 0, -0.1i)
    end
    #f

    #ternKDE = kde(mat_of_distances)
    #display(f)
    #current_figure()
    save("ridgeline.png",f)
    #println("gets here")
    #savefig("vectors_wrapped.png")
    return mat_of_distances,f
end

#angles0,distances0,angles1,distances1 = post_proc_viz(mat_of_distances)




# Store it in the data folder. `safesave` makes sure that nothing is overwritten.

# Do some of the above automatically with
function run_simulation(config)
    #@unpack pop_size,sim_duration,division_size = config
    
    #pyimport("tonic")
    println("trying to simulate neurons")
    (times,nodes,E,I) = hide_scope(config)
    println("succeeded to simulate neurons")

    @unpack division_size = config
    @time (mat_of_distances,f) = get_plot(times,nodes,division_size)
    display(f)

    @time (angles1,distances1) = final_plots(mat_of_distances)
    name = savename(config)
    parse_savename(name)
    #result = simulate(n,w,opt)

    result = (mat_of_distances,fig,f)
    @dict(result)
    safesave(datadir("simulations", savename(config, "sim_results")), (angles1,distances1,fig,nodes,times))

    
end
division_size = 20
pop_size::UInt64=10000
sim_type = Vector{Float32}(zeros(1))
#pop_size::Int32=100
SNN.@load_units
sim_duration = 3.0second

config = @dict(pop_size,sim_duration,division_size)

produce_or_load(datadir("mysimulation"),
                config,
                run_simulation)


#![res](layers.png)
