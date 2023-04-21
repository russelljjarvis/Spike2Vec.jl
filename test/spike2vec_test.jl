using Plots
using SpikingNeuralNetworks
using OnlineStats
using SparseArrays
SNN.@load_units

using Test
using Revise
##
# Since Odesa is not an officially Julia registered package, running this example would require:
# using Pkg
# Pkg.add(url=https://github.com/russelljjarvis/Odesa.jl-1")
# ] resolve
# ] activate
# ] instantiate
##
using Odesa
using StatsBase
#using UMAP
using ProgressMeter

pop_size::Int32=100
sim_type = Vector{Float32}(zeros(1))
sim_duration = 1.0second
u1 = Float32[10.0*abs(4.0*rand()) for i in 0:1ms:sim_duration]

#ERROR: LoadError: MethodError: no method matching 
#SpikingNeuralNetworks.IFNF(::Int32, ::Vector{Float32}, ::Vector{Float32}, ::Vector{Float32}, ::Vector{Bool}, ::Vector{Float32}, ::Vector{Int32}, ::Dict{Any, Any}, ::Vector{Array{UInt64}})

E = SNN.IFNF(pop_size,sim_type)
I = SNN.IFNF(pop_size,sim_type)
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


function get_plot(times,nodes)
    #times,nodes = get_()
    division_size = 10
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
        #@show(self_distances)
    end
    cs1 = ColorScheme(distinguishable_colors(spike_distance_size, transform=protanopic))
    p=nothing
    for (ind,_) in enumerate(eachrow(mat_of_distances))
        temp = (mat_of_distances[ind,:].- mean(mat_of_distances[ind,:]))./std(mat_of_distances[ind,:])
        n = length(temp)
        θ = LinRange(0, 2pi, n)
        if ind==1
            p = plot(θ, temp, proj=:polar,color=cs1[ind])#, layout = length(mat_of_distances))
        else 
            plot!(p,θ,mat_of_distances[ind,:], proj=:polar,color=cs1[ind])  |>display
        end

    end
    return mat_of_distances
end

SNN.sim!(P, C;conn_map= connection_map, current_stim = u1, duration = sim_duration)

print("simulation done !")
(times,nodes) = SNN.get_trains([E,I])#,Gx,Gy])
mat_of_distances = get_plot(times,nodes)

display(SNN.raster([E,I]))
println("hodld up b")

#display(plot_umap(nodes,times))
println("hodld up c")
final_timesurf = get_ts(nodes,times);
@show(sum(final_timesurf))
display(Plots.heatmap(final_timesurf))

