using SpikingNeuralNetworks
using ProgressMeter
using Revise
ENV["PYTHON"]="/home/rjjarvis/anaconda3/bin/python"
#using Pkg
#Pkg.build("PyCall")
using PyCall
using JLD

pyimport("feather")
println("no read in data")

py"""
import feather
#import pandas as pd
def read():
    df = feather.read_dataframe("spikes.feather")
    return df
"""

df = py"read"();
println("read in data")
function ragged_to_uniform_col_vec(df)
    times = df.t
    nodes = df.i
    
    nnn=[];
    ttt=[];
    @showprogress for (i, t) in enumerate(times)
        for tt in t
            push!(nnn,i);push!(ttt,tt)
        end
    end
    (nnn,ttt)
end

(nnn,ttt) = ragged_to_uniform_col_vec(df)
@save "pablos_spikes.jld" nnn ttt
println("gets here")
division_size = maximum(ttt)/10.0

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
    @showprogress for (ind,toi) in enumerate(end_window)
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
    f = Figure()
    Axis(f[1, 1], title = "State visualization",)#yticks = ((1:length(mat_of_distances)) ,String([Char(i) for i in collect(1:length(mat_of_distances))])))
    @showprogress for (ind,_) in enumerate(eachrow(mat_of_distances))
        d = Makie.density!(mat_of_distances[ind,:],offset=ind*2,color = :x, colormap = :thermal, colorrange = (-10, 10),strokewidth = 1, strokecolor = :black, bandwidth = 0.02)
    end
    save("ridgeline.png",f)
    return mat_of_distances,f
end

get_plot(ttt, nnn, division_size[1])
