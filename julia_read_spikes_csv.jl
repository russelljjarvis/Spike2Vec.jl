using PyCall
using SpikingNeuralNetworks
using ProgressMeter
using Revise
py"""
import feather
def read():
    df = feather.read_dataframe("~/git/brian2-sims/sim_data/ai_exp_latest/win2.5_bg60/spikes.feather")
    return df
"""

df = py"read"();
times = df.t
nodes = df.i
division_size = maximum(times)/10.0

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

get_plot(ttt, nnn, division_size[1])
nnn=[];
ttt=[];
for (i, t) in enumerate(times)
    for tt in t
        push!(nnn,i);push!(ttt,tt)
    end
end
