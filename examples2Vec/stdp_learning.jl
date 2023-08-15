using CSV
using DataFrames
using Plots 
using OnlineStats
using Revise
using SpikeTime
using JLD
using ProgressMeter
df2=  CSV.read("output_spikes.csv",DataFrame)
# make nodes of type float.
nodes = df2.id
times = df2.time_ms
function getHeatMap(times,nodes)
    xy = zip(times,nodes)
    # set node and time dimensions:
    xargs = minimum(nodes):1:maximum(nodes);
    yargs = minimum(times):25:maximum(times)
    @time o = fit!(HeatMap(yargs,xargs),xy)
    @time plot(o,marginals=false)#,marginals=false)
    #display(plot(o))#,marginals=false))
    savefig("stdp_exp.png")
end
getHeatMap(times,nodes)
# deal with Julia being not base 0, like Python.
#times = times[1:125000]
#nodes = nodes[1:125000]
times = Vector{Float32}(times)

nodes = [UInt32(n+1) for n in nodes ]
nodes = Vector{UInt32}(nodes)


"""
A method to get collect the Inter Spike Intervals (ISIs) per neuron, and then to collect them together to get the ISI distribution for the whole cell population
Also output a ragged array (Array of unequal length array) of spike trains. 
"""
function create_ISI_histogram(nodes::Vector{UInt32}, times::Vector{Float32}) # Any
    spikes_ragged = []
    isis = Float32[] # the total lumped population ISI distribution.
    isi_s = []
    numb_neurons=Int(maximum(nodes))+1 # Julia doesn't index at 0.
    @inbounds for n in 1:numb_neurons
        push!(spikes_ragged,[])
    end
    @inbounds @showprogress for i in 1:numb_neurons
        for (n,t) in zip(nodes,times)
            if i==n
                push!(spikes_ragged[i],t)
            end
        end
    end
    @inbounds @showprogress for (i, times) in enumerate(spikes_ragged)
        push!(isi_s,[])
        for (ind,x) in enumerate(times)
            if ind>1
                isi_current = x-times[ind-1]
                push!(isi_s[i],isi_current)
            end
        end
        append!(isis,isi_s[i])
    end
    (isis:: Vector{Float32},spikes_ragged::Vector{Any},numb_neurons)
end
"""
Visualize one epoch, as a spike train raster and then an ISI histogram.
Note this takes a very long time to compute.
"""
function analyse_ISI_distribution(nodes,times)
    @time global_isis,spikes_ragged,numb_neurons = create_ISI_histogram(nodes,times)
    println("evaluated")
    @show(global_isis)

    @save "ISIdistribution" global_isis spikes_ragged numb_neurons
    p1 = Plots.plot()
    Plots.scatter!(p1,times,nodes,legend = false,markersize = 0.8,markerstrokewidth=0,alpha=0.8, bgcolor=:snow2, fontcolor=:blue,xlabel="time (Seconds)",ylabel="Cell Id")
    savefig("scatter_plot_exp.svg")

    #savefig("scatter_plot.png")
    b_range = range(minimum(global_isis), mean(global_isis)+std(global_isis), length=21)
    p2 = Plots.plot()
    Plots.histogram!(p2,global_isis, bins=b_range, normalize=:pdf, color=:gray,xlim=[0.0,mean(global_isis)+std(global_isis)])
    Plots.plot(p1,p2)
    savefig("Spike_raster_and_ISI_bar_plot.svg")
end
ε=7.1

resolution = 150
numb_neurons = Int64(maximum(nodes))
maxt = maximum(times)

(distmat,tlist,nlist,start_windows,end_windows,spike_distance_size) = get_divisions(nodes,times,resolution,numb_neurons,maxt,plot=false)
Plots.heatmap(distmat)
savefig("pre_Distmat_sqaure.png")

sqr_distmat = label_online_distmat!(distmat;threshold=ε)
Plots.heatmap(sqr_distmat)
savefig("Distmat_sqaure.png")
(R,sort_idx,assign) = cluster_distmat!(sqr_distmat)

assing_progressions,assing_progressions_times = get_state_transitions(start_windows,end_windows,sqr_distmat,assign;threshold= ε)
repeated_windows = state_transition_trajectory(start_windows,end_windows,sqr_distmat,assign,assing_progressions,assing_progressions_times;plot=true,file_name="pablo_xx.png")
assign[unique(i -> assign[i], 1:length(assign))].=0.0

function plotss_1(assign,nlist,tlist)

    p = Plots.plot()
    collect_isi_bags = []
    collect_isi_bags = []
    collect_isi_bags_map = []
    p = Plots.plot()
    collect_isi_bags = []
    @showprogress for (ind,a) in enumerate(assign)
        if a!=0
            Tx = tlist[ind]
            xlimits = maximum(Tx)
            Nx = nlist[ind]
            Plots.scatter!(p,Tx,Nx,legend = false, markercolor=a,markersize = 0.8,markerstrokewidth=0,alpha=0.8, bgcolor=:snow2, fontcolor=:blue, xlims=(0, xlimits))
            push!(collect_isi_bags,bag_of_isis(Nx,Tx))
            push!(collect_isi_bags_map,a)

        end
    end
    Plots.plot(p)
    savefig("repeating_states.png")
    collect_isi_bags,collect_isi_bags_map
end
plotss_1(assign,nlist,tlist)
p1 = Plots.plot()
Plots.scatter!(p1,times[1:15000],nodes[1:15000],legend = false,markersize = 0.8,markerstrokewidth=0,alpha=0.8, bgcolor=:snow2, fontcolor=:blue,xlabel="time (Seconds)",ylabel="Cell Id")
savefig("scatter_plot_exp2.png")

