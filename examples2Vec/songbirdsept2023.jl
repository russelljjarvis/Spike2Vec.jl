using DrWatson
using Plots
using SpikeTime
using Revise
using StatsBase
using ProgressMeter
using LinearAlgebra
using JLD
using ProgressMeter
using Plots
using Plots
import DelimitedFiles: readdlm
using DrWatson
using Unitful
#using StatsBase

#using Infliltrator
numb_neurons = 77
max_time = 22.2

(nnn,ttt) = load_zebra_finche_nmc_dataset()
#display(Plots.scatter(ttt,nnn))
maxt = maximum(ttt)
#@show(maxt)
resolution = 100
ε=20.7

spikes_ragged,numb_neurons = create_spikes_ragged(nnn,ttt)
div_spike_mat,start_windows,end_windows = spike_matrix_divided(spikes_ragged,resolution,numb_neurons,max_time;displace=false)
window_length = end_windows[1]-start_windows[1]
window_length_unit_time = maxt/window_length
@show(window_length_unit_time)
#@show(window_lengths)
#window_lengths2 = end_windows[2]-start_windows[2]
#@show(window_lengths2)


distmat = compute_metrics_on_matrix_divisions(div_spike_mat,metric="LV")
@show(variance)
Plots.heatmap(distmat)
savefig("LV_Distmat.png")

distmat = compute_metrics_on_matrix_divisions(div_spike_mat,metric="count")
@show(variance)
Plots.heatmap(distmat)
savefig("count_Distmat.png")
distmat = compute_metrics_on_matrix_divisions(div_spike_mat,metric="kreuz")
@show(variance)
Plots.heatmap(distmat)
savefig("kreuz_Distmat.png")

sqr_distmat = label_exhuastively_distmat(distmat;threshold=ε,disk=false)
Plots.heatmap(distmat)
savefig("Distmat_sqaure.png")
(R,sort_idx,assign) = cluster_distmat(sqr_distmat)
assing_progressions,assing_progressions_times,assing_progressions_time_indexs = get_state_transitions(start_windows,end_windows,sqr_distmat,assign;threshold= ε)
for i in 1:length(assing_progressions) 
    if assing_progressions[i]==1
       assing_progressions[i]=-1
    end
end
#@show(assing_progressions[assing_progressions==3])

#@show(mode(assing_progressions))
#@show(times)
#assing_progressions_times= [t+window_length_unit_time for t in assing_progressions_times]
#p1 = Plots.plot()
div_spike_mat,start_windows,end_windows = spike_matrix_divided(spikes_ragged,resolution,numb_neurons,max_time;displace=true)

p2 = Plots.scatter(assing_progressions_times,assing_progressions,markercolor=assing_progressions,legend=false,markersize = 1.1,markerstrokewidth=0,alpha=0.8)
p3 = Plots.scatter()
for (ti,category) in zip(assing_progressions_time_indexs,assing_progressions)
    if 3==category
        (nodes,times)=return_spike_item_from_matrix(div_spike_mat,ti)
        times=times.+assing_progressions_times[ti]
    #@show(nodes,times)
        Plots.scatter!(p3,times,nodes,markersize = 1.1,markerstrokewidth=0,alpha=0.8)
    end
end
savefig("what_was_thing.png")

    #only_one_neuron_spike_times = mat_of_spikes[neuron_id,:]
    #nodes = [Int32(neuron_id) for (_,_) in enumerate(only_one_neuron_spike_times)]
    #display(Plots.scatter!(p1,only_one_neuron_spike_times,nodes,legend = false,xlabel="time (Seconds)",ylabel="Cell Id"))

#p3 = Plots.scatter(nodes,times,markersize = 1.1,markerstrokewidth=0,alpha=0.8)
#p4 = Plots.scatter(assing_progressions_times,end_windows,markersize = 1.1,markerstrokewidth=0,alpha=0.8)
layout = @layout [a ; b ]
Plots.plot(p2, p3, layout=layout,legend=false)
#https://github.com/open-risk/transitionMatrix
savefig("state_transition_trajectory.png")

#repeated_windows = state_transition_trajectory(start_windows,end_windows,sqr_distmat,assign,assing_progressions,assing_progressions_times;plot=true,file_name="")

#assign[unique(i -> assign[i], 1:length(assign))].=0.0
assign = Vector{UInt32}(assign)
#@save "zebra_finche.jld" assign
#@load "zebra_finche.jld" assign #repeated_windows assing_progressions assing_progressions_times distmat sqr_distmat
#spike_motif_dict_both = 
div_spike_mat_no_displacement,_,_ = spike_matrix_divided(spikes_ragged,resolution,numb_neurons,max_time;displace=true)

internal_validation_dict(assign,div_spike_mat_no_displacement;file_path=projectdir())

#@show(spike_motif_dict_both)
#labels2cols = 
#internal_validation1(assign,div_spike_mat_no_displacement;file_path=projectdir())

#mat_of_distances = get_plot(ttt,nnn,resolution)
#plot_umap(mat_of_distances;file_name="UMAP_song_bird.png")
#nclasses=10
#(scatter_indexs,yes,sort_idx) = label_online(mat_of_distances,nclasses)
#display(mat_of_distances)
#@show(mat_of_distances)
#using CategoricalArrays
#distmat = label_online_distmat(mat_of_distances)#,nclasses)
#display(distmat)
#(R,sort_idx,assign) = cluster_distmat(distmat)
#display(Plots.heatmap(distmat))

