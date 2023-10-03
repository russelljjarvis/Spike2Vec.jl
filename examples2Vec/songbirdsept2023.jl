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
using Infliltrator

@show(datadir())
@show(projectdir())
# Songbird metadata
numb_neurons = 77
max_time = 22.2
# Randomly permute neuron labels.
# (This hides the sequences, to make things interesting.)
#_p = Random.randperm(num_neurons)

# Load spikes.
#spikes = seq.Spike[]

(nnn,ttt) = load_zebra_finche_nmc_dataset()
#display(Plots.scatter(ttt,nnn))
maxt = maximum(ttt)
@show(maxt)
resolution = 125
ε=20.7

spikes_ragged,numb_neurons = create_spikes_ragged(nnn,ttt)
@time div_spike_mat_no_displacement,start_windows,end_windows = spike_matrix_divided(spikes_ragged,resolution,numb_neurons,max_time;displace=false)
@time (distmat,variance,mat2vec_hybrid) = compute_metrics_on_matrix_divisions(div_spike_mat_no_displacement,metric="kreuz")
sqr_distmat = label_exhuastively_distmat(distmat;threshold=ε,disk=false)#,nclasses)
Plots.heatmap(sqr_distmat)
savefig("Distmat_sqaure.png")
(R,sort_idx,assign) = cluster_distmat(sqr_distmat)
assing_progressions,assing_progressions_times = get_state_transitions(start_windows,end_windows,sqr_distmat,assign;threshold= ε)
repeated_windows = state_transition_trajectory(start_windows,end_windows,sqr_distmat,assign,assing_progressions,assing_progressions_times;plot=true,file_name="long_duration.png")
assign[unique(i -> assign[i], 1:length(assign))].=0.0
assign = Vector{UInt32}(assign)
@save "zebra_finche.jld" assign
@load "zebra_finche.jld" assign #repeated_windows assing_progressions assing_progressions_times distmat sqr_distmat

#labels2cols = 
internal_validation1(assign,div_spike_mat_no_displacement)

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

