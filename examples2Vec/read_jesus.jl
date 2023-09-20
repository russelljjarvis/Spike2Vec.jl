using Plots
using MAT
using StatsBase
using JLD
using Plots
using SpikeTime
using DrWatson
using ProgressMeter
using OnlineStats
using SparseArrays
using OhMyREPL
using DelimitedFiles
using DataFrames
#if isfile("280_neurons.jld")
#    @load "280_neurons.jld" new_t new_n current_max_t
#else
(nodes,times,whole_duration,global_isis,spikes,numb_neurons) = load_datasets_calcium_jesus()
#end
maxt = maximum(times)
resolution = 200
#@time div_spike_mat = spike_matrix_divided(spikes,resolution,numb_neurons,maxt;displace=true)
@time div_spike_mat_no_displacement = spike_matrix_divided(spikes,resolution,numb_neurons,maxt;displace=false)

ε=17.7

if !isfile("jesus_int.jld")
    @time (distmat,variance) = compute_metrics_on_matrix_divisions(div_spike_mat_no_displacement,metric="LV")
    @save "jesus_int.jld" distmat variance
else 
    @load "jesus_int.jld" distmat variance
end
@time (distmat,tlist,nlist,start_windows,end_windows,spike_distance_size,variance) = compute_metrics_on_divisions(nodes,Vector{Float64}(times),resolution,numb_neurons,maxt,plot=false,metric="LV")
#Plots.heatmap(distmat)
#savefig("pre_Distmat_sqaure.png")
sqr_distmat = label_online_distmat(distmat;threshold=ε,disk=false)#,nclasses)
#Plots.heatmap(sqr_distmat)
#savefig("Distmat_sqaure.png")
R,sort_idx,horizonta_assign = horizontal_sort_into_tasks(sqr_distmat)
(R,sort_idx,assign) = cluster_distmat(sqr_distmat)
assing_progressions,assing_progressions_times = get_state_transitions(start_windows,end_windows,sqr_distmat,assign;threshold= ε)
repeated_windows = state_transition_trajectory(start_windows,end_windows,sqr_distmat,assign,assing_progressions,assing_progressions_times;plot=true,file_name="calcium.png")
assign[unique(i -> assign[i], 1:length(assign))].=0.0
plotss_1(assign,nlist,tlist)

list_of_correlations,list_of_heats = plotss_2(assign,div_spike_mat_no_displacement)
#@show(o)

#@show(unique(assign))
#@show(list_of_heats)
#=
nslices=length(start_windows)
get_repeated_scatter(nlist,tlist,start_windows,end_windows,repeated_windows,nodes,times,nslices,file_name="calcium.png")
get_division_scatter_identify(div_spike_mat,nlist,tlist,start_windows,end_windows,sqr_distmat,assign,nodes,times,repeated_windows,file_name="calcium.png";threshold= ε)
#get_division_scatter_identify2
get_division_scatter_identify2(div_spike_mat,nlist,tlist,start_windows,end_windows,sqr_distmat,assign,nodes,times,repeated_windows,file_name="calcium.png";threshold= ε)
#get_division_scatter_identify2
=#