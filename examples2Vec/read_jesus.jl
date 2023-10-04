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
if !isfile("jesus_data_set.jld")
    (nodes,times,whole_duration,spikes_ragged,numb_neurons)  = load_datasets_calcium_jesus()
#    @show(nodes)
    @save "jesus_data_set.jld" nodes times whole_duration spikes_ragged numb_neurons

else
    @load "jesus_data_set.jld" nodes times whole_duration spikes_ragged numb_neurons
end
#display(Plots.scatter(times,nodes))

#if !isfile("jesus_int.jld")
    #end
    maxt = maximum(times)
    resolution = 225

    #@time div_spike_mat = spike_matrix_divided(spikes,resolution,numb_neurons,maxt;displace=true)
    @time div_spike_mat_no_displacement,start_windows,end_windows = spike_matrix_divided(spikes_ragged,resolution,numb_neurons,maxt;displace=false)
    
    @save "jesus_int.jld" div_spike_mat_no_displacement start_windows end_windows
#else 
    @load "jesus_int.jld" div_spike_mat_no_displacement start_windows end_windows
#end
ε=20.7
#@time time_windows = Vector{Any}([Tuple(s,e) for (s,e) in zip(start_windows,end_windows)])

#@show(div_spike_mat_no_displacement)
#if !isfile("jesus_int_processed2.jld")

    @time (distmat,variance,mat2vec_hybrid) = compute_metrics_on_matrix_divisions(div_spike_mat_no_displacement,metric="hybrid")

    p1=Plots.histogram(mat2vec_hybrid)
    (distmat,variance,mat2vec_kreuz) = compute_metrics_on_matrix_divisions(div_spike_mat_no_displacement,metric="kreuz")

    p2=Plots.histogram!(p1,mat2vec_kreuz)
    (distmat,variance,mat2vec_LV) = compute_metrics_on_matrix_divisions(div_spike_mat_no_displacement,metric="LV")#,label="LV")

    p3=Plots.histogram!(p2,mat2vec_LV)

    (distmat,variance,mat2vec_sum) = compute_metrics_on_matrix_divisions(div_spike_mat_no_displacement,metric="count")#,label="LV")
    p4=Plots.histogram!(p3,mat2vec_sum)

    Plots.plot(p4)
    savefig("variance_of.png")

    #@show(variance)
    #@time (distmat,variance) = compute_metrics_on_matrix_divisions(div_spike_mat_no_displacement,metric="LV")
    #@show(variance)

    #@time (distmat,tlist,nlist,start_windows,end_windows,spike_distance_size,variance) = compute_metrics_on_divisions(nodes,Vector{Float64}(times),resolution,numb_neurons,maxt,plot=false,metric="LV")
    Plots.heatmap(distmat)
    savefig("pre_Distmat_sqaure.png")
    sqr_distmat = label_exhuastively_distmat(distmat;threshold=ε,disk=false)#,nclasses)
    Plots.heatmap(sqr_distmat)
    savefig("Distmat_sqaure.png")
    #R,sort_idx,horizonta_assign = horizontal_sort_into_tasks(sqr_distmat)
    (R,sort_idx,assign) = cluster_distmat(sqr_distmat)
    assing_progressions,assing_progressions_times = get_state_transitions(start_windows,end_windows,sqr_distmat,assign;threshold= ε)
    repeated_windows = state_transition_trajectory(start_windows,end_windows,sqr_distmat,assign,assing_progressions,assing_progressions_times;plot=true,file_name="long_duration.png")
    assign[unique(i -> assign[i], 1:length(assign))].=0.0
    @save "jesus_int_processed2.jld" assign
    # repeated_windows assing_progressions assing_progressions_times distmat sqr_distmat

#else
    @load "jesus_int_processed2.jld" assign #repeated_windows assing_progressions assing_progressions_times distmat sqr_distmat

#end
labels2cols = internal_validation1(assign,div_spike_mat_no_displacement);

#times = div_spike_mat_no_displacement[:,3]

#nodes3,times3=ragged_to_uniform(times)
#times = div_spike_mat_no_displacement[:,2]
#nodes2,times2=ragged_to_uniform(times)

#display(Plots.scatter(nodes3,times3))
#display(Plots.scatter(nodes2,times2))

#list_of_correlations,list_of_heats = 

#plotss_1(assign,div_spike_mat_no_displacement)
#=
function ragged_to_uniform1(times)
    n=Vector{UInt32}([])
    ttt=Vector{Float32}([])
    for (i, t) in enumerate(times)
        if length(t)!=0
            for tt in t
                push!(n,i);
                for t in tt 
                    push!(ttt,Float32(t)) 
                end
            end
        end
    end
    (n::Vector{UInt32},ttt::Vector{Float32})
end
@show(times)
=#
#@show(typeof(times))
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