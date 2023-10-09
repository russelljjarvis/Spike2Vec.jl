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
resolution = 300

if !isfile("jesus_data_set.jld")
    (nodes,times,whole_duration,spikes_ragged,numb_neurons)  = load_datasets_calcium_jesus()
    @show(nodes)
    @save "jesus_data_set.jld" nodes times whole_duration spikes_ragged numb_neurons

else
    @load "jesus_data_set.jld" nodes times whole_duration spikes_ragged numb_neurons
end
#display(Plots.scatter(times,nodes))

if !isfile("jesus_int.jld")
    #end
    maxt = maximum(times)

    #@time div_spike_mat = spike_matrix_divided(spikes,resolution,numb_neurons,maxt;displace=true)
    @time div_spike_mat_no_displacement,start_windows,end_windows = spike_matrix_divided(spikes_ragged,resolution,numb_neurons,maxt;displace=false)
    
    @save "jesus_int.jld" div_spike_mat_no_displacement start_windows end_windows
else 
    @load "jesus_int.jld" div_spike_mat_no_displacement start_windows end_windows
end
ε=25.7
#@time time_windows = Vector{Any}([Tuple(s,e) for (s,e) in zip(start_windows,end_windows)])

#@show(div_spike_mat_no_displacement)
#if !isfile("jesus_int_processed2.jld")
    distmat = compute_metrics_on_matrix_divisions(div_spike_mat_no_displacement,metric="hybrid")
    #@show(variance)
    #Plots.heatmap(distmat)
    #savefig("Blah.png")
    #p1=Plots.histogram(mat2vec_hybrid)
    distmat = compute_metrics_on_matrix_divisions(div_spike_mat_no_displacement,metric="kreuz")

    #@show(variance)
    #p2=Plots.histogram!(p1,mat2vec_kreuz)
    distmat = compute_metrics_on_matrix_divisions(div_spike_mat_no_displacement,metric="LV")#,label="LV")

    #@show(variance)
    #p3=Plots.histogram!(p2,mat2vec_LV)

    #distmat = compute_metrics_on_matrix_divisions(div_spike_mat_no_displacement,metric="count")#,label="LV")
    #@show(variance)
    
    #distmat = compute_metrics_on_matrix_self_past_divisions(div_spike_mat_no_displacement)
    #Plots.heatmap(distmat)
    #savefig("relative_to_self_Distmat_sqaure.png")

    #compute_metrics_on_matrix_self_past_divisions!(div_spike_mat_no_displacement,metric="count")
    #Plots.heatmap(distmat)

    #p4=Plots.histogram!(p3,mat2vec_sum)

    #Plots.plot(p4)
    #savefig("variance_of.png")

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
    assing_progressions,assing_progressions_times,assing_progressions_time_indexs = get_state_transitions(start_windows,end_windows,sqr_distmat,assign;threshold= ε)
    for i in 1:length(assing_progressions) 
        if assing_progressions[i]==1
        assing_progressions[i]=-1
        end
    end
    max_time = maximum(times)

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
    p1 = Plots.scatter(times,nodes)
        #only_one_neuron_spike_times = mat_of_spikes[neuron_id,:]
        #nodes = [Int32(neuron_id) for (_,_) in enumerate(only_one_neuron_spike_times)]
        #display(Plots.scatter!(p1,only_one_neuron_spike_times,nodes,legend = false,xlabel="time (Seconds)",ylabel="Cell Id"))
    
    #p3 = Plots.scatter(nodes,times,markersize = 1.1,markerstrokewidth=0,alpha=0.8)
    #p4 = Plots.scatter(assing_progressions_times,end_windows,markersize = 1.1,markerstrokewidth=0,alpha=0.8)
    layout = @layout [a ; b ;c]
    Plots.plot(p2, p3, p1, layout=layout,legend=false)
    #https://github.com/open-risk/transitionMatrix
    savefig("state_transition_trajectory.png")
    
#@show(assing_progressions[assing_progressions==3])

    @show(mode(assing_progressions))
    #assing_progressions,assing_progressions_times = get_state_transitions(start_windows,end_windows,sqr_distmat,assign;threshold= ε)
    #repeated_windows = state_transition_trajectory(start_windows,end_windows,sqr_distmat,assign,assing_progressions,assing_progressions_times;plot=true,file_name="long_duration.png")
    #assign[unique(i -> assign[i], 1:length(assign))].=0.0
    @save "jesus_int_processed2.jld" assign
    # repeated_windows assing_progressions assing_progressions_times distmat sqr_distmat

#else

#end
@load "jesus_int_processed2.jld" assign #repeated_windows assing_progressions assing_progressions_times distmat sqr_distmat

#labels2cols = internal_validation1(assign,div_spike_mat_no_displacement);
assign = Vector{UInt32}(assign)
#@save "zebra_finche.jld" assign
#@load "zebra_finche.jld" assign #repeated_windows assing_progressions assing_progressions_times distmat sqr_distmat
#spike_motif_dict_both = 
#div_spike_mat_no_displacement,_,_ = spike_matrix_divided(spikes_ragged,resolution,numb_neurons,max_time;displace=true)

internal_validation_dict(assign,div_spike_mat_no_displacement;file_path=projectdir())

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
