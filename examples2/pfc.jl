using JLD
using Plots
using SpikeTime
using DrWatson
using ProgressMeter
using RecurrenceAnalysis
using Clustering
import DelimitedFiles: readdlm
function load_datasets_pfc()
    spikes = []
    file_read_list =  readdlm("../data2/150628_SpikeData.dat", '\t', Float64, '\n')
    nodes = [n for (t, n) in eachrow(file_read_list)]
    numb_neurons=Int(maximum(nodes))+1
    @inbounds for (t, n) in eachrow(file_read_list)
        if length(t)!=0
            push!(spikes,[])
            #@show(spikes[Int32(n)])

        end
    end
    @inbounds for (t, n) in eachrow(file_read_list)
        if length(t)!=0

            push!(spikes[Int32(n)],t)
            #@show(spikes[Int32(n)])
        end
    end
    nnn_scatter=Vector{UInt32}([])
    ttt_scatter=Vector{Float32}([])
    @inbounds @showprogress for (i, t) in enumerate(spikes)
        @inbounds for tt in t
            if length(t)!=0
                push!(nnn_scatter,i)
                push!(ttt_scatter,Float32(tt))
            end
        end
    end
    maxt = (maximum(ttt_scatter))    
    #@show(spikes)
    (nnn_scatter,ttt_scatter,spikes,numb_neurons,maxt)
    
end
(nodes,times,spikes,numb_neurons,maxt)= load_datasets_pfc()
resolution = 65
ε=6
div_spike_mat=spike_matrix_divided(nodes,times,spikes,resolution,numb_neurons,maxt)

(distmat,tlist,nlist,start_windows,end_windows,spike_distance_size) = get_divisions(nodes,times,resolution,numb_neurons,maxt,plot=false)

function sort_by_row(distmat,nodes,times,resolution,numb_neurons,maxt,spikes)
    horizontalR = kmeans(distmat,3)
    horizontalR_sort_idx =  sortperm(assignments(horizontalR))
    spikes = spikes[horizontalR_sort_idx]
    nodes=Vector{UInt32}([])
    times=Vector{Float32}([])

    @inbounds @showprogress for (i, t) in enumerate(spikes)
        @inbounds for tt in t
            if length(t)!=0
                push!(nodes,i)
                push!(times,Float32(tt))
            end
        end
    end
    (distmat,tlist,nlist,start_windows,end_windows,spike_distance_size) = get_divisions(nodes,times,resolution,numb_neurons,maxt,plot=false)
end
(distmat,tlist,nlist,start_windows,end_windows,spike_distance_size) = sort_by_row(distmat,nodes,times,resolution,numb_neurons,maxt,spikes)
distmat = distmat[horizontalR_sort_idx,:]
div_spike_mat = div_spike_mat[horizontalR_sort_idx,:]
sqr_distmat = label_online_distmat!(distmat)#,nclasses)

(R,sort_idx,assign) = cluster_distmat!(sqr_distmat)


#rqa,xs, ys,sss,R = get_division_scatter_identify_via_recurrence_mat(sqr_distmat,assign,nlist,tlist,start_windows,end_windows,nodes,times,ε*2)
#@show(rqa)
#get_division_via_recurrence(R,xs, ys,sss,div_spike_mat,start_windows;file_name="recurrence.png")

assing_progressions,assing_progressions_times = get_state_transitions(start_windows,end_windows,sqr_distmat,assign;threshold= ε)
repeated_windows = state_transition_trajectory(start_windows,end_windows,sqr_distmat,assign,assing_progressions,assing_progressions_times;plot=true,file_name="pfcpfc.png")
nslices=length(start_windows)
get_repeated_scatter(nlist,tlist,start_windows,end_windows,repeated_windows,nodes,times,nslices,file_name="pfcpfc.png")
get_division_scatter_identify(div_spike_mat,nlist,tlist,start_windows,end_windows,sqr_distmat,assign,nodes,times,repeated_windows,file_name="pfcpfc.png";threshold= ε)
#get_division_scatter_identify2
get_division_scatter_identify2(div_spike_mat,nlist,tlist,start_windows,end_windows,sqr_distmat,assign,nodes,times,repeated_windows,file_name="pfcpfc.png";threshold= ε)
#get_division_scatter_identify2