using Plots
using MAT
using StatsBase
using JLD
using Plots
using SpikeTime
using DrWatson
using ProgressMeter
#using RecurrenceAnalysis
#using Clustering
#import DelimitedFiles: readdlm
#using UMAP
using OnlineStats
using SparseArrays

"""
A method to re-represent dense boolean vectors as a two dense vectors of spikes, and times.
spikes is a matrix with regularly sampled windows, populated by spikes, with calcium spikes.
"""
function convert_bool_matrice_to_ts(read_spike_dense::Matrix{Bool}, frame_width::Real)
    nodes = UInt32[]
    times = Float32[]
    for (indy,row) in enumerate(eachrow(read_spike_dense))
        for (indx,x) in enumerate(row)
            if x
                push!(nodes,indy)
                push!(times,indx*frame_width)                
            end
        end
    end
    whole_duration = length(read_spike_dense[1,:])*frame_width
    (nodes::Vector{UInt32},times::Vector{Float32},whole_duration::Real)
end

"""
A method to get collect the Inter Spike Intervals (ISIs) per neuron, and then to collect them together to get the ISI distribution for the whole cell population
Also output a ragged array (Array of unequal length array) of spike trains. 
"""
function create_ISI_histogram(nodes::Vector{UInt32},times::Vector{Float32})
    spikes_ragged = []
    global_isis =Float32[] # the total lumped population ISI distribution.
    isi_s = []
    numb_neurons=Int(maximum(nodes))+1 # Julia doesn't index at 0.
    @inbounds for n in 1:numb_neurons
        push!(spikes_ragged,[])
    end
    @inbounds for i in 1:numb_neurons
        for (n,t) in zip(nodes,times)
            if i==n
                push!(spikes_ragged[i],t)
            end
        end
    end
    @inbounds for (i, times) in enumerate(spikes_ragged)
        push!(isi_s,[])
        for (ind,x) in enumerate(times)
            if ind>1
                isi_current = x-times[ind-1]
                push!(isi_s[i],isi_current)
            end
        end
        append!(global_isis,isi_s[i])
    end
    (global_isis:: Vector{Float32},spikes_ragged::Vector{Any},numb_neurons)
end

function load_datasets_calcium()
    """
    Of course the absolute paths below will need to be wrangled to match your directory tree.
    """
    read_spike_dense0 = matread("../datasets/M1_d1A_S.mat")["GC06_M1963_20191204_S1"]["Transients"]["Raster"]

    read_spike_dense1 = matread("../datasets/M1_d1A_V.mat")["GC06_M1963_20191204_V4"]["Transients"]["Raster"]
    read_spike_dense2 = matread("../datasets/M4_d46_S.mat")["GfO01_M1875_20200222_S1"]["Transients"]["Raster"]
    #@show(names(read_spike_dense1))
    FPS = matread("../datasets/M1_d1A_S.mat")["GC06_M1963_20191204_S1"]["Movie"]["FPS"]
    frame_width = 1.0/FPS #0.08099986230023408 #second, sample_rate =  12.3457#Hz


    (nodes,times,whole_duration) = convert_bool_matrice_to_ts(read_spike_dense0,frame_width)
    #=
    (nodes1,times1,whole_duration) = convert_bool_matrice_to_ts(read_spike_dense1,frame_width)
    (nodes2,times2,whole_duration) = convert_bool_matrice_to_ts(read_spike_dense2,frame_width)

    offset0 = maximum(times)
    offset1 = maximum(times1) + offset0

    append!(nodes,nodes1)
    append!(nodes,nodes2)

    times1 = offset0 .+ times1#[ofor t in times1]
    times2 = offset1 .+ times2#[ofor t in times1]

    append!(times,times1)
    append!(times,times2)
    =#

    global_isis,spikes_ragged,numb_neurons = create_ISI_histogram(nodes,times)
    (nodes,times,whole_duration,global_isis,spikes_ragged,numb_neurons)
end
(nodes,times,whole_duration,global_isis,spikes,numb_neurons) = load_datasets_calcium()
maxt = maximum(times)
"""
Visualize one epoch, as a spike train raster and then an ISI histogram.
"""
p1 = Plots.plot()
Plots.scatter!(p1,times,nodes,legend = false,markersize = 0.8,markerstrokewidth=0,alpha=0.8, bgcolor=:snow2, fontcolor=:blue,xlabel="time (Seconds)",ylabel="Cell Id")
#savefig("scatter_plot.png")
b_range = range(minimum(global_isis), mean(global_isis)+std(global_isis), length=21)
p2 = Plots.plot()
Plots.histogram!(p2,global_isis, bins=b_range, normalize=:pdf, color=:gray,xlim=[0.0,mean(global_isis)+std(global_isis)])
Plots.plot(p1,p2)
savefig("Spike_raster_and_ISI_bar_plot.png")
resolution = 130
ε=7.7
div_spike_mat=spike_matrix_divided(nodes,times,spikes,resolution,numb_neurons,maxt)
div_spike_mat_no_displacement=spike_matrix_divided_no_displacement(nodes,times,spikes,resolution,numb_neurons,maxt)

#(distmat,tlist,nlist,start_windows,end_windows,spike_distance_size) = sort_by_row(distmat,nodes,times,resolution,numb_neurons,maxt,spikes)

(distmat,tlist,nlist,start_windows,end_windows,spike_distance_size) = get_divisions(nodes,times,resolution,numb_neurons,maxt,plot=true,metric="CV")
(distmat,tlist,nlist,start_windows,end_windows,spike_distance_size) = get_divisions(nodes,times,resolution,numb_neurons,maxt,plot=false)


#(distmat,tlist,nlist,start_windows,end_windows,spike_distance_size) = get_divisions(nodes,times,resolution,numb_neurons,maxt,plot=true,metric="autocov")

#(distmat,tlist,nlist,start_windows,end_windows,spike_distance_size) = get_divisions(nodes,times,resolution,numb_neurons,maxt,plot=true,metric="LV")

#(distmat,tlist,nlist,start_windows,end_windows,spike_distance_size) = get_divisions(nodes,times,resolution,numb_neurons,maxt,plot=true,metric="kreuz")

#distmat = distmat[horizontalR_sort_idx,:]

#div_spike_mat = div_spike_mat[horizontalR_sort_idx,:]
Plots.heatmap(distmat)
savefig("pre_Distmat_sqaure.png")

sqr_distmat = label_online_distmat!(distmat;threshold=ε)#,nclasses)
Plots.heatmap(sqr_distmat)
savefig("Distmat_sqaure.png")
(R,sort_idx,assign) = cluster_distmat!(sqr_distmat)
#@show(assign)

#plot_umap(distmat;file_name="UMAP_calcium.png")

#rqa,xs, ys,sss,R = get_division_scatter_identify_via_recurrence_mat(sqr_distmat,assign,nlist,tlist,start_windows,end_windows,nodes,times,ε*2)
#@show(rqa)
#get_division_via_recurrence(R,xs, ys,sss,div_spike_mat,start_windows;file_name="recurrence.png")

assing_progressions,assing_progressions_times = get_state_transitions(start_windows,end_windows,sqr_distmat,assign;threshold= ε)
repeated_windows = state_transition_trajectory(start_windows,end_windows,sqr_distmat,assign,assing_progressions,assing_progressions_times;plot=true,file_name="calcium.png")
assign[unique(i -> assign[i], 1:length(assign))].=0.0
@show(assign)

function plotss_1(assign,nlist,tlist)

    p = Plots.plot()
    collect_isi_bags = []
    ##for (ind,a) in enumerate(assign)
     #   if a!=0
     #       push!(collect_isi_bags,[])
     #       push!(collect_isi_bags[a],[])

    #    end
    #end
    collect_isi_bags = []
    collect_isi_bags_map = []
    #@show(length(collect_isi_bags))
    p = Plots.plot()
    collect_isi_bags = []
    for (ind,a) in enumerate(assign)
        if a!=0
            Tx = tlist[ind]
            #@show(div_spike_mat[:])

            xlimits = maximum(Tx)
            Nx = nlist[ind]
            Plots.scatter!(p,Tx,Nx,legend = false, markercolor=a,markersize = 0.8,markerstrokewidth=0,alpha=0.8, bgcolor=:snow2, fontcolor=:blue, xlims=(0, xlimits))
            #@show(bag_of_isis(Nx,Tx))
            push!(collect_isi_bags,bag_of_isis(Nx,Tx))
            push!(collect_isi_bags_map,a)

            #temp = div_spike_mat[:,ind] #.+sw
            #@show(length(temp))
            #Plots.scatter!(p,temp,legend=false, markercolor=a)
        end
    end
    display(Plots.plot(p))# = Plots.plot()
    collect_isi_bags,collect_isi_bags_map
end
plotss_1(assign,nlist,tlist)
function plotss_2(assign,div_spike_mat_no_displacement)
    list_of_correlations = []
    list_of_heats = []

    for un in unique(assign)
        Nxag = Float32[]
        Txag = Float32[]
        for (ind,a) in enumerate(assign)
            if a==un
                p = Plots.plot()
                pscatter = Plots.plot()
                pbar = Plots.plot()
        
                Nx = Float32[]
                Tx = Float32[]
                for (indy,row) in enumerate(div_spike_mat_no_displacement[:,ind])
                    for (indx,x) in enumerate(row)
                        if length(x)!=0
                            append!(Nx,indy)
                            append!(Tx,x)                
                        end
                    end
                end
                append!(Nxag,Nx)
                append!(Txag,Tx)
                if length(Tx)!=0
                    Plots.scatter!(pscatter,Tx,Nx,legend = false, markercolor=ind,markersize = 1.8,markerstrokewidth=0,alpha=0.8, bgcolor=:snow2, fontcolor=:blue, xlims=(minimum(Tx),maximum(Tx)))
                    savefig("scatter_match_$un.png")
                end

            end
        end
        o = OnlineStats.HeatMap(minimum(Nxag):maximum(Nxag)/30:maximum(Nxag), minimum(Txag):maximum(Txag)/30:maximum(Txag))        
        fit!(o, zip(Nxag, Txag))
        display(plot(o, marginals=false, legend=true))
        savefig("heatmap_$un.png")

        push!(list_of_heats,length(sparse(o.counts).nzval))
        push!(list_of_correlations,sum(abs.(cor(o.counts))))
    end
    list_of_correlations,list_of_heats
end
list_of_correlations,list_of_heats   = plotss_2(assign,div_spike_mat_no_displacement)
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