using JLD
using Plots
using SpikeTime
using DrWatson
using ProgressMeter
using RecurrenceAnalysis
using GraphMakie, CairoMakie

import DelimitedFiles: readdlm
function load_datasets_pfc()
    spikes = []
    file_read_list =  readdlm("../data2/150628_SpikeData.dat", '\t', Float64, '\n')
    nodes = [n for (t, n) in eachrow(file_read_list)]
    numb_neurons=Int(maximum(nodes))+1
    @inbounds for _ in 1:numb_neurons
        push!(spikes,[])
    end
    @inbounds for (t, n) in eachrow(file_read_list)
        push!(spikes[Int32(n)],t)
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
    (nnn_scatter,ttt_scatter,spikes,numb_neurons,maxt)
    
end
(nodes,times,spikes,numb_neurons,maxt)= load_datasets_pfc()
resolution = 50 # 65
ε=5
div_spike_mat=spike_matrix_divided(nodes,times,spikes,resolution,numb_neurons,maxt)
#@show(div_spike_mat)

(mat_of_distances,tlist,nlist,start_windows,end_windows,spike_distance_size) = get_divisions(nodes,times,resolution,numb_neurons,maxt,plot=false)

plot_umap_of_dist_vect(mat_of_distances; file_name="umap_PFC.png")
distmat = label_online_distmat(mat_of_distances)#,nclasses)
(R,_,assign) = cluster_distmat(distmat)
#sss =  StateSpaceSet(hcat(mat_of_distances))
#R = RecurrenceMatrix(sss, ε; metric = Euclidean(), parallel=true)
#display(RecurrenceAnalysis.recurrenceplot(R; ascii = true))
#rqa(R)
#xs, ys = RecurrenceAnalysis.coordinates(R)# -> xs, ys
#network = RecurrenceAnalysis.SimpleGraph(R)
#display(graphplot(network))
                                       #function get_state_transitions(start_windows,end_windows,distmat,assign;threshold::Real=5)

assing_progressions,assing_progressions_times = get_state_transitions(start_windows,end_windows,distmat,assign;threshold= ε)

#assing_progressions,assing_progressions_times = get_state_transitions(start_windows,end_windows,distmat,assign;threshold=threshold)
repeated_windows = state_transition_trajectory(start_windows,end_windows,distmat,assign,assing_progressions,assing_progressions_times;plot=true,file_name="pfcpfc.png")
nslices=length(start_windows)
#@show(repeated_windows)
get_repeated_scatter(nlist,tlist,start_windows,end_windows,repeated_windows,nodes,times,nslices,file_name="pfcpfc.png")
get_division_scatter_identify(nlist,tlist,start_windows,end_windows,distmat,assign,nodes,times,repeated_windows,file_name="pfcpfc.png";threshold= ε)
rqa,xs, ys,sss = get_division_scatter_identify_via_recurrence_mat(mat_of_distances,assign,nlist,tlist,start_windows,end_windows,nodes,times;file_name="recurrence_pfc.png",ε=5)
#get_division_scatter_identify_via_recurrence_mat(nlist,tlist,start_windows,end_windows,nodes,times;file_name::String="empty.png",ε::Real=5)

#display(Plots.scatter(assing_progressions))

#=



using SparseArrays
import Graphs: DiGraph,add_edge!, inneighbors,outneighbors, nv, ne, weights, is_strongly_connected,strongly_connected_components,edges,degree_histogram
function plot_stn(stn;filename="stn.pdf",nodesize=1,nodefillc="orange",linetype="straight",max_edgelinewidth=1,nodelabels=false)
    g = SimpleWeightedDiGraph(stn)

    nr_vertices = nv(g)
	x = []
	y = []
		
	#for i in 1:nr_vertices
	#	ilabel = label_for(stn,i)
	#	push!(x,stn[ilabel][:x])
#		push!(y,stn[ilabel][:y])
	#end
	#x = Int32.(x)
	#y = Int32.(y)
	
	w = sparse(stn)
	
	gp = gplot(g,edgelinewidth = reduce(vcat, [w[i,:].nzval for i in 1:nv(g)]) .^ 0.33,
		nodelabel = if nodelabels 1:nv(g) else nothing end,
		nodesize=nodesize,
		nodefillc=nodefillc,
		linetype=linetype,
		EDGELINEWIDTH=max_edgelinewidth)
	
	draw(PDF(filename),gp)

end
#stn = sparse(empty)
#plot_stn(empty)

#m = [0.8 0.2; 0.1 0.9]

=#
#using GLMakie, SGtSNEpi#, SNAPDatasets

#GLMakie.activate!()

#g = loadsnap(:as_caida)
#y = sgtsnepi(empty);
#show_embedding(y;
#  A = adjacency_matrix(empty),        # show edges on embedding
#  mrk_size = 1,                   # control node sizes
#  lwd_in = 0.01, lwd_out = 0.001, # control edge widths
#  edge_alpha = 0.03 )             # control edge transparency

#g = SimpleWeightedDiGraph(m)

#edge_label = Dict((i,j) => string(m[i,j]) for i in 1:size(m, 1), j in 1:size(m, 2))

#graphplot(g; names = 1:length(m), edge_label)

##
# 55 chanels and 25 seconds
##
#=
function get_division_scatter(times,nodes,division_size,yes,)
    #yes = yes[]
    step_size = maximum(times)/division_size
    end_windows = collect(step_size:step_size:step_size*division_size)
    spike_distance_size = length(end_windows)
    #start_windows = collect(0:step_size:step_size*division_size-1)
    start_windows = collect(0:step_size:(step_size*division_size)-step_size)

    mat_of_distances = zeros(spike_distance_size,maximum(unique(nodes))+1)
    segment_length = end_windows[3] - start_windows[3]
    p=Plots.scatter()
    #p=Plots.scatter(times,nodes,legend=true)
    end_windows = end_windows[]
    start_windows = start_windows[]

    @showprogress for (ind,toi) in enumerate(end_windows)
        sw = start_windows[ind]
        spikes = divide_epoch(nodes,times,sw,toi)    
        Nx=Vector{Int32}([])
        Tx=Vector{Float32}([])
        for (i, t) in enumerate(spikes)
            for tt in t
                if length(t)!=0
                    push!(Nx,i)
                    ###
                    ##
                    ##
                    #push!(Tx,Float32(tt))
                    ##
                    ###
                    ##
                    push!(Tx,Float32(sw+tt))
                    ##
                end
            end
            if ind in yes
                Plots.scatter!(p,Tx,Nx,marker_z=yes[ind],legend = false, markersize = 2,markerstrokewidth=0,alpha=1.0)
            end
            #@show(last(sw))

        end
        #if ind in yes
        #    Plots.vline!(p,[first(sw),0,50],markersize = 0.5,markerstrokewidth=0.5,alpha=0.5)

        #    Plots.vline!(p,[last(sw),0,50],markersize = 0.5,markerstrokewidth=0.5,alpha=0.5)
        #end
        #if yes[ind]>0.0
        Plots.scatter!(p,Tx,Nx,marker_z=1, markersize = 0.55,markerstrokewidth=0,alpha=0.25)
            
        #end
    end
    Plots.scatter!(p,xlabel="time (ms)",ylabel="Neuron ID")

    #display(p)
    savefig("repeated_pattern__pfc.png")
end
=#
#get_division_scatter(ttt,nnn,resolution,yes,)
#Plots.scatter!(ttt,nnn, markersize = 0.65)
#savefig("normal.png")
#nnn,ttt = load_datasets()
#label_online(mat_of_distances)
