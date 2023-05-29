using JLD
using Plots
using SpikeTime
using DrWatson
using ProgressMeter
using RecurrenceAnalysis

import DelimitedFiles: readdlm

function load_song_bird()

    spikes = []
    
    nodes = [n for (n, t) in eachrow(readdlm("songbird_spikes.txt", '\t', Float64, '\n'))]
    for _ in 1:maximum(unique(nodes))+1
        push!(spikes,[])
    end

    for (n, t) in eachrow(readdlm("songbird_spikes.txt", '\t', Float64, '\n'))
        push!(spikes[Int32(n)],t)
    end
    nnn=Vector{Int32}([])
    ttt=Vector{Float32}([])
    for (i, t) in enumerate(spikes)
        for tt in t
            if length(t)!=0
                push!(nnn,i);
                push!(ttt,Float32(tt))
            end
        end
    end
    numb_neurons=Int(maximum(nodes))+1

    maxt = (maximum(ttt))    
    (nnn,ttt,spikes,numb_neurons,maxt)
end

(nodes,times,spikes,numb_neurons,maxt)= load_song_bird()
resolution = 100 # 65
ε=10.5

(mat_of_distances,tlist,nlist,start_windows,end_windows,spike_distance_size) = get_divisions(nodes,times,resolution,numb_neurons,maxt,plot=false)


sss =  StateSpaceSet(hcat(mat_of_distances))
R = RecurrenceMatrix(sss, ε; metric = Euclidean(), parallel=true)
#display(RecurrenceAnalysis.recurrenceplot(R; minh = 25, maxh = 0.5))
display(RecurrenceAnalysis.recurrenceplot(R; ascii = true))
#rqa(R)
xs, ys = RecurrenceAnalysis.coordinates(R)# -> xs, ys
#get_division_scatter_identify(nlist,tlist,start_windows,end_windows,distmat,sort_idx,assign,nodes,times,repeated_windows,plots=true,file_name="songbird.png";threshold= ε)

#fig, ax = 
#display(Plots.scatter(xs, ys; markersize = 1))
#ax.aspect = 1
#display(fig)


plot_umap_of_dist_vect(mat_of_distances; file_name="umap_songbird.png")
distmat = label_online_distmat(mat_of_distances;threshold= ε)#,nclasses)
(R,sort_idx,assign) = cluster_distmat(distmat)
assing_progressions,assing_progressions_times = get_state_transitions(start_windows,end_windows,distmat,sort_idx,assign;threshold= ε)

#assing_progressions,assing_progressions_times = get_state_transitions(start_windows,end_windows,distmat,sort_idx,assign;threshold=threshold)
repeated_windows = state_transition_trajectory(start_windows,end_windows,distmat,sort_idx,assign,assing_progressions,assing_progressions_times;plot=true,file_name="songbird.png")
nslices=length(start_windows)
#@show(repeated_windows)
get_repeated_scatter(nlist,tlist,start_windows,end_windows,repeated_windows,nodes,times,nslices,file_name="songbird.png")
get_division_scatter_identify(nlist,tlist,start_windows,end_windows,distmat,sort_idx,assign,nodes,times,repeated_windows,plots=true,file_name="songbird.png";threshold= ε)
get_division_scatter_identify_via_recurrence_mat(nlist,tlist,start_windows,end_windows,nodes,times;file_name::String="recurrence_bird.png",ε::Real=5)
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
function get_division_scatter(times,nodes,division_size,yes,sort_idx)
    #yes = yes[sort_idx]
    step_size = maximum(times)/division_size
    end_windows = collect(step_size:step_size:step_size*division_size)
    spike_distance_size = length(end_windows)
    #start_windows = collect(0:step_size:step_size*division_size-1)
    start_windows = collect(0:step_size:(step_size*division_size)-step_size)

    mat_of_distances = zeros(spike_distance_size,maximum(unique(nodes))+1)
    segment_length = end_windows[3] - start_windows[3]
    p=Plots.scatter()
    #p=Plots.scatter(times,nodes,legend=true)
    end_windows = end_windows[sort_idx]
    start_windows = start_windows[sort_idx]

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
    savefig("repeated_pattern__songbird.png")
end
=#
#get_division_scatter(ttt,nnn,resolution,yes,sort_idx)
#Plots.scatter!(ttt,nnn, markersize = 0.65)
#savefig("normal.png")
#nnn,ttt = load_datasets()
#label_online(mat_of_distances)
