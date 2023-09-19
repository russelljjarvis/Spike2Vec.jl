using JLD
using Plots
using SpikeTime
using Revise
using StatsBase
using ProgressMeter
using UMAP
using CSV
using DataFrames
using OnlineStats
using Distributions
using OhMyREPL


function load_datasets()
    df=  CSV.read("output_spikes.csv",DataFrame)
    nodes = Vector{UInt32}(df.id)
    nodes = [UInt32(n+1) for n in nodes]
    times = Vector{Float32}(df.time_ms)
    (nodes,times)
end

function onlinestats_pca(nodes,times)
    #if !isfile("psth.jld")
        psth = hist2dHeat(nodes, Vector{Float32}(times), Float32(1825))
        @save "psth.jld" psth
    #else
        @load "psth.jld" psth
        Plots.heatmap(psth)
        savefig("psth.png")
    #end
    #@show(psth)
    #final_timesurf = get_ts(nodes,times,dt,tau;disk=true)
    #o = CCIPCA(2, length(psth))                # Project 10-dimensional vectors into 2D
    #fit!(o, psth)                      # Fit to u1
    ##OnlineStats.fittransform!(o, psth[:]) # Fit u4 and then project u4 into the space
    #sort!(o)                         # Sort from high to low eigenvalues
    #@show(o)
    #o[1]                             # Get primary (1st) eigenvector
    #OnlineStats.relativevariances(o)    
    #@show(o[1])
    #Plots.plot(Plots.scatter(o[1], o[2], title="Spike Time OnlineStatsPCA", markersize = 2.5,markerstrokewidth=0,alpha=1.0, bgcolor=:snow2,legend=false))
    #savefig("OnlineStatsPCA.png")
    plot_umap!(psth; file_name="UMAPPSTHSTDP.png")

    (o[1],o[2],o,psth) 


end
(nodes,times) = load_datasets()
o_1,o_2,o,psth = onlinestats_pca(nodes,times)
function plot_umap!(mat_of_distances; file_name::String="empty.png")
    Q_embedding = umap(mat_of_distances,5,n_neighbors=5)#, min_dist=0.01, n_epochs=100)
    Plots.plot(Plots.scatter(Q_embedding[1,:], Q_embedding[2,:], title="Spike Time Distance UMAP", markersize = 2.5,markerstrokewidth=0,alpha=1.0, bgcolor=:snow2,legend=false))
    savefig(file_name)
end
#=
function onlinePCA_(distmat; file_name::String="empty.png")
    tmp = mktempdir()
    # distance_matrix
    OnlinePCA.writecsv(joinpath(tmp, "Data.csv"), distmat)
    # Binarization
    OnlinePCA.csv2bin(csvfile=joinpath(tmp, "Data.csv"), binfile=joinpath(tmp, "Data.zst"))

    # Summary of data
    @show(OnlinePCA.sumr(binfile=joinpath(tmp, "Data.zst"), outdir=tmp))

end
=#
function plotting_routines_and_preprocessing()
    if !isfile("stdp.jld")
        @time (nodes,times) = load_datasets()
        p1 = Plots.plot()
        Plots.scatter!(p1,times,nodes,legend = false,markersize = 0.05,markerstrokewidth=0,alpha=0.5, bgcolor=:snow2, fontcolor=:blue,xlabel="time (Seconds)",ylabel="Cell Id")
        savefig("scatter_plot_exp2.png")
        resolution = 20
        numb_neurons = UInt64(maximum(nodes))
        maxt = maximum(times)        
        (distmat,tlist,nlist,start_windows,end_windows,spike_distance_size) = get_divisions(nodes,times,resolution,numb_neurons,maxt;plot=false,disk=false)
        @save "stdp.jld" distmat tlist nlist start_windows end_windows spike_distance_size nodes times
        Plots.heatmap(distmat)
        savefig("Distmat_sqaure.png")
        plot_umap!(distmat; file_name="UMAPSTDP.png")
    else
        @load "stdp.jld" distmat tlist nlist start_windows end_windows spike_distance_size nodes times
    end
    
    if !isfile("stdpLabels.jld")
        ε=250.0

    sqr_distmat = label_online_distmat(distmat;threshold=ε,disk=false)
    @save "STDP_labels.jld" sqr_distmat
    else
        @load "STDP_labels.jld" sqr_distmat
    end
    println("completed 2")
    if !isfile("Cluster_stdpLabels.jld")
    
        (R,sort_idx,assign) = cluster_distmat!(sqr_distmat)
        println("completed 3")
    else
        @load "Cluster_stdpLabels.jld" R sort_idx assign
    end

    assing_progressions,assing_progressions_times = get_state_transitions(start_windows,end_windows,sqr_distmat,assign;threshold= ε)
    repeated_windows = state_transition_trajectory(start_windows,end_windows,sqr_distmat,assign,assing_progressions,assing_progressions_times;plot=false)#,file_name="state_transitions_stdp.png")
    assign[unique(i -> assign[i], 1:length(assign))].=0.0
    sqr_distmat,distmat, assign,tlist, nlist
end
function plot_the_repeat_spikes(assign,nlist,tlist,distmat)
    p = Plots.plot()
    @showprogress for (ind,a) in enumerate(assign)
        if a!=0
            Tx = tlist[ind]
            xlimits = maximum(Tx)
            Nx = nlist[ind]
            Plots.scatter!(p,Tx,Nx,legend = false, markercolor=a,markersize = 0.1,markerstrokewidth=0,alpha=0.5, bgcolor=:snow2, fontcolor=:blue, xlims=(0, xlimits))

        end
    end
    Plots.plot(p)
    savefig("repeating_states.png")
end
function plot_the_repeat_vectors(assign,nlist,tlist,distmat)

    for aa in unique(assign)
        p1 = Plots.plot()
        for (ind,a) in enumerate(assign)
            if a==aa && a!=0
                p1 = Plots.plot!(p1,distmat[:,a])#, markercolor=a)
            end
        end
        Plots.plot(p1)
        savefig("repeating_states_vectors$aa.png")
    end
end
sqr_distmat, distmat,assign,tlist, nlist = plotting_routines_and_preprocessing()
plot_the_repeat_spikes(assign,nlist,tlist,distmat)
plot_the_repeat_vectors(assign,nlist,tlist,distmat)

#=
function label_online_distmat(mat_of_distances)#,nclasses)

    distance_matrix = zeros(length(eachrow(mat_of_distances)),length(eachrow(mat_of_distances)))
    display(distance_matrix)
    all_perm_pairs = []


    @showprogress for (ind,row) in enumerate(eachrow(mat_of_distances))
        push!(all_perm_pairs,[])
        for (ind2,row2) in enumerate(eachrow(mat_of_distances))
            best_distance = 100000.0
            distance = evaluate(Euclidean(),row,row2)
            if distance<7
                push!(all_perm_pairs[ind],ind2)
                distance_matrix[ind,ind2] = distance
            else
                distance_matrix[ind,ind2] = -10.0
            end
        end
    end
    display(Plots.heatmap(distance_matrix))
    #@show(all_perm_pairs)
    distance_matrix
 end
function cluster_distmat(mat_of_distances)
    display(mat_of_distances)
    R = affinityprop(mat_of_distances)
    sort_idx =  sortperm(assignments(R))
    assign = R.assignments
    R,sort_idx,assign
end
=#
#=
(nnn,ttt)= load_datasets()
#display(Plots.scatter(ttt,nnn))
resolution = 90
mat_of_distances = get_plot(ttt,nnn,resolution)
display(mat_of_distances)
distmat = label_online_distmat(mat_of_distances)#,nclasses)
display(distmat)
(R,sort_idx,assign) = cluster_distmat(distmat)
display(Plots.heatmap(distmat))
function get_division_scatter2(times,nodes,division_size,distmat,sort_idx,assign)
    step_size = maximum(times)/division_size
    end_window = collect(step_size:step_size:step_size*division_size)
    spike_distance_size = length(end_window)
    start_windows = collect(0:step_size:(step_size*division_size)-step_size)

    mat_of_distances = zeros(spike_distance_size,maximum(unique(nodes))+1)
    segment_length = end_window[3] - start_windows[3]
    fig = Figure(backgroundcolor=RGBf(0.6, 0.6, 0.96))

    p=Plots.scatter(backgroundcolor=RGBf(0.6, 0.6, 0.96))
    witnessed_unique=[]
    @showprogress for (ind,toi) in enumerate(end_window)
        sw = start_windows[ind]
        spikes = divide_epoch(nodes,times,sw,toi)    
        Nx=Vector{Int32}([])
        Tx=Vector{Float32}([])
        for (i, t) in enumerate(spikes)
            for tt in t
                if length(t)!=0
                    push!(Nx,i)
                    push!(Tx,Float32(sw+tt))
                end
            end
        end
        for (ii,xx) in enumerate(distmat[ind,:])
            if abs(xx)<7
                push!(witnessed_unique,assign[ii])
                
                Plots.scatter!(p,Tx,Nx, markercolor=assign[ii],legend = false, markersize = 2.0,markerstrokewidth=0,alpha=1.0, bgcolor=:snow2, fontcolor=:blue)#, marker=:auto)#,group=categorical(length(unique(sort_idx))))
             end
        end


    end
    nunique = length(unique(witnessed_unique))

    Plots.scatter!(p,xlabel="time (ms)",ylabel="Neuron ID", yguidefontcolor=:black, xguidefontcolor=:black,title = "N observed states $nunique")##,backgroundcolor=RGBf(0.6, 0.6, 0.96))
    p2 = Plots.scatter(times,nodes,legend = false, markersize = 1.9,markerstrokewidth=0,alpha=0.7, bgcolor=:snow2, fontcolor=:blue,thickness_scaling = 1)#,group=categorical(length(unique(sort_idx))))
    Plots.scatter!(p2,xlabel="time (ms)",ylabel="Neuron ID", yguidefontcolor=:black, xguidefontcolor=:black,title = "Un-labeled spike raster")##,backgroundcolor=RGBf(0.6, 0.6, 0.96))

    Plots.plot(p,p2, layout = (2, 1))

    savefig("repeated_pattern_song_bird2.png")
end
get_division_scatter2(ttt,nnn,resolution,distmat,sort_idx,assign)

function get_state_transitions(times,nodes,division_size,distmat,sort_idx,assign)
    step_size = maximum(times)/division_size
    end_window = collect(step_size:step_size:step_size*division_size)
    spike_distance_size = length(end_window)
    start_windows = collect(0:step_size:(step_size*division_size)-step_size)
    nunique = length(unique(assign))
    assing_progressions=[]
    assing_progressions_times=[]

    @showprogress for (ind,toi) in enumerate(end_window)
        sw = start_windows[ind]
        push!(assing_progressions_times,sw)
    end
    for row in eachrow(distmat)
        for (ii,xx) in enumerate(row)
            if abs(xx)<7
                push!(assing_progressions,assign[ii])
 
            end
        end
    end
    assing_progressions,assing_progressions_times
end
get_division_scatter2(ttt,nnn,resolution,distmat,sort_idx,assign)
assing_progressions,assing_progressions_times = get_state_transitions(ttt,nnn,resolution,distmat,sort_idx,assign)


#using SimpleWeightedGraphs, Graphs
nunique = length(unique(assign))
#using Plots, GraphRecipes
empty = zeros(nunique,nunique)
for (ind,x) in enumerate(assing_progressions)
    if ind < length(assing_progressions)
        empty[x,assing_progressions[ind+1]]+=1
    end 
end
repeated_windows = Vector{UInt32}([])
repititions = zeros(size(empty))
for (ind,_) in enumerate(eachrow(empty))
    for (y,val) in enumerate(empty[ind,:])
        if val==1
            #@show(ind,y)
            repititions[ind,y] = 0.0

        elseif val>1
            repititions[ind,y] = val 

            push!(repeated_windows,ind)


        end 

    end
end

#@show(store_non_zero)
display(Plots.heatmap(empty))

assing_progressions[unique(i -> assing_progressions[i], 1:length(assing_progressions))].=-1
p1 =Plots.plot()
Plots.scatter!(p1,assing_progressions,legend=false)
#https://github.com/open-risk/transitionMatrix

Plots.plot!(p1,assing_progressions,legend=false)
display(p1)
savefig("state_transition_trajectory.png")
g = SimpleWeightedDiGraph(empty)

edge_label = Dict((i,j) => string(empty[i,j]) for i in 1:size(empty, 1), j in 1:size(empty, 2))

graphplot(g; names = 1:length(empty), weights=empty)#,line_z=empty)#, edge_label)
savefig("state_transition_matrix.png")

function get_repeated_scatter(times,nodes,division_size,repeated_windows)
    step_size = maximum(times)/division_size
    end_window = collect(step_size:step_size:step_size*division_size)
    spike_distance_size = length(end_window)
    start_windows = collect(0:step_size:(step_size*division_size)-step_size)

    segment_length = end_window[3] - start_windows[3]
    fig = Figure(backgroundcolor=RGBf(0.6, 0.6, 0.96))

    p=Plots.scatter(backgroundcolor=RGBf(0.6, 0.6, 0.96))
    @showprogress for (ind,toi) in enumerate(end_window)

    #for (ind,repeat_window) in enumerate(repeated_windows)
        if repeated_windows[ind]!=-1
            toi = end_window[ind]
            sw = start_windows[ind]
            spikes = divide_epoch(nodes,times,sw,toi)    
            Nx=Vector{Int32}([])
            Tx=Vector{Float32}([])
            for (i, t) in enumerate(spikes)
                for tt in t
                    if length(t)!=0
                        push!(Nx,i)
                        push!(Tx,Float32(sw+tt))
                    end
                end
            end
            Plots.scatter!(p,Tx,Nx,legend = false, markercolor=repeated_windows[ind],markersize = 2.0,markerstrokewidth=0,alpha=1.0, bgcolor=:snow2, fontcolor=:blue, xlims=(0, maximum(times)))
        end

    end
    nunique = length(unique(repeated_windows))

    #
    Plots.scatter!(p,xlabel="time (ms)",ylabel="Neuron ID", yguidefontcolor=:black, xguidefontcolor=:black,title = "N observed states $nunique", xlims=(0, 22.2))##,backgroundcolor=RGBf(0.6, 0.6, 0.96))
    #Plots.plot!(p,
    #titlefont = font(0.017, "Courier")
    #)
    #display(p)R,M,sort_idx
    #end
    p2 = Plots.scatter(times,nodes,legend = false, markersize = 1.9,markerstrokewidth=0,alpha=0.7, bgcolor=:snow2, fontcolor=:blue,thickness_scaling = 1)#,group=categorical(length(unique(sort_idx))))
    Plots.scatter!(p2,xlabel="time (ms)",ylabel="Neuron ID", yguidefontcolor=:black, xguidefontcolor=:black,title = "Un-labeled spike raster")##,backgroundcolor=RGBf(0.6, 0.6, 0.96))

    Plots.plot(p,p2, layout = (2, 1))

    savefig("genuinely_repeated_pattern_song_bird.png")
end
@show(assing_progressions)
get_repeated_scatter(ttt,nnn,resolution,assing_progressions)
#assing_progressions[(i -> assing_progressions[i]==0, 1:length(assing_progressions))].=0

display(Plots.scatter(assing_progressions))

=#