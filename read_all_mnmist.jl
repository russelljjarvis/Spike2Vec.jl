using JLD
using DrWatson
#using Plots
using SpikingNeuralNetworks
using Revise
using StatsBase
using ColorSchemes
using PyCall
using LinearAlgebra
using ProgressMeter
using Distances
using Plots
using StaticArrays
using OnlineStats
using Heatmap
#using Clustering

#using OnlineStats
#using GigaSOM
#using Gadfly
#using GigaScatter
using Clustering
import OnlineStats
#using Gadfly
#import Cairo
#using Compose
#using SGtSNEpi, Random

function filter_on_off(x,y,times,p,l,nodes)
    xo=[];yo=[];timeso=[];po=[];lo=[];nodeso=[]
    x1=[];y1=[];times1=[];p1=[];l1=[];nodes1=[]

    for (x_,y_,times_,p_,l_,nodes_) in zip(x,y,times,p,l,nodes)
        if p_==1
            #if x_<UInt32(50) && y_<UInt32(50)
            push!(xo,x_)
            push!(yo,y_)
            push!(timeso,times_)
            push!(lo,l_)
            push!(nodeso,nodes_)
            #end
        else
            push!(x1,x_)
            push!(y1,y_)
            push!(times1,times_)
            push!(l1,l_)
            push!(nodes1,nodes_)


        end
    end
    (xo,yo,timeso,po,lo,nodeso,x1,y1,times1,p1,l1,nodes1)

end

function load_datasets()
    @load "all_mnmist.jld" storage
    xx=[]
    yy=[]
    tt = []
    pp = []
    ll = []
    nn = []
    window_size = 0
    @showprogress for (ind,s) in enumerate(storage)

        (x,y,times,p,l,nodes) = (s[1],s[2],s[3],s[4],s[5],s[6])
        if  maximum(times) > window_size
            window_size = maximum(times)
        end
        append!(xx,x)
        append!(yy,y)
        append!(tt,times)
        append!(pp,p)
        append!(ll,l)
        append!(nn,nodes)
    end

    (x,y,times,p,l,nodes,x1,y1,times1,p1,l1,nodes1) = filter_on_off(xx,yy,tt,pp,ll,nn)

    #(xo,yo,timeso,po,lo,nodeso,x1,y1,times1,p1,l1,nodes1) = filter_on_off(x,y,times,p,l,nodes)
    #@show(p)
    labels = [Int32(ll) for ll in l]
    #perm = sortperm(labels)
    times = [Float32(t) for t in times]
    nodes = [Int32(t) for t in nodes]
    p = [Int32(p) for t in p]

    x = [UInt32(x_+1) for x_ in x ]
    y = [UInt32(y_+1) for y_ in y ]
    return (x,y,times,p,l,nodes,window_size)
end
#   (x,y,times,p,labels,nodes) = load_datasets()
#labels = l
function make_spike_movie(x,y,times,labels)
    cnt=0
    l_old=1
    x_l = []
    y_l = []
    mymatrix = zeros((36,36))
    @showprogress for (t_,x_,y_,l) in zip(times,x,y,labels)    
        if l==l_old
            push!(x_l,x)
            push!(y_l,y)

        end
        if l!=l_old && length(x_l)>1 && length(y_l)>1
            @show(length(x_l))
            @show(length(y_l))
            @show(l)
            @show(l_old)
            @show(mymatrix[x_l,y_l]) #.= 10.0
            #display(UnicodePlots.spy(mymatrix))
            Plots.heatmap(mymatrix)
            Plots.savefig("NMNIST_matrix$l.png")
    
            mymatrix = zeros((359,359))
            x_l = []
            y_l = []
        end
        cnt+=1
        l_old=l
    end
end
#make_spike_movie(x,y,times,labels)

#a[perm]
#SpikeTime
#=
nodes = nodes[perm]
times = times[perm]
labels = labels[perm]
x = x[perm]
y = y[perm]
=#
#using UnicodePlots

function get_changes(times,labels)
    time_break = []
    l_old=0
    t_old=0
    @showprogress for (t,l) in zip(times,labels)
        if l==l_old
            @show(abs(t-t_old))
        end
        if l!=l_old
            push!(time_break,Float32(t))
        end
        l_old = l
        t_old = t
    end
    time_break
end
#time_break = get_changes(times,labels)
#@show(time_break)

#display(Plots.scatter(times,nodes,markersize=0.01))
#p=Plots.scatter(times,nodes,markersize=0.01)
#Plots.vline!(p,time_break,markersize=0.01)
#savefig("nmist_raster_plot.png")
#nodes = convert(Vector{Int32},nodes)

#dt = 0.1
#tau = 0.4
#plot_umap(nodes,times,dt,tau;file_name="UMAP_OF_NMNIST.png")


function get_plot_uniform(storage)
    times,nodes = storage[1][3],storage[1][6]
    spike_distance_size = length(storage)
    n0ref = expected_spike_format(nodes,times)
    non_zero_spikings=[]
    @inbounds for tx in n0ref
        if length(tx)!=0
            push!(non_zero_spikings,length(tx)) 
        end
    end
    mean_spk_counts = Int32(round(mean(non_zero_spikings)))
    temp = LinRange(0.0, maximum(times), mean_spk_counts)
    linear_uniform_spikes = Vector{Float32}([i for i in temp])

    p = Plots.plot()
    @inbounds for xx in collect(1:1250)
        altitude = [xx for i in collect(1:1250)]
        Plots.scatter!(p,linear_uniform_spikes,altitude, legend = false,markersize=0.1)
    end
    savefig("UniformSpikes.png")
    #p=nothing
    #p = Plots.plot()
    #o = fit!(IndexedPartition(Float32, KHist(length(unique(nodes))), length(unique(nodes)), zip(nodes, times)))
    #plot(o)

    #@time o = IndexedPartition(Float64, KHist(100), 100)
    #@time fit!(o,zip(convert(Vector{Float64},nodes),convert(Vector{Float64},times)))
    #o = HeatMap(0:.1:maximum(times), 1:1:length(unique(nodes)))
    #fit!(o, zip(linear_uniform_spikes, altitude))

    #x, y = randn(10^6), 5 .+ randn(10^6)
    #plot(o, marginals=false, legend=true)
    #plot(o) 
    #savefig("UniformLabelledSpikesPartition$l.png")

    return (linear_uniform_spikes,mean_spk_counts,nodes,length(storage))
end

function expected_spike_format(nodes1,times1)
    nodes1 = [i+1 for i in nodes1]
    n0ref =  []
    @inbounds for i in collect(1:1250)
        push!(n0ref,[])
    end
    @inbounds for i in collect(1:1250)
        @inbounds for (neuron, t) in zip(nodes1,times1)
            if i == neuron
                push!(n0ref[Int32(i)],Float32(t))
            end            
        end
    end
    n0ref
end

function get_plot2(storage)
    (linear_uniform_spikes,mean_spk_counts,nodes,spike_distance_size) = get_plot_uniform(storage)
    p = Plots.plot()
    #mat_of_distances = Array{Float32}(zeros(spike_distance_size,length(unique(nodes))+1))
    list_lists = Vector{Any}([])
    px = Plots.plot()
    prev = 0.0
    @inbounds @showprogress for (ind,s) in enumerate(storage)
        (times,labels,nodes) = (s[3],s[5],s[6])
        #(x,y,times,p,l,nodes) = (s[1],s[2],s[3],s[4],s[5],s[6])
        #p
        #@show(times)
        #@show(nodes)
        #if length(nodes)>2
        ll=labels[1]
        l = convert(Int32,ll)
        p = Plots.plot()
        Plots.scatter(times,nodes, legend = false,markersize=0.1)
        savefig("LabelledSpikes$l.png")
        #p = Plots.plot()
        #o = fit!(IndexedPartition(Float32, KHist(length(unique(nodes))), length(unique(nodes)), zip(nodes, times)))
        #plot(o)
        #@time o = IndexedPartition(Float64, KHist(100), 100)
        #@time fit!(o,zip(convert(Vector{Float64},nodes),convert(Vector{Float64},times)))
        o = HeatMap(0:1:maximum(times),1:1:length(unique(nodes)))
        fit!(o, zip(times,nodes))
        plot(o, marginals=false, legend=true)
        savefig("LabelledSpikesPartition$l.png")
        self_distance_populate = Array{Float32}(zeros(1250))
        if length(times) > 1
            times = expected_spike_format(nodes,times)
            #self_distance_populate = Array{Float32}(zeros(length(times)))
            get_vector_coords_uniform!(linear_uniform_spikes,times,self_distance_populate)
            #Plots.plot!(px,self_distance_populate.+prev,label="$l")
            push!(list_lists,self_distance_populate)
            #prev += maximum(self_distance_populate)
            #mat_of_distances[:,ind] = self_distance_populate
        end

    end

    @save "matrix_vectors.jld" list_lists

    return list_lists
    #return mat_of_distances
end
#division_size=maximum(times)/10.0
#label_inventory_size = length(unique(labels))


function get_labels(storage)
    ll = []
    @inbounds @showprogress for (ind,s) in enumerate(storage)
        labels = s[5]
        push!(ll,labels[1])
    end
    return ll
end


function function_do(list_lists,ll)
    p = Plots.plot()
    prev=0.0
    @inbounds @showprogress for (ind,row) in enumerate(eachrow(mat_of_distances))
        l = ll[ind]
        Plots.plot!(p,row.+prev,label="$l")
        prev += maximum(row)

    end
    Plots.plot!(p,legend=:outerbottom, legendcolumns=length(mat_of_distances))
    savefig("NMNIST_codes.png")
end

function plot_lists(list_lists,ll)

    p = Plots.plot()
    prev = 0.0
    for (ind,l) in zip(collect(250:265),list_lists[250:265])
        l_ = ll[ind]

        p = Plots.plot!(p,l[1:1250].+prev,label="$l_")
        prev += maximum(l[1:1250])
    end
    savefig("vector_differences_another.png")
end

function get_matrix(list_lists,ll)
    min_ = Int(1000000)
    @inbounds @showprogress for (ind,l) in enumerate(list_lists)
        if Int(length(l)) < min_
            min_ = Int(length(l))   
        end
    end
    mat_of_distances = Array{Float32}(zeros(length(list_lists),1250))
    @inbounds @showprogress for (ind,l) in enumerate(list_lists)
        mat_of_distances[ind,:] = l#[1:min_]
    end
    mat_of_distances[isnan.(mat_of_distances)] .= 0.0
    Plots.heatmap(mat_of_distances)#, axis=(xticks=(1:5, xs), yticks=(1:10, ys), xticklabelrotation = pi/4) ))
    savefig("Unormalised_heatmap.png")
    @inbounds @showprogress for (ind,col) in enumerate(eachcol(mat_of_distances))
        mat_of_distances[:,ind] .= (col.-mean(col))./std(col)
    end
    mat_of_distances[isnan.(mat_of_distances)] .= 0.0
    time_bin_lengths = collect(1:300)
    Plots.heatmap(mat_of_distances)
    savefig("heatmap.png")
    p = Plots.plot()
    Plots.plot!(p,mat_of_distances[10,:])
    Plots.plot!(p,mat_of_distances[20,:])
    savefig("vector_differences_another_NMNIST.png")

    @save "mat_of_distances.jld" mat_of_distances

    return mat_of_distances
end
function post_proc_viz(mat_of_distances)
    # ] add https://github.com/JeffreySarnoff/AngleBetweenVectors.jl
    # ] add Distances
    angles0 = []
    distances0 = []
    @inbounds @showprogress for (ind,self_distances) in enumerate(eachrow(mat_of_distances))
        temp0 = mat_of_distances[ind,:]
        temp1 = Vector{Float32}([0.01 for i in 1:length(temp0)])
        θ = angle(temp0,temp1)
        r = evaluate(Euclidean(),temp0, temp1)
        append!(angles0,θ)
        append!(distances0,r)        
    end
    @save "distances_angles.jld" angles0 distances0#,angles1,distances1
    return angles0,distances0#,angles1,distances1)
end


function final_plots(mat_of_distances,ll)
    p=nothing
    p = Plots.plot()
    angles0,distances0 = post_proc_viz(mat_of_distances)
    #Gadfly.plot(x =distances0, y= angles0, color = ll)
    savefig("relative_to_uniform_referenceNMMIST.png")  
     
end

#function cluster()

#end

#=
function cluster_distances_graph_embedd(mat_of_distances,L)
    #mat_of_distances = mat_of_distances
    #mat_of_distances = copy(transpose(mat_of_distances))
    dim = 2


    #@show(size(mat_of_distances))


    som = initGigaSOM(mat_of_distances, 20, 20)    # random initialization of the SOM codebook
    som = trainGigaSOM(som, mat_of_distances)      # SOM training
    clusters = mapToGigaSOM(som, mat_of_distances) # extraction of per-cell cluster IDs
    e = embedGigaSOM(som, mat_of_distances)        # EmbedSOM projection to 2D
    #import Pkg; Pkg.add("GigaScatter")
    #using GigaScatter
    #Plots.heatmap(e')
    savePNG("Levine13-CD4.png",
    solidBackground(rasterize((200,200),        # bitmap size
    Matrix{Float64}(e'),                      # the embedding coordinates
    expressionColors(
    scaleNorm(Array{Float64}(L)),  # 5th column contains CD4 expressions
    expressionPalette(100, alpha=1.0)))))   # colors for plotting (based on RdYlBu)
    savefig("didit_work_NM.png")

    #savePNG("Levine13-CD4.png",
    #solidBackground(rasterize((100,100),        # bitmap size
    #Matrix{Float64}(e'),                      # the embedding coordinates
    #expressionColors(
    #scaleNorm(Array{Float64}(L)),  # 5th column contains CD4 expressions
    #expressionPalette(100, alpha=1.0)))))  
    #savefig("didit_work_NM.png")
    #savePNG("Levine13-CD4.png",Matrix{Float64}(e'))   # colors for plotting (based on RdYlBu)
    #Y = sgtsnepi(mat_of_distances; d=dim, Y0 = Y0, max_iter = 500);


    #- `lwd_in=0.5`: line width for internal edges
    #- `lwd_out=0.3`: line width for external edges
    #- `edge_alpha=0.2`: the alpha channel for the edges
    #- `clr_in=nothing`: set color for all intra-cluster edges (if nothing, color by `cmap`)
    #- `clr_out=colorant"#aabbbbbb"`: the color of inter-cluster edges
    #show_embedding(Y, Lx ,clr_out=cmap,edge_alpha=0.5)#; A = pot_conn, res = (5000, 5000) )
    #A = pot_conn
    #Y0 = 0.01 * randn( size(A,1), 3 );
    #Y = sgtsnepi(mat_of_distances; d = 2, Y0 = Y0, max_iter = 500);
    #neighbor_recall(pot_conn, Y)
    #sc = scatter( Y[:,1], Y[:,2], color = L, colormap = cmap, markersize = 5)
    #savefig("reduced_dimension_clusters_NMINST.png")
    #scene.center = false

    #save("potjans_static_wiring_network_embedding.png")
    #show_embedding( Y, L ; A = pot_conn)#, res = (5000, 5000) )
end
=#
function penultimate(angles0,distances0,ll)
    #@load "distances_angles.jld" angles0 distances0
    #@show(ll)
    #myplot = Gadfly.plot(x =distances0, y= angles0, color = ll)
    #@show(distances0)
    #@show(angles0)
    #compose(render(myplot), compose(context(), rectangle(), fill("white")))
    #draw(PNG("classify_to_uniform_referenceNMMIST.png", 3inch, 3inch), myplot)   
    
    #angles0,distances0 = post_proc_viz(mat_of_distances)
    #Gadfly.plot(x =distances0, y= angles0, color = ll)
    #savefig("relative_to_uniform_referenceNMMIST.png")  
    o = HeatMap(Float64(minimum(angles0)):.1:Float64(maximum(angles0)), 0:1:10)

    #x, y = randn(10^6), 5 .+ randn(10^6)
    
    fit!(o, zip(Vector{Float64}(angles0), Vector{Float64}(ll)))
    
    #plot(o)

    p = Plots.plot()
    display(Plots.plot(x=angles0,y=ll,color=ll))#, color=ll)
    #Plots.scatter(distances0,angles0, markercolor = ll)
    #savefig("scatternmnist_angles.png")
    
    p = Plots.plot()
    display(Plots.scatter(distances0,ll))#, color=ll)
    #Plots.scatter(distances0,angles0, markercolor = ll)
    savefig("scatternmnist_distances.png")
    o
end

using ColorSchemes
function umap_plots(mat_of_distances,ll)
    cs1 = ColorScheme(distinguishable_colors(length(ll), transform=protanopic))
    #@show(sizeof(M))
    temp = [1.0 for i in ll]
    Plots.plot(scatter(ll, temp, marker_z=R.assignments, legend=true))
    savefig("clustering_NMNIST.png")

    plot_umap(M,mat_of_distances,ll; file_name="umap_of_NMNIST_Data.png")
    #savefig("didit_workpablo.png")
    return M
end


#@load "matrix_vectors.jld" list_lists 
@load "ll.jld" ll 
#mat_of_distances = get_matrix(list_lists,ll)

@load "mat_of_distances.jld" mat_of_distances
#post_proc_viz(mat_of_distances)

#M = umap_plots(mat_of_distances,ll)

function label_online(mat_of_distances,ll)
    classes = 10
    R = kmeans(mat_of_distances', classes; maxiter=2000, display=:iter)
    a = assignments(R) # get the assignments of points to clusters
    c = counts(R) # get the cluster sizes
    M = R.centers # get the cluster centers
    #M = copy(M'[:])
    labelled_mat_of_distances = copy(mat_of_distances)
    for (ind,row) in enumerate(eachrow(mat_of_distances))

        for ix in collect(1:classes)
            current_centre = M[:,ix]
            distance = sum(abs.(row .- current_centre))
            #@show(distance,ind)
            #@show(size(centre))
            if distance < 960.0
                labelled_mat_of_distances[ind,:] .= ll[ind]
            else
                distance = 10000.0

                #@show(distance)
                #@show(ind)
                #@show(ll[ind])
                #@show(ix)
                #@show(centre)
            end

        end
    end
    Plots.heatmap(labelled_mat_of_distances)
    savefig("labelled_mat_of_distancesNMINST.png")
    #return M
end
label_online(mat_of_distances,ll)
#set_default_plot_size(8inch, 6inch)
#mtcars = dataset("datasets", "mtcars")
#Z = zscore(Matrix{Float64}(mtcars[:,2:11]),1)
#xx=[i for i in collect(1:length(mat_of_distances))]
#using Heatmap, Gadfly, StatsBase
#using Clustering
#using StatsPlots
#using Distances

#hm = Heatmap

#Generate distance matrix, perform hierarchical clustering, plot
#dist_mat = pairwise(Euclidean(1e-12), mat_of_distances, dims=2)
#hcl1 = hclust(dist_mat, linkage=:ward)
#dg = Plots.plot(hcl1, xticks=false, yticks=true)
#display(plot!(size=(800,200)))
#display()
#display(Gadfly.plot(z=dist_mat, hm.heatmap(matrix="raw"),
#    Geom.rectbin, hm.dendrogram(dendrogram="both"),
# Scale.color_continuous(colormap=Scale.lab_gradient("steelblue3","white","darkred"))
#))

#=
angles0,distances0 = post_proc_viz(mat_of_distances)


#cluster_distances_graph_embedd(mat_of_distances,ll)
o = penultimate(angles0,distances0,ll)
display(plot(o, marginals=false, legend=true))
=#
#@load "mat_of_distances.jld" mat_of_distances
#mat_of_distances[isnan.(mat_of_distances)] .= 0.0
#final_plots(mat_of_distances,ll)

#=
if isfile("matrix_vectors.jld")
    #(linear_uniform_spikes,mean_spk_counts,nodes,spike_distance_size) = get_plot_uniform(storage)
    #list_lists = get_plot2(storage)

    @load "matrix_vectors.jld" list_lists 
    #plot_lists(list_lists)

    #@load "all_mnmist.jld" storage
    #ll = get_labels(storage)
    #@save "ll.jld" ll

    @load "ll.jld" ll 
    mat_of_distances = get_matrix(list_lists,ll)
    
    #@show(mat_of_distances)
    #display(mat_of_distances)
    @save "mat_of_distances.jld" mat_of_distances
    

    mat_of_distances[isnan.(mat_of_distances)] .= 0.0
    r = evaluate(Euclidean(),mat_of_distances[1,:], Vector{Float32}([0.0 for i in 1:length(mat_of_distances[1,:])]))

    #@show(ll)
    
    #function_do(mat_of_distances,ll)
else
    @load "all_mnmist.jld" storage
    list_lists = get_plot2(storage)
    function_do(mat_of_distances)

end
=#
#using Pkg
#Pkg.add("ElectricalEngineering")
#using ElectricalEngineering
#angulardimension

#plot_lists(list_lists)

#@load "all_mnmist.jld" storage
#ll = get_labels(storage)
#@save "ll.jld" ll
