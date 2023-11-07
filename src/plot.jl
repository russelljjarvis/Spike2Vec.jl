using .Plots
using StatsPlots
using StatsBase
using OnlineStats
using UMAP
using LinearAlgebra
import LinearAlgebra: normalize
using Plots
using ColorSchemes
using Statistics


function normalised_matrix!(X)
    foreach(normalize!, eachcol(X))
    #return data
end

function raster(p::IFNF)
    fire = p.records[:fire]
    if typeof(fire) == CuArray{Bool}
        fire = convert(Vector{Bool},fire)
    end    
    x, y = Float32[], UInt32[]
    for t = eachindex(fire)
        for n in findall(fire[t])
            push!(x, t)
            push!(y, n)
        end
    end
    x, y
 
end

function plot_umap_of_dist_vect(mat_of_distances; file_name::String="stateTransMat.png")
    Q_embedding = umap(mat_of_distances',20,n_neighbors=20)
    Plots.plot(Plots.scatter(Q_embedding[1,:], Q_embedding[2,:], title="Spike Time Distance UMAP, reduced precision,", marker=(1, 1, :auto, stroke(0.05)),legend=true))
    savefig(file_name)
    Q_embedding
end


"""
A method to get collect the Inter Spike Intervals (ISIs) per neuron, and then to collect them together to get the ISI distribution for the whole cell population
Also output a ragged array (Array of unequal length array) of spike trains. 
"""
function create_ISI_histogram(nodes::Vector{UInt32},times::Vector{Float32})
    global_isis,spikes_ragged,isi_s = Float32[],Any[],Float32[]
    #global_isis = # the total lumped population ISI distribution.
    
    numb_neurons=Int(maximum(nodes))+1 # Julia doesn't index at 0.
    @inbounds for n in 1:numb_neurons
        push!(spikes_ragged,[])
    end
    @inbounds for (n,t) in zip(nodes,times)
        @inbounds for i in 1:numb_neurons
            if i==n
                push!(spikes_ragged[n],t)
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



function raster(P::Vector)
    y0 = UInt64[0]
    X = Float32[]; Y = UInt64[]
    for p in P
        x, y = raster(p)
        append!(X, x)
        append!(Y, y .+ sum(y0))
        push!(y0, p.N)
    end
    plt = scatter(X, Y, m = (0.5, :black), leg = :none,
                  xaxis=("t", (0, Inf)), yaxis = ("neuron",))
    y0 = y0[2:end-1]
    !isempty(y0) && hline!(plt, cumsum(y0), linecolor = :red)
    return plt
end
"""
Create a 2D histogram/heatmap Its usually good to normalize this retrospectively
"""
function hist2dHeat(nodes::Vector{UInt32}, times::Vector{Float32}, denom_for_bins::Float32)
    stimes = sort(times)
    ns = maximum(unique(nodes))    
    temp_vec = collect(0:Float64(maximum(stimes)/denom_for_bins):maximum(stimes))
    templ = []
    for (cnt,n) in enumerate(collect(1:maximum(nodes)+1))
        push!(templ,[])
    end
    for (cnt,n) in enumerate(nodes)
        push!(templ[n+1],times[cnt])    
    end

    # An artifact row is probably just a neuron that doesn't fire.
    list_of_artifact_rows = [] # These will be deleted as they bias analysis.
    @inbounds @showprogress for (ind,t) in enumerate(templ)
        psth = fit(Histogram,t,temp_vec)
        if sum(psth.weights[:]) == 0.0
            append!(list_of_artifact_rows,ind)
            #@assert sum(t)==0
        end
    end
    adjusted_length = ns+1-length(list_of_artifact_rows)
    data = Matrix{Float64}(undef, adjusted_length, Int(length(temp_vec)))#-1))
    cnt = 1
    @inbounds @showprogress  for t in templ
        psth = fit(Histogram,t,temp_vec)        
        if sum(psth.weights[:]) != 0.0
            data[cnt,:] = psth.weights[:]
            @assert sum(data[cnt,:])!=0
            cnt +=1
        end
    end
    # Normalize data
    @inbounds for (ind,col) in enumerate(eachcol(data))
        data[:,ind] .= (col.-mean(col))./std(col)
    end
    # Clean Nan's
    data[isnan.(data)] .= 0.0
    # Julia's normalizer doesn't work that well, lets apply it too for good measure
    LinearAlgebra.normalize(data)
    data::Matrix{Float64}
end

"""
Pre-allocation for get time surface
"""
function get_ts(nodes,times,dt,tau;disk=false)
    num_neurons = Int(length(nodes))+1
    total_time =  Int(round(maximum(times)))
    time_resolution = Int(round(total_time/dt))
    #@show(time_resolution)

    if !disk
        final_timesurf = zeros((num_neurons, time_resolution+1))

    else
        io = open("/tmp/mmap.bin", "w+")
        # We'll write the dimensions of the array as the first two Ints in the file
        final_timesurf = mmap(io, Matrix{Float32}, (num_neurons,time_resolution+1))
    end

    # Final output. 
    # Timestamp and membrane voltage store for generating time surface
    timestamps = zeros((num_neurons)) .- Inf
    mv = zeros((num_neurons))
    
    get_ts!(nodes,times,final_timesurf,timestamps,num_neurons,total_time,time_resolution,mv,dt,tau)
    return final_timesurf
end
"""
get time surface
"""
function get_ts!(nodes,times,final_timesurf,timestamps,num_neurons,total_time,time_resolution,mv,dt,tau)
    last_t = 0

    @inbounds @showprogress for (tt,nn) in zip(times,nodes)

        #Get the current spike
        neuron = Int(round(nn))

        time = Int(trunc(Int32,tt))       
        # If time of the next spikes leaps over, make sure to generate 
        # timesurfaces for all the intermediate dt time intervals and fill the 
        # final_timesurface.
        if time > last_t
            timesurf = similar(final_timesurf[:,1])
            for t in collect(last_t:dt:time)
                @. timesurf = mv*exp((timestamps-t)/tau)
                final_timesurf[:,1+Int(round(t/dt))] = timesurf
            end
            last_t = time
        end
        # Update the membrane voltage of the time surface based on the last value and time elapsed
        mv[neuron] =mv[neuron]*exp((timestamps[neuron]-time)/tau) +1
        timestamps[neuron] = time
        # Update the latest timestamp at the channel. 
    end
    # Generate the time surface for the rest of the time if there exists no other spikes. 
    timesurf = similar(final_timesurf[:,1])
    @inbounds @showprogress  for t in collect(last_t:dt:total_time)
        @. timesurf = mv*exp((timestamps-t)/tau)
        final_timesurf[:,1+Int(round(t/dt))] = timesurf
    end
end



function color_time_plot(nodes::Vector{Int32}, times::Vector{Float32}, file_name::String)
    perm = sortperm(times)
    nodes = nodes[perm]
    times = times[perm]
    CList = Vector{Float32}(collect(0:1:length(times)))
    cmap = :balance
    Plots.plot(scatter(times,nodes,zcolor=CList, title="Color Time Plot", marker=(2, 2, :auto, stroke(0.0005)),legend=false))
    Plots.savefig("$file_name.color_time.png")
    return CList::Vector{Float32}
end

function get_mean_isis(times,nodes)
    spike_dict = Dict()
    for n in unique(nodes)
        spike_dict[n] = []
    end
    for (st,n) in zip(times,nodes)
        append!(spike_dict[n],st)
    end
    all_isis = []
    for (k,v) in pairs(spike_dict)
        time_old = 0
        for time in spike_dict[k][1:end-1]
            isi = time - time_old
            append!(all_isis,isi)
            time_old = time
        end
    end
    mean_isi = StatsBase.mean(all_isis)
end

function plot_umap(amatrix,mat_of_distances, CList_; file_name::String="empty.png")
    #model = UMAP_(mat_of_distances', 10)
    #Q_embedding = transform(model, amatrix')
    #cs1 = ColorScheme(distinguishable_colors(length(CList_), transform=protanopic))

    Q_embedding = umap(mat_of_distances',10,n_neighbors=10, n_epochs=5000)#, min_dist=0.01, n_epochs=100)
    Plots.plot(scatter(Q_embedding[1,:], Q_embedding[2,:],zcolor=CList_, title="NMNIST Spike Distance UMAP", marker=(1, 1, :auto, stroke(1.0)),legend=true))
    #Plots.plot(scatter!(p,model.knns)
    savefig(file_name)
    Q_embedding
end
"""
Visualize one epoch, as a spike train raster and then an ISI histogram.
"""

function plot_ISI_and_raster_scatter(nodes,times,global_isis)
    p1 = Plots.plot()
    Plots.scatter!(p1,times,nodes,legend = false,markersize = 0.8,markerstrokewidth=0,alpha=0.8, fontcolor=:blue,xlabel="time (Seconds)",ylabel="Cell Id")
    savefig("scatter_plot.png")Spike
    b_range = range(minimum(global_isis), mean(global_isis)+std(global_isis), length=21)
    p2 = Plots.plot()
    Plots.histogram!(p2,global_isis, bins=b_range, normalize=:pdf, color=:gray,xlim=[0.0,mean(global_isis)+std(global_isis)])
    Plots.plot(p1,p2)
    savefig("Spike_raster_and_ISI_bar_plot.png")
    (p1,p2)
end
function plot_umap_ts(nodes::Vector{Int32}, times::Vector{Float32},dt,tau; file_name::String="empty.png")
    perm = sortperm(times)
    nodes = nodes[perm]
    times = times[perm]
    CList = color_time_plot(nodes, times, file_name)

    #CList = collect(0:1:length(times))

    #cmap = :balance

    #Plots.plot(scatter(times,nodes,zcolor=CList, title="Color Time Plot", marker=(2, 2, :auto, stroke(0.0005)),legend=false))
    #Plots.savefig("color_time.png")
    time_end = Int(length(times))
    cmap = :balance
    final_timesurf = get_ts(nodes,times,dt,tau);
    normalize!(final_timesurf)
    Plots.heatmap(final_timesurf)
    Plots.savefig("TimeSurface.png")

    #hist_weights = bespoke_2dhist(nodes,times,denom_for_bins)
    #hist_weights =  hist_weights'
    #Plots.heatmap(final_timesurf)
    #Plots.savefig("heatmap.png")

    @time res_jl = umap(final_timesurf',n_neighbors=20, min_dist=0.001, n_epochs=100)
    #@show(size(final_timesurf'))
    CList_ = [maximum(CList)/i for i in collect(1:length(final_timesurf')) ]

    #@show(length(CList))
    #@show(length(res_jl))
    #CList = collect(0:length(times)/length(res_jl):length(times))

    display(Plots.plot(scatter(res_jl[1,:], res_jl[2,:],zcolor=CList_, title="Spike Timing: UMAP", marker=(2, 2, :auto, stroke(3.5)),legend=false)))
    Plots.savefig(file_name)
    
    #return 
end
function vecplot(p, sym)
    v = getrecord(p, sym)
    y = hcat(v...)'
    x = 1:length(v)
    plot(x, y, leg = :none,
    xaxis=("t", extrema(x)),
    yaxis=(string(sym), extrema(y)))
end

function vecplot(P::Array, sym)
    plts = [vecplot(p, sym) for p in P]
    N = length(plts)
    plot(plts..., size = (600, 400N), layout = (N, 1))
end

function windowsize(p)
    A = sum.(p.records[:fire]) / length(p.N)
    W = round(Int32, 0.5p.N / mean(A)) # filter window, unit=1
end

function density(p, sym)
    X = getrecord(p, sym)
    t = 1:length(X)
    xmin, xmax = extrema(vcat(X...))
    edge = linspace(xmin, xmax, 50)
    c = center(edge)
    ρ = [fit(Histogram, x, edge).weights |> float for x in X] |> x -> hcat(x...)
    ρ = smooth(ρ, windowsize(p), 2)
    ρ ./= sum(ρ, 1)
    p = @gif for t = 1:length(X)
        bar(c, ρ[:, t], leg = false, xlabel = string(sym), yaxis = ("p", extrema(ρ)))
    end
    #is_windows() && run(`powershell start $(p.filename)`)
    #is_unix() && run(`xdg-open $(p.filename)`)
    p
end

function rateplot(p, sym)
    r = getrecord(p, sym)
    R = hcat(r...)
end

function rateplot(P::Array, sym)
    R = vcat([rateplot(p, sym) for p in P]...)
    y0 = [p.N for p in P][2:end-1]
    plt = heatmap(flipdim(R, 1), leg = :none)
    !isempty(y0) && hline!(plt, cumsum(y0), line = (:black, 1))
    plt
end

function activity(p)
    A = sum.(p.records[:fire]) / length(p.N)
    W = windowsize(p)
    A = smooth(A, W)
end

function activity(P::Array)
    A = activity.(P)
    t = 1:length(P[1].records[:fire])
    plot(t, A, leg=:none, xaxis=("t",), yaxis=("A", (0, Inf)))
end

function if_curve(model, current; neuron = 1, dt = 0.1ms, duration = 1second)
    E = model(neuron)
    monitor(E, [:fire])
    f = Float32[]
    for I = current
        clear_records(E)
        E.I = [I]
        SNN.sim!([E], []; dt = dt, duration = duration)
        push!(f, activity(E))
    end
    plot(current, f)
end


function plotss_isi(assign,nlist,tlist)

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
            #push!(collect_isi_bags,bag_of_isis(Nx,Tx))
            #push!(collect_isi_bags_map,a)

            #temp = div_spike_mat[:,ind] #.+sw
            #@show(length(temp))
            #Plots.scatter!(p,temp,legend=false, markercolor=a)
        end
    end
    display(Plots.plot(p))# = Plots.plot()
    savefig("the_jesus_examplar.png")
    #collect_isi_bags,collect_isi_bags_map
end

function internal_validation0(assign,)
    p = Plots.plot()
    collect_isi_bags = []
    for (ind,a) in enumerate(assign)
        if a!=0
            #Tx = tlist[ind]
            #xlimits = maximum(Tx)
            #Nx = nlist[ind]
            Plots.scatter!(p,Tx,Nx,legend = false, markercolor=a,markersize = 0.8,markerstrokewidth=0,alpha=0.8, bgcolor=:snow2, fontcolor=:blue, xlims=(0, xlimits))
        end
    end
    #display(Plots.plot(p))# = Plots.plot()
    savefig("the_jesus_examplar.png")
end

function ragged_to_uniform(times)
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
function internal_validation1(assign::Vector{UInt32}, div_spike_mat_no_displacement::Matrix{Vector{Vector{Float32}}};file_path="")

    labels = Vector{Float32}(unique(assign))
    #@show(labels)
    #labels2cols = Vector{Any}([])
    
    #for l in labels
    #    push!(labels2cols,[])
    #end
    @inbounds for l in labels
        Nxag = Float32[]
        Txag = Float32[]
        color_code = Int64[]
        color_codes = Int64[]

        pscatter = Plots.plot()
        Nxold = []
        Txold = []

        @inbounds for (col,label) in enumerate(assign)
            px = Plots.plot()
            if label==l
                

                Nx = UInt32[]
                Tx = Vector{Float32}([])
                @inbounds for (ind_cell,row) in enumerate(eachrow(div_spike_mat_no_displacement[:,col]))
                    if length(row)!=0
                        for times in row
                            for tt in times
                                for t in tt
                                    push!(Tx,t) 
                                    push!(Nx,ind_cell)
                                    push!(color_code,col)
                                end
                            end
                        end
                    end               
                end
                append!(Nxag,Nx)
                append!(Txag,Tx)
                append!(color_codes,color_code)
                if length(Txold)>0
                    if length(Tx)>1
                        o0 = HeatMap(1:1:length(div_spike_mat_no_displacement[:,col]), 0.0:maximum(Tx)/length(Tx):maximum(Tx))
                        fit!(o0, zip(Nx, Tx))
                        ts0 = copy(o0.counts)
                        o1 = HeatMap(1:1:length(div_spike_mat_no_displacement[:,col]), 0.0:maximum(Txold)/length(Txold):maximum(Txold))
                        fit!(o1, zip(Nxold, Txold))                    
                        ts1 = copy(o1.counts)
                        temp = cor(ts0,ts1)    
                        avg_len=length(temp)
                        temp[isnan.(temp)] .= 0.0
                        temp = mean(temp)/avg_len
                        #@show(mean(temp),sum(temp))
                        if temp>0.1
                            p2=Plots.heatmap(ts0,legend=false)
                            p3=Plots.heatmap(ts1,legend=false)
                            layout = @layout [a ; b ]
                            Plots.plot(p2, p3, layout=layout,legend=false)
                            savefig("zebra_correlated_heatmap_$temp.png")
                        end

                    end
                    end
                append!(Nxold,Nx)
                append!(Txold,Tx)
            end
            p2 = Plots.scatter!(px,Txag,Nxag,markercolor=Int(l),markersize = 1.2,markerstrokewidth=0,alpha=0.8, fontcolor=:blue,legend=false)
            p3 = Plots.scatter!(p2,Txold,Nxold,markercolor=Int(l)+1,markersize = 1.8,markerstrokewidth=0,alpha=0.8, fontcolor=:blue,legend=false)
            Plots.plot(p3,legend=false)
            ##
            # TODO use DrWatson there.
            ## to save the path
            #@show(l)
            savefig("correlated_Scatter_.$col.png")
            
    
    
        end
        #if length(Txag)!=0
            #@show()
            #display(Plots.scatter!(pscatter,Txag,Nxag,legend = false, markercolor=color_codes,markersize = 0.8,markerstrokewidth=0,alpha=0.6, bgcolor=:snow2, fontcolor=:blue))
            #savefig("NEW_scatter_match_$l.png")
        #end                            

    end
    #labels2cols
end
function internal_validation_dict(assignments::Vector{UInt32}, div_spike_mat_no_displacement::Matrix{Vector{Vector{Float32}}};file_path="")
    labels = Vector{Float32}(unique(assignments))
    spike_motif_dict_both = Dict()
    spike_motif_dict_times = Dict()
    spike_motif_dict_nodes = Dict()

    for l in labels 
        spike_motif_dict_both[l] = []
    end

    @inbounds for l in labels
        @inbounds for (col,label) in enumerate(assignments)
            if label==l
                append!(spike_motif_dict_both[l],col)#
            end
        end
    end
    internal_validation_dict!(div_spike_mat_no_displacement,spike_motif_dict_both,spike_motif_dict_nodes,spike_motif_dict_times)
end
function unpack_spikes_from_columns!(div_spike_mat_no_displacement,column,Tx,Nx)
    row = div_spike_mat_no_displacement[:,column]
    for (ind_cell,times) in enumerate(row)
        for tt in times
            for t in tt
                push!(Tx,t) 
                push!(Nx,ind_cell)
            end
        end
    end
end


function internal_validation_dict!(div_spike_mat_no_displacement,spike_motif_dict_both::Dict,spike_motif_dict_nodes::Dict,spike_motif_dict_times::Dict)
    for (key,value) in pairs(spike_motif_dict_both)
        spike_motif_dict_nodes[key] = Dict()
        spike_motif_dict_times[key] = Dict()
    end

    for (key,value) in pairs(spike_motif_dict_both)
        for (pattern_occurance_ind,column) in enumerate(value)
            Tx = []
            Nx = []
            unpack_spikes_from_columns!(div_spike_mat_no_displacement,column,Tx,Nx)
            spike_motif_dict_nodes[key][pattern_occurance_ind] = copy(Nx)
            spike_motif_dict_times[key][pattern_occurance_ind] = copy(Tx)

        end

    end
    plot_internal_validation_dict(spike_motif_dict_nodes,spike_motif_dict_times)
end

function plot_internal_validation_dict(spike_motif_dict_nodes,spike_motif_dict_times)
    for k0 in keys(spike_motif_dict_nodes)
        Px = Plots.scatter()    
        for  (k1,_) in pairs(spike_motif_dict_nodes[k0])
            nodes = spike_motif_dict_nodes[k0][k1]
            times = spike_motif_dict_times[k0][k1]
            Plots.scatter!(Px,times,nodes,markercolor=Int(k1),label="occurance_$k1",legend=false)
        end
        len=length(spike_motif_dict_nodes[k0])
        savefig("scatter_match.$k0.$len.png")

    end
end
        #=
            @inbounds for (ind_cell,row) in enumerate(eachrow(div_spike_mat_no_displacement[:,col]))
                if length(row)!=0
                    for times in row
                        for tt in times
                            for t in tt
                                push!(Tx,t) 
                                push!(Nx,ind_cell)
                                push!(color_code,col)
                            end
                        end
                    end
                    append!(spike_motif_dict_nodes[label],Nx)
                    append!(spike_motif_dict_times[label],Tx)

                end               
            end
            append!(Nxag,Nx)
            append!(Txag,Tx)
            append!(color_codes,color_code)
            if length(Txold)>0
                if length(Tx)>1
                    o0 = HeatMap(1:1:length(div_spike_mat_no_displacement[:,col]), 0.0:maximum(Tx)/length(Tx):maximum(Tx))
                    fit!(o0, zip(Nx, Tx))
                    ts0 = copy(o0.counts)
                    o1 = HeatMap(1:1:length(div_spike_mat_no_displacement[:,col]), 0.0:maximum(Txold)/length(Txold):maximum(Txold))
                    fit!(o1, zip(Nxold, Txold))                    
                    ts1 = copy(o1.counts)
                    temp = cor(ts0,ts1)    
                    avg_len=length(temp)
                    temp[isnan.(temp)] .= 0.0
                    temp = mean(temp)/avg_len
                    #@show(mean(temp),sum(temp))
                    if temp>0.1
                        p2=Plots.heatmap(ts0,legend=false)
                        p3=Plots.heatmap(ts1,legend=false)
                        layout = @layout [a ; b ]
                        Plots.plot(p2, p3, layout=layout,legend=false)
                        savefig("zebra_correlated_heatmap_$temp.png")
                    end

                end
                end
            append!(Nxold,Nx)
            append!(Txold,Tx)
        end
        p2 = Plots.scatter!(px,Txag,Nxag,markercolor=Int(l),markersize = 1.2,markerstrokewidth=0,alpha=0.8, fontcolor=:blue,legend=false)
        p3 = Plots.scatter!(p2,Txold,Nxold,markercolor=Int(l)+1,markersize = 1.8,markerstrokewidth=0,alpha=0.8, fontcolor=:blue,legend=false)
        Plots.plot(p3,legend=false)
        ##
        # TODO use DrWatson there.
        ## to save the path
        #@show(l)
        savefig("correlated_Scatter_.$col.png")
        


    end
    #if length(Txag)!=0
        #@show()
        #display(Plots.scatter!(pscatter,Txag,Nxag,legend = false, markercolor=color_codes,markersize = 0.8,markerstrokewidth=0,alpha=0.6, bgcolor=:snow2, fontcolor=:blue))
        #savefig("NEW_scatter_match_$l.png")
    #end                            

end
#labels2cols
end
=#

# export density
# function density(p, sym)
#   X = getrecord(p, sym)
#   t = dt*(1:length(X))
#   xmin, xmax = extrema(vcat(X...))
#   edge = linspace(xmin, xmax, 100)
#   c = center(edge)
#   ρ = [fit(Histogram, x, edge).weights |> reverse |> float for x in X] |> x->hcat(x...)
#   ρ = smooth(ρ, windowsize(p), 2)
#   ρ ./= sum(ρ, 1)
#   surface(t, c, ρ, ylabel="p")
# end
# function density(P::Array, sym)
#   plts = [density(p, sym) for p in P]
#   plot(plts..., layout=(length(plts),1))
# end
