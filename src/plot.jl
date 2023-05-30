using .Plots
using StatsPlots
using StatsBase
using OnlineStats
using UMAP
using LinearAlgebra
import LinearAlgebra: normalize
using Plots
using ColorSchemes
function normalised_2dhist(data)
    foreach(normalize!, eachcol(data'))
    return data
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
    t0 = times
    n0 = nodes
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
    list_of_artifact_rows = [] # These will be deleted as they bias analysis.
    for (ind,t) in enumerate(templ)
        psth = fit(Histogram,t,temp_vec)
        if sum(psth.weights[:]) == 0.0
            append!(list_of_artifact_rows,ind)
        end
    end
    adjusted_length = ns+1-length(list_of_artifact_rows)
    data = Matrix{Float64}(undef, adjusted_length, Int(length(temp_vec)-1))
    cnt = 1
    for t in templ
        psth = fit(Histogram,t,temp_vec)        
        if sum(psth.weights[:]) != 0.0
            data[cnt,:] = psth.weights[:]
            @assert sum(data[cnt,:])!=0
            cnt +=1
        end
    end
    #weights = 
    LinearAlgebra.normalize(data)
    #Plots.plot(heatmap(data))
    #Plots.savefig("heatmap_normalized.png")
    return data
end

"""
Pre-allocation for get time surface
"""
function get_ts(nodes,times,dt,tau)
    num_neurons = Int(length(nodes))+1
    total_time =  Int(round(maximum(times)))
    time_resolution = Int(round(total_time/dt))
    # Final output. 
    final_timesurf = zeros((num_neurons, time_resolution+1))
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

    @showprogress for (tt,nn) in zip(times,nodes)

        #Get the current spike
        neuron = Int(round(nn))
        #@show(neuron)
        #@show(length(mv))

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
    @showprogress for t in collect(last_t:dt:total_time)
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
    @time Plots.savefig(file_name)
    
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
