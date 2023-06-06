using Plots
using MAT
using StatsBase
using ProgressMeter
spikes = matread("M1_d1A_S.mat")["GC06_M1963_20191204_S1"]["Transients"]["Raster"]
FPS = matread("M1_d1A_S.mat")["GC06_M1963_20191204_S1"]["Movie"]["FPS"]
frame_width = 1.0/FPS #0.08099986230023408 #second, sample_rate =  12.3457#Hz
function convert_bool_matrice_to_ts(spikes,frame_width)
    nodes = UInt32[]
    times = Float32[]
    for (indy,row) in enumerate(eachrow(spikes))
        for (indx,x) in enumerate(row)
            if x
                push!(nodes,indy)
                push!(times,indx*frame_width)                
            end
        end
	end
    whole_duration = length(spikes[1,:])*frame_width
    (nodes,times,whole_duration)
end
(nodes,times,whole_duration) = convert_bool_matrice_to_ts(spikes,frame_width)
function create_ISI_histogram(nodes,spikes)
    spikes = []
    numb_neurons=Int(maximum(nodes))+1
    @inbounds for n in 1:numb_neurons
        push!(spikes,[])
    end
    @inbounds @showprogress for (i, _) in enumerate(spikes)
        for (n,t) in zip(nodes,times)
            if i==n
                push!(spikes[i],t)
            end
        end
    end
    global_isis = []
    isi_s = []
    @inbounds @showprogress for (i, times) in enumerate(spikes)
        push!(isi_s,[])

        for (ind,x) in enumerate(times)
            if ind>1
                isi_current = x-times[ind-1]
                push!(isi_s[i],isi_current)

            end
        end
        append!(global_isis,isi_s[i])
    end
    global_isis
end

global_isis = create_ISI_histogram(nodes,spikes)
Plots.scatter(times,nodes,legend = false,markersize = 0.8,markerstrokewidth=0,alpha=0.8, bgcolor=:snow2, fontcolor=:blue,xlabel="time (ms)",ylabel="Cell id")
b_range = range(minimum(global_isis), mean(global_isis)+var(global_isis), length=21)
display(Plots.histogram(global_isis, bins=b_range, normalize=:pdf, color=:gray))

