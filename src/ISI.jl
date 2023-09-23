
function bag_of_isis(nodes::Vector{<:Integer},times::Vector{<:Real})
    spikes_ragged = Vector{Any}([])
    numb_neurons=Int(maximum(nodes))+1 # Julia doesn't index at 0.
    @inbounds for n in 1:numb_neurons
        push!(spikes_ragged,Vector{Float32}[])
    end
    @inbounds for i in 1:numb_neurons
        for (n,t) in zip(nodes,times)
            if i==n
                push!(spikes_ragged[UInt32(i)],t)
            end
        end
    end
    processed_isis = bag_of_isis(spikes_ragged)
    processed_isis::Vector{Any}
end

"""
On a RTSP packet, get a bag of ISIs. So we can analyse the temporal structure of RTSPs regardless of spatial structure.
"""
function bag_of_isis(spikes_ragged::AbstractArray)
    bag_of_isis = Vector{Any}([]) # the total lumped population ISI distribution.
    isi_s = Float32[]
    @inbounds for (i, times) in enumerate(spikes_ragged)
        push!(isi_s,[])
    end
    @inbounds for (i, times) in enumerate(spikes_ragged)
        
        for (ind,x) in enumerate(times)
            if ind>1
                isi_current = x-times[ind-1]
                push!(isi_s[i],isi_current)
            end
        end
        append!(bag_of_isis,Vector{Float32}(isi_s[i]))
    end
    bag_of_isis::Vector{Any}
end
