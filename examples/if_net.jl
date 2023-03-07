using Plots
#using UnicodePlots
using Plots

using SpikingNeuralNetworks
#unicodeplots()

SNN.@load_units

include("genPotjans.jl")
function get_Ncell(scale=1.0::Float64)
	ccu = Dict{String, Int32}("23E"=>20683,
		    "4E"=>21915, 
		    "5E"=>4850, 
		    "6E"=>14395, 
		    "6I"=>2948, 
		    "23I"=>5834,
		    "5I"=>1065,
		    "4I"=>5479)
	ccu = Dict{String, Int32}((k,ceil(Int64,v*scale)) for (k,v) in pairs(ccu))
	Ncells = Int32(sum([i for i in values(ccu)])+1)
	Ne = Int32(sum([ccu["23E"],ccu["4E"],ccu["5E"],ccu["6E"]]))
    Ni = Int32(Ncells - Ne)
    Ncells, Ne, Ni, ccu

end

function potjans_layer()
    scale = 1.0/100.0
    Ncells,Ne,Ni, ccu = get_Ncell(scale)    
    pree = prie = prei = prii = 0.1
    K = round(Int, Ne*pree)
    sqrtK = sqrt(K)
    pree = prie = prei = prii = 0.1
    g = 1.0
    tau_meme = 10   # (ms)
    je = 2.0 / sqrtK * tau_meme * g
    ji = 2.0 / sqrtK * tau_meme * g 
    jee = 0.15je 
    jei = je 
    jie = -0.75ji 
    jii = -ji
    genStaticWeights_args = (;Ncells,jee,jie,jei,jii,ccu,scale)
    (potjans_weights(genStaticWeights_args),Ne,Ni)
end

(w0Weights,Lee,Lie,Lei,Lii),Ne,Ni = potjans_layer()
spy(w0Weights) 
savefig("potjanswiring.png")

(LeeSyn,LeiSyn,LiiSyn,LieSyn,pree,poste,prei,posti) = SNN.SpikingSynapse(w0Weights,Lee,Lei,Lii,Lie)#Lexc,Linh)
P = [pree,poste,prei,posti] # populations 
C = [LeeSyn,LeiSyn,LiiSyn,LieSyn] # connections
SNN.monitor([pree,poste,prei,posti], [:fire])
duration = 1second
SNN.sim!(P, C; duration = duration)
SNN.train!(P, C; duration = duration)


SNN.raster(P) 
savefig("raster_all.png")
@show(nodes,times)
(nodes,times) = SNN.get_trains(P);


function bespoke_2dhist(nbins::Float32,nodes::Vector{Float32},times::Vector{Float32},fname=nothing)
    stimes = sort(times)
    ns = maximum(unique(nodes))    
    temp_vec = collect(0:Float64(maximum(stimes)/nbins):maximum(stimes))
    templ = []
    for (cnt,n) in enumerate(collect(1:maximum(nodes)+1))
        push!(templ,[])
    end
    for (cnt,n) in enumerate(nodes)

        push!(templ[n+1],times[cnt])    
        @show(templ[n+1])
    end
    list_of_artifact_rows = []
    #data = Matrix{Float64}(undef, ns+1, Int(length(temp_vec)-1))
    for (ind,t) in enumerate(templ)
        psth = fit(Histogram,t,temp_vec)
        #data[ind,:] = psth.weights[:]
        if sum(psth.weights[:]) == 0.0
            append!(list_of_artifact_rows,ind)
        end
    end
    @show(list_of_artifact_rows)
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

    ##
    #
    ##
    #data = view(data, vec(mapslices(col -> any(col .!= 0), data, dims = 2)), :)[:]
    #@show(first(data[:]))
    #@show(last(data[:]))
    ##
    # All neuron s are block normalised according to a global mean/std rate
    ##

    #data .= (data .- StatsBase.mean(data))./StatsBase.std(data)
    #@show(size(data))
    return data
end


function normalised_2dhist(data)
    ##
    # Each neuron is indipendently normalised according to its own rate
    ##
    
    #for (ind,row) in enumerate(eachrow(data))
    #    data[ind,:] .= row .- StatsBase.mean(row)./sum(row)
    #    @show(data[ind,:]) 
    #end
    data = data[:,:]./maximum(data[:,:])
    return data
end


nbins = 425.0
data = bespoke_2dhist(nbins,nodes,times)
datan = normalised_2dhist(data)
Plots.plot(heatmap(datan),legend = false, normalize=:pdf)
Plots.savefig("heatmap_normalised.png")

function divide_epoch(nodes,times,duration)
    t1=[]
    n1=[]
    t0=[]
    n0=[]
    for (n,t) in zip(nodes,times)
        if t<=duration
            append!(t0,t)
            append!(n0,n)            
        else
            append!(t1,t)
            append!(n1,n)
        end
    end
    (t0,n0,t1,n1)
end

(t0,n0,t1,n1) = divide_epoch(nodes,times,duration);


function get_vector_coords()
    (t0,n0,t1,n1) = divide_epoch(nodes,times,duration)
    maxt = findmax(sort!(unique(vcat(spikes,ground_spikes))))[1]

    #@show(times)

end



function nloss(E,ngt_spikes,ground_spikes)
    spikes = get_spikes(E)
    spikes = [s/1000.0 for s in spikes]
	maxt = findmax(sort!(unique(vcat(spikes,ground_spikes))))[1]
    if size(spikes)[1]>1
        t, S = SPIKE_distance_profile(spikes, ground_spikes;t0=0,tf = maxt)
        spkdistance = SpikeSynchrony.trapezoid_integral(t, S)/(t[end]-t[1]) # == SPIKE_distance(y1, y2)
    else
        spkdistance = 10.0
    end
	if length(spikes)>1
		custom_raster2(spikes,ground_spikes)
		custom_raster(spikes,ground_spikes)
	end
	spkdistance*=spkdistance

    delta = abs(size(spikes)[1] - ngt_spikes)
    return spkdistance+delta

end