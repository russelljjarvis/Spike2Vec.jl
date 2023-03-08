using Plots
#using UnicodePlots
using Plots
#using JLD
using SpikingNeuralNetworks
#unicodeplots()
#using OhMyREPL
SNN.@load_units
using SparseArrays
import LinearAlgebra.normalize!
#using ProfileView

function potjans_layer(scale=1.0::Float64)
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
    scale = 1.0/50.0
    #Ncells,Ne,Ni, ccu = potjans_layer(scale)    
    Ncells,Ne,Ni, ccu = potjans_layer(scale)    

     #profile_test(1)  # run once to trigger compilation (ignore this one)
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
    (SNN.potjans_weights(genStaticWeights_args),Ne,Ni)
end
#=
function get_trains(x::Vector{Float32},y::Vector{Int64})
    lenx=length(x)
    leny=length(y)
    x = SVector{(4225333),Float32}(x)
    y = SVector{(lenx),Int32}(y)

    #y = @SVector y
    return (x,y)
    #(SVector{lenx}(x),SVector{leny}(y))
end
=#
#type_ = SpikingNeuralNetworks.IF{Vector{Float32}, Vector{Bool}}
#@show(type)
type_ = SparseMatrixCSC{Float32, Int64}
function getpopulations(Lee::type_,Lei::type_,Lii::type_,Lie::type_)
    eerow = [x for (x,y,v) in zip(findnz(Lee)...) ]
    iirow = [x for (x,y,v) in zip(findnz(Lii)...) ]
    eirow = [x for (x,y,v) in zip(findnz(Lei)...) ]
    ierow = [x for (x,y,v) in zip(findnz(Lie)...) ]
    eecol = [y for (x,y,v) in zip(findnz(Lee)...) ]
    iicol = [y for (x,y,v) in zip(findnz(Lii)...) ]
    eicol = [y for (x,y,v) in zip(findnz(Lei)...) ]
    iecol = [y for (x,y,v) in zip(findnz(Lie)...) ]
    @assert length(eecol) == length(eerow) 
    @assert length(iicol) == length(iirow) 
    @assert length(eicol) == length(eirow) 
    @assert length(iecol) == length(ierow) 

    @assert !(0.0 in ierow)
    @assert !(0.0 in eirow)
    @assert !(0 in iirow)
    @assert !(0 in eerow)
    @assert !(0.0 in eecol)
    @assert !(0.0 in iicol)
    @assert !(0.0 in eicol)
    @assert !(0.0 in iecol)
    ee_src = SNN.IF(;N = length(eerow), param = SNN.IFParameter(;El = -49mV))
    ii_src = SNN.IF(;N = length(iirow), param = SNN.IFParameter(;El = -49mV))
    ei_src = SNN.IF(;N = length(eirow), param = SNN.IFParameter(;El = -49mV))
    ie_src = SNN.IF(;N = length(ierow), param = SNN.IFParameter(;El = -49mV))
    ee_tgt = SNN.IF(;N = length(eerow), param = SNN.IFParameter(;El = -49mV))
    ii_tgt = SNN.IF(;N = length(iirow), param = SNN.IFParameter(;El = -49mV))
    ei_tgt = SNN.IF(;N = length(eirow), param = SNN.IFParameter(;El = -49mV))
    ie_tgt = SNN.IF(;N = length(ierow), param = SNN.IFParameter(;El = -49mV))

    new_weight = Lee+Lei+Lie+Lii
    ww = unique([x for (x,y,v) in zip(findnz(new_weight)...) ])
    return (ee_src,ii_src,ei_src,ie_src,ee_tgt,ii_tgt,ei_tgt,ie_tgt)
end

function global_scope_sucks()
    @time (_,Lee,Lie,Lei,Lii),Ne,Ni = potjans_layer()
    print("wiring done")
    (ee_src,ii_src,ei_src,ie_src,ee_tgt,ii_tgt,ei_tgt,ie_tgt) = getpopulations(Lee,Lei,Lii,Lie)
    (LeeSyn,LeiSyn,LiiSyn,LieSyn) = SNN.SpikingSynapse(ee_src,ii_src,ei_src,ie_src,ee_tgt,ii_tgt,ei_tgt,ie_tgt,Lee,Lei,Lii,Lie)#Lexc,Linh)
    P = [ee_src,ii_src,ei_src,ie_src,ee_tgt,ii_tgt,ei_tgt,ie_tgt] # populations 
    C = [LeeSyn] # connections
    return P,C
end
P, C = global_scope_sucks()
SNN.monitor(P, [:fire])
@time SNN.sim!(P, C; duration = 1second)
SNN.raster(P)
(times,nodes) = SNN.get_trains(P)
nbins = 525
data = SNN.bespoke_2dhist(nbins,times,nodes)
foreach(normalize!, eachcol(data'))
Plots.plot(heatmap(data),legend = false)#, normalize=:pdf)
Plots.savefig("untrainedHeatMap_raster_trained.png")
Plots.savefig("default_raster_untrained.png")
#Plots.savefig("heatmap_untrained_unnormalised.png")
SNN.train!(P, C; duration = 1second)
SNN.raster(P)
Plots.savefig("default_raster_trained.png")
(times,nodes) = SNN.get_trains(P)
nbins = 525
data = SNN.bespoke_2dhist(nbins,times,nodes)
foreach(normalize!, eachcol(data'))
Plots.plot(heatmap(data),legend = false)#, normalize=:pdf)
Plots.savefig("trainingHeatMap_raster_trained.png")

#println("my code failed")


#P,C = global_scope_sucks()
#@show(P)
#@show(C)
    #nbins = 525
    #(times,nodes) = SNN.get_trains(P);

    #SNN.train!(P, C; duration = duration)
    #SNN.monitor(P, [:fire])

    #SNN.raster(P) 
    #savefig("trained_raster_all.png")
    #(times,nodes) = SNN.get_trains(P);

    

#(times,nodes) = 
#=
#(LeeSyn,LeiSyn,LiiSyn,LieSyn) = global_scope_sucks()
println("makes wiring okay!, but may fail at sim")
#P = [ee_src,ii_src,ei_src,ie_src,ee_tgt,ii_tgt,ei_tgt,ie_tgt] # populations 
#C = [LeeSyn,LeiSyn,LiiSyn,LieSyn] # connections
#SNN.monitor(P, [:fire])
#duration = 1second
#println("Monitor sim okay but fails elsewhere!")
#SNN.sim!(P, C; duration = duration)
#@profview SNN.sim!(P, C; duration = duration)
println("Does sim okay but fails elsewhere!")
#
#SNN.raster(P) 
#println("Does sim okay but fails elsewhere 2!")
#savefig("untrained_raster_all.png")
nbins = 525
#(times,nodes) = SNN.get_trains(P);
#valgrind --smc-check=all-non-file julia if_net.jl

data = SNN.bespoke_2dhist(nbins,times,nodes)
Plots.plot(heatmap(data),legend = false, normalize=:pdf)
Plots.savefig("heatmap_untrained_unnormalised.png")



savefig("trained_raster_all.png")

(times,nodes) = SNN.get_trains(P);
    #@save "costly_sim.jld" nodes times P
#else
    #@load "costly_sim.jld" nodes times P
    #(times,nodes) = SNN.get_trains(P);

    #(times,nodes) = get_trains(times,nodes)
#end
spy(w0Weights) 
savefig("corrupted_potjanswiring.png")


#nbins = 2425
#data = bespoke_2dhist(nbins,nodes,times)
SNN.raster(P) 
savefig("raster_all.png")

data = SNN.bespoke_2dhist(nbins,times,nodes)
#datan = SNN.normalised_2dhist(data)
Plots.plot(heatmap(data),legend = false, normalize=:pdf)
Plots.savefig("heatmap_normalised.png")

(t0,n0,t1,n1) = SNN.divide_epoch(nodes,times,duration);
=#