
using SGtSNEpi, Random
using Revise
#import SGtSNEpi.neighbor_recall
using CairoMakie, Colors, LinearAlgebra
using GLMakie
using Graphs
#import Graphs.label_propagation
#using Leiden
import StatsBase.mean

GLMakie.activate!()
include("../src/models/genPotjansWiring.jl")
function protect_variable()
    scale = 0.05
    (pot_conn,x,y,ccu) = potjans_layer(scale)
    @show(pot_conn)
    #Lx = Vector{Int64}(zeros(size(pot_conn)))
    
    return pot_conn,x,y,ccu,scale#,Lx
end
pot_conn,x,y,ccu,scale = protect_variable()
@show(pot_conn)
#=
function scoped_fix(ccu,Lx,scale)
    v_old=1
    K = length(ccu)
    cum_array = Array{Array{UInt32}}(undef,K)
    for i in 1:K
        cum_array[i] = Array{UInt32}([])
    end
    for (k,v) in pairs(ccu)
        push!(cum_array,collect(v_old:v+v_old)[:])
        v_old=v+v_old
    end
    start = 1
    for (ind_,val) in enumerate(cum_array)
        Lx[val] .= ind_ 
    end
end
scoped_fix(ccu,Lx,scale)
Lx = convert(Vector{Int64},Lx)
Lx = Lx[(Lx.!=0)]
L = convert(Vector{Int64},Lx)

dim = 2
Random.seed!(0);
stored_min = abs(minimum(pot_conn))
for (ind,row) in enumerate(eachrow(pot_conn))
    for (j,colind) in enumerate(row)
        if pot_conn[ind,j] < 0.0
            pot_conn[ind,j] = pot_conn[ind,j]+stored_min
        end
    end
    #   image.png@show(pot_conn[ind,:])
    @assert mean(pot_conn[ind,:]) >= 0.0
end

pot_conn[diagind(pot_conn)] .= 0.0
#=
G = DiGraph(pot_conn)
results = label_propagation(G)#, maxiter=1000; rng=nothing, seed=nothing)
@show(results)
γ = 0.25
result = Leiden.leiden(Symmetric(pot_conn), resolution = γ)
@show(unique(result[2]))
=#
Y0 = 0.1 * randn( size(pot_conn,1), dim);


cmap = distinguishable_colors(
           maximum(L) - minimum(L) + 1,
           [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

Y = sgtsnepi(pot_conn; d=dim, Y0 = Y0, max_iter = 500);


#- `lwd_in=0.5`: line width for internal edges
#- `lwd_out=0.3`: line width for external edges
#- `edge_alpha=0.2`: the alpha channel for the edges
#- `clr_in=nothing`: set color for all intra-cluster edges (if nothing, color by `cmap`)
#- `clr_out=colorant"#aabbbbbb"`: the color of inter-cluster edges
show_embedding(Y, Lx ,clr_out=cmap,clr_out=cmap,edge_alpha=0.5)#; A = pot_conn, res = (5000, 5000) )
A = pot_conn
Y0 = 0.01 * randn( size(A,1), 3 );
Y = sgtsnepi(A; d = 3, Y0 = Y0, max_iter = 500);
#neighbor_recall(pot_conn, Y)
sc = scatter( Y[:,1], Y[:,2], Y[:,3], color = L, colormap = cmap, markersize = 5)
#scene.center = false

#save("potjans_static_wiring_network_embedding.png")
#show_embedding( Y, L ; A = pot_conn)#, res = (5000, 5000) )

#sc
#@show(L)
#dim = 2
#Random.seed!(0);
#Y0 = 0.01 * randn( size(pot_conn[1],1), dim);
#pot_conn = abs.(pot_conn[1])

#=

cmap = distinguishable_colors(
           maximum(L) - minimum(L) + 1,
           [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

Y = sgtsnepi(pot_conn; d=dim, Y0 = Y0, max_iter = 600);
#@show(length(Y))
scene = show_embedding( Y, L )#; A = pot_conn, res = (5000, 5000) )
scene.center = false
save("potjans_static_wiring_network_embedding.png")
scene = show_embedding( Y, L ; A = pot_conn, res = (5000, 5000) )
#scene.center = false
#savefig("potjans_st
#atic_wiring_network_embedding_wires.png")
save("potjans_static_wiring_network_embedding_wires.png")
=#
#=
L = zeros(size(pot_conn[1])[1])
@show(sizeof(L))
labelsx = [ind for (ind,row) in enumerate(eachrow(pot_conn[2])) if sum(row)!=0 ] 
L[labelsx] .= 1
labelsy = [ind for (ind,row) in enumerate(eachrow(pot_conn[3])) if sum(row)!=0 ] 
L[labelsy] .= 2

labelsz = [ind for (ind,row) in enumerate(eachrow(pot_conn[4])) if sum(row)!=0 ] 
L[labelsz] .= 3

labelsa = [ind for (ind,row) in enumerate(eachrow(pot_conn[5])) if sum(row)!=0 ] 
L[labelsa] .= 4

=#
#L = [1+sign(StatsBase.mean(row)) for row in eachrow(pot_conn)]


#using Colors

#Random.seed!(0);
#display(plot(sc))
#record(sc, "sgtsnepi-animation.gif", range(0, 1, length = 24*8); framerate = 24) do ang
#  rotate_cam!( sc.figure.scene.children[1], 2*π/(24*8), 0, 0 )
#end
=#