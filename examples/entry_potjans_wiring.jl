
using SGtSNEpi, Random
using CairoMakie, Colors, LinearAlgebra
using GLMakie
GLMakie.activate!()
include("genPotjansWiring.jl")
scale = 1.0/40.0
(pot_conn,x,y,ccu) = potjans_layer(scale)
#scale = 1.0/40.0

Lx = Vector{Int64}(zeros(size(pot_conn[1])[1]))
function scoped_fix(ccu,Lx,scale)
    cumulative = Dict{String, Vector{Int64}}()
    v_old=1
    cum_array = Any[]
    for (k,v) in pairs(ccu)
        ## update the cummulative cell count
        cumulative[k]=collect(v_old:v+v_old)
        push!(cum_array,collect(v_old:v+v_old)[:])
        v_old=v+v_old
    end
    @show(cum_array)
    start = 1
    grab_v = [Int(sum(v)) for v in values(cumulative)]
    println("sanity check")

    for (ind_,val) in enumerate(cum_array)
        Lx[val] .= ind_ 
    end
    #ccuv = sort(grab_v)
    #for (ind_,val) in enumerate(ccuv)
    #    Lx[start:val] .= ind_ 
    #    start = val#
    #end
    #grab_v
end
scoped_fix(ccu,Lx,scale)
@show(size(pot_conn[1])[1])
#@show(sum(grab_v))
Lx = convert(Vector{Int64},Lx)
Lx = Lx[(Lx.!=0)]
L = convert(Vector{Int64},Lx)

dim = 2
Random.seed!(0);
pot_conn = abs.(pot_conn[1])
dropzeros!(pot_conn)
Y0 = 0.01 * randn( size(pot_conn,1), dim);


cmap = distinguishable_colors(
           maximum(L) - minimum(L) + 1,
           [RGB(1,1,1), RGB(0,0,0)], dropseed=true)

Y = sgtsnepi(pot_conn; d=dim, Y0 = Y0, max_iter = 5600);
#@show(length(Y))
show_embedding( Y, Lx )#; A = pot_conn, res = (5000, 5000) )
#scene.center = false

#save("potjans_static_wiring_network_embedding.png")
#show_embedding( Y, L ; A = pot_conn)#, res = (5000, 5000) )

sc
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
Y0 = 0.01 * randn( size(A,1), 3 );
A = pot_conn
Y = sgtsnepi(A; d = 3, Y0 = Y0, max_iter = 500);

        sc = scatter( Y[:,1], Y[:,2], Y[:,3], color = L, colormap = cmap, markersize = 5)
#display(plot(sc))
#record(sc, "sgtsnepi-animation.gif", range(0, 1, length = 24*8); framerate = 24) do ang
#  rotate_cam!( sc.figure.scene.children[1], 2*π/(24*8), 0, 0 )
#end