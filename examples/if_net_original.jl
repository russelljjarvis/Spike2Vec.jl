using Plots
using SpikingNeuralNetworks
using OnlineStats
using SparseArrays
import Plots.heatmap
#import Plots.density
#import UnicodePlots.DotCanvas
#import UnicodePlots.BrailleCanvas
#using LinearAlgebra
#using Makie
#unicodeplots()
#unicodeplots();
#using Plots

#using ImageView
SNN.@load_units
using CUDA
#print("\u001B[?25h") # visible cursor
#@allowscalar
function main(current)
    pop_sizes=1220
    current = CUDA.CuArray{Float32}([0.001 for i in 0:1ms:1.25second])


    E = SNN.IFNF(;N=pop_sizes, I=current)
    I = SNN.IFNF(;N=pop_sizes,pop_indexs=pop_sizes,I=current)#,PT=::CuArray{Float32})#;El = -49mV))

    EE = SNN.SpikingSynapse(E, E, :ge; σ = 60*0.27/1, p = 0.02)
    EI = SNN.SpikingSynapse(E, I, :ge; σ = 60*0.27/1, p = 0.02)
    IE = SNN.SpikingSynapse(I, E, :gi; σ = -20*4.5/1, p = 0.03)
    II = SNN.SpikingSynapse(I, I, :gi; σ = -20*4.5/1, p = 0.03)
    P = [E, I]
    C = [EE, EI, IE, II]


    #=
    cnt_synapses=0
    weights_for_movie=sparse(C[1].I,C[1].J, C[1].index)
    for sparse_connections in C
        #display(C.g)
        cnt_synapses+=length(sparse_connections.W)
        sp=sparse(sparse_connections.I,sparse_connections.J, sparse_connections.index)
        weights_for_movie+=sp
    end
    println("synapses simulated: ",cnt_synapses)
    =#
    SNN.monitor([E,I], [:fire])
    @time SNN.sim!(P, C; duration = 1.5second)
    display(SNN.raster([E,I]))
    print("simulation done !")
    (times,nodes) = SNN.get_trains([E,I])

    Matrix(weights_for_movie),pop_sizes,E,I,times,nodes

end

function analyse_results1()
    weights_for_movie,pop_sizes,_,_,_,nodes = main()
    temp_rows = Vector{Int64}([])
    print("\33[2J")
    wm = spzeros(pop_sizes,pop_sizes)
    wm,nodes,temp_rows,weights_for_movie
end


function analyse_results(current)
    weights_for_movie,pop_sizes,E,I,times,nodes = main(current)
    temp_rows = Vector{Int64}([])
    print("\33[2J")
    w = zeros(pop_sizes,pop_sizes)#.*0.00001
    wm = spzeros(pop_sizes,pop_sizes)      
    for (ind,n) in enumerate(nodes)
        println("\33[H")
        append!(temp_rows,n)
        if ind%20==0    
            w[:,temp_rows] .= weights_for_movie[:,temp_rows] 
            pst=[]
            for (x,tt) in enumerate(w[:,temp_rows])
                tt_ = tt +1.0*exp(tt)
                if tt_== Inf
                    tt_ = tt +tt

                    #tt_ =tt
                end
                #end
                append!(pst,tt_)
                
            end
            wm[:,temp_rows] = pst

            for (x,y,v) in zip(findnz(wm)...)
                w[x,y] = wm[x,y]
            end
            temp_rows=[]
            replace!(w, -Inf=>0.0)
            replace!(wm, Inf=>0.0)
            
            tit = times[ind]
            display(heatmap(w,normalizee=:pdf, interpolate = true,color=:viridis,title="time = $tit")) #|>display
            for (x,y,v) in zip(findnz(wm)...)
                if v<0.0
                    wm[x,y] = 0.0
                end
                if v>0.0

                    if abs(wm[x,y]-exp(v))== -Inf
                        wm[x,y] = abs(wm[x,y]-v)
                    
                    else
                        wm[x,y] = abs(wm[x,y]-exp(v))

                    end

                end
                # = wm[x,y]
            end
        end
    end
    times,nodes,E,I
end   
weights_for_movie,pop_sizes,E,I,times,nodes = main(current)

#times,nodes,E,I = analyse_results(current)
#o1 = HeatMap(zip(minimum(times):maximum(times)/1000.0:maximum(times),minimum(nodes):1:maximum(nodes)) )
#fit!(o1,zip(times,convert(Vector{Float64},nodes)))
#plot(o1, marginals=false, legend=true) |>display 
SNN.raster([E,I])#, [:fire])


#=

    #https://github.com/cesaraustralia/DynamicGrids.jl/blob/4680da6a9dd1bcb314a91376a9e1404ba7e237b7/src/outputs/gif.jl#L45
    #=
    function savegif(filename::String, o::ImageOutput; fps=fps(o), kw...)
        length(o) == 1 && @warn "The output has length 1: the saved gif will be a single image"
        #data = SimData(o, ruleset)
        imsize = size(first(grids(data)))
        gif = Array{ARGB32}(undef, imsize..., length(o))
        foreach(firstindex(o):lastindex(o)) do f
            #@set! data.currentframe = f
            render!(view(gif, :, :, f), renderer(o), o, data, o[f])
        end
        #FileIO.save(File{format"GIF"}(filename), gif; fps=fps, kw...)
        return filename
    end
    =#
    #Plots.savefig("default_heatmap.png")

    #SNN.raster(P)
    SNN.monitor([E, I], [:fire])
    SNN.monitor([C], [:W])

    SNN.train!(P, C; duration = 2second)
    print("\33[2J")

    for (ind,(t,n)) in enumerate(zip(times,nodes))
        wm = spzeros(pop_sizes,pop_sizes)
        append!(temp_rows,n)
        append!(temp_times,t)
        println("\33[H")

        #print("\e[0;0H\e[2J")
        t_old=t
        if ind%100==0
            wm[temp_rows,:] .= weights_for_movie[temp_rows,:] *100.0
            heatmap(wm,color=:viridis, canvas=DotCanvas, height=10000, width=10000,normalizee=:pdf,title="trained stdp") |>display
            temp_rows=[]
            #temp_times=[]
        end
    end
    (times,nodes) = SNN.get_trains([E,I])
    @time o1 = HeatMap(zip(minimum(times):maximum(times)/100.0:maximum(times),minimum(nodes):maximum(nodes/100.0):maximum(nodes)) )
    @time fit!(o1,zip(times,convert(Vector{Float64},nodes)))
    plot(o1, marginals=false, legend=true) #|>display 
    #Plots.savefig("default_heatmap_train.png")
    #W = SNN.get_weights()
    nodes,times,W
end
nodes,times,W = main()
=#