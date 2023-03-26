using Plots
using SparseArrays
using SpikingNeuralNetworks
SNN.@load_units
using Test
using Revise

function SpikingSynapseReliable(pre::SpikingNeuralNetworks.IFNF, post::SpikingNeuralNetworks.IFNF,sim_type::Any; σ = 0.0)
    w = σ * sparse(ones(post.N, pre.N)) 
    rowptr, colptr, I, J, index,V = SNN.dsparse(w,sim_type)
    g::typeof(sim_type) = zeros(size(rowptr))#(w[:]).*sign.(minimum(w[:,1]))   
    V::typeof(sim_type) = convert(typeof(sim_type),V)
    SNN.SpikingSynapse(rowptr,colptr,I,J,index,V,g,pre,post)
end

pop_size::UInt64=1
sim_type = Vector{Float32}(zeros(1))
u1 = Float32[10.052929259 for i in 50:0.1ms:150ms]

ge = Float32[0.4*rand() for i in 50:0.1ms:150ms]
ge = abs.(ge)

#p.u .= 10.4*randn(P[1].N)
#ge .= 10.4*randn(P[1].N)
#p.u = abs.(p.u)
E = SNN.IFNF(pop_size,sim_type,u1)
E.ge = ge
I = SNN.IFNF(pop_size,sim_type)


##
# two layer ff connections only
##
EE = SpikingSynapseReliable(E, I,sim_type; σ = 1000*0.27/1)
EI = SpikingSynapseReliable(E, I,sim_type; σ = 1000*0.27/1)
IE = SpikingSynapseReliable(I, E,sim_type; σ = -2*4.5/1)
P = [I, E]

C = [EI, IE]#,EE]
SNN.monitor([C], [:g])
SNN.monitor([I,E], [:v])
SNN.monitor([I,E], [:fire])


#inh_connection_map=[(E,"can be infered from 1 and 4",1,I)]
#exc_connection_map=[(I,"can be infered from 1 and 4",IE,-1,E)]

inh_connection_map=[(E,EI,1,I)]
exc_connection_map=[(I,IE,-1,E)]
#EE_connection_map=[(E,EE,1,E)]

connection_map = [exc_connection_map,inh_connection_map]#,EE_connection_map]
SNN.sim!(P, C;conn_map= connection_map, duration = 1.25second)
    


#SNN.vecplot([C], :g) |> display
#SNN.raster([E,I]) |> display
SNN.vecplot([E], :v) |> display
SNN.vecplot([I], :v) |> display

