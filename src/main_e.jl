using Revise
using StatsBase
#using SetField
using ProgressMeter

function set_syn_values!(container::SpikingSynapse, new_values::CuArray{Bool})
    @set  container.fireJ = new_values
end

function set_syn_values!(container::SpikingSynapse, new_values::Array{Bool})
    @set  container.fireJ = new_values
end


#function sim!(P, C;conn_map=nothing, dt = 0.1ms, duration = 10ms,current_stim=nothing)

function count_syn(C::Vector{SpikingSynapse},testval::SpikeTime.SpikingSynapse{SparseMatrixCSC})
    cnt_synapses=0
    for sparse_connections in C
        cnt_synapses+=length(sparse_connections.W.nzval)
    end    
    println("synapses to be simulated: ",cnt_synapses)
end

function count_syn(C,testval::SpikeTime.SpikingSynapse{CuArray})
    cnt_synapses=0
    for sparse_connections in C
        cnt_synapses+=length(sparse_connections.W)
    end    
    println("synapses to be simulated: ",cnt_synapses)
end

function integrate_neuron!(N::Integer,v::Vector,dt::Real,ge::Vector,gi::Vector,fire::Vector{Bool},u::Vector{<:Real},tr::Vector{<:Number})
    τm = 20ms
    τe = 5ms
    τi = 10ms
    Vt = -50mV
    Vr = -60mV
    El = Vr
    tref = 10.0/dt

    @inbounds for i = 1:N

        fire[i] = false
        v[i] += dt * (ge[i] + gi[i] - (v[i] - El) + u[i]) / τm
        ge[i] += 0.01* (dt * -ge[i]) / τe
        gi[i] += 0.01* (dt * -gi[i]) / τi

        # decay conductances after application of them        
        #for seeable problem these equations may only work for a particular value of dt
        if tr[i] > 0  # check if in refractory period
            v[i] = Vr  # set voltage to reset
            tr[i] = tr[i] - dt # reduce running counter of refractory period

        end
        if tr[i]<0
            tr[i] = 0.0
        end
        if tr[i] == 0
            if v[i] >  Vt
                fire[i] = v[i] >  Vt
                tr[i] = Int(round(tref*dt))  # set refractory time
        
            end
        end
    
    end
    replace!(v, Inf=>(Vr+Vt)/2.0)
    replace!(v, NaN=>(Vr+Vt)/2.0)   
    replace!(v,-Inf16=>(Vr+Vt)/2.0)
    replace!(v,-Inf32=>(Vr+Vt)/2.0)
    replace!(v, NaN32=>(Vr+Vt)/2.0)   
    replace!(v, NaN16=>(Vr+Vt)/2.0)       
end

"""
# impinge a current proportional to weight on post synaptic cell
# membrane.
"""
function forwards_euler_weights!(post_targets::Array{Array{UInt64}},W::Array{Array{Float64}}, fireJ::Vector{Bool},g::Vector)    
    @inline for (ind,cell) in enumerate(post_targets)
        if fireJ[ind]
            for s in cell
                if W[s]>0
                    post_targets.ge[ind] = W[s] 
                else
                    post_targets.gi[ind] = W[s]
                end
            end 
        end
    end
    replace!(post_targets.gi, Inf=>0.0)
    replace!(post_targets.gi, NaN=>0.0)   
    replace!(post_targets.gi,-Inf16=>0.0)
    replace!(post_targets.gi, NaN32=>0.0) 
    replace!(post_targets.ge, Inf=>0.0)
    replace!(post_targets.ge, NaN=>0.0)   
    replace!(post_targets.ge,-Inf16=>0.0)
    replace!(post_targets.ge, NaN32=>0.0) 
      
end


function sim!(P; C=[], dt)
    for p in P
        integrate!(p, p.param, Float32(dt))
        record!(p)
    end
    for c in C
        forward!(c, c.param)
        record!(c)
    end
end


function sim!(p,dt,verbose=true;current_stim=0.0)
    for (ind,p) in enumerate(P.post_synaptic_targets)
        p.fire = Vector{Bool}([false for i in 1:length(p.fire)])
        integrate_neuron!(p.N, p.v, dt, p.ge, p.gi, p.fire, p.u, p.tr)
        record!(p)
        pre_synaptic_cell_fire_map = copy(p.fire)
        g = zeros(sizeof(pre_fire_map))
        forwards_euler_weights!(p,W,pre_synaptic_cell_fire_map,g)         
        pre_synaptic_cell_fire_map = Vector{Bool}([false for i in 1:length(pre_fire_map)])
        #record!()
    end
end 
function sim!(P, C;conn_map=nothing, dt = 1ms, duration = 10ms,current_stim=nothing)
    @showprogress for (ind,t) in enumerate(0ms:dt:(duration - dt))
        sim!(P, C, Float32(dt),conn_map,current_stim=current_stim[ind])
                ##
        # TODO Throttle maximum firing rate
        # at physiologically plausible levels
    end
end



function train!(P, C, dt, t = 0)
    for p in P
        integrate!(p, p.param, Float32(dt))
        record!(p)
    end
    for c in C
        forward!(c, c.param)
        plasticity!(c, c.param, Float32(dt), Float32(t))
        record!(c)
    end
end

function train!(P, C; dt = 0.1ms, duration = 10ms)
    for t = 0ms:dt:(duration - dt)
        train!(P, C, Float32(dt), Float32(t))
    end
end

function show_net(C)
    for sparse_connections in C
        display(sparse_connections.W)
    end    
    
end
#Base.show(io::IO, network::Vector{SpikingNeuralNetworks.IFNF{Any}}) = show_net(network)
