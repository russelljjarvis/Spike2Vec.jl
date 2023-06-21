using Revise
using StatsBase
#using SetField
using ProgressMeter

function set_syn_values!(container::SpikingSynapse, new_values::CuArray{Bool})
    @set container.fireJ = new_values
end

function set_syn_values!(container::SpikingSynapse, new_values::Array{Bool})
    @set container.fireJ = new_values
end

function expected_spike_format(empty_spike_cont,nodes1,times1,maxt)
    nodes1 = [i+1 for i in nodes1]

    @inbounds for i in collect(1:1220)
        @inbounds for (neuron, t) in zip(nodes1,times1)
            if i == neuron
                push!(empty_spike_cont[Int32(i)],Float32(t)+Float32(maxt))
            end            
        end
    end
    empty_spike_cont,minimum(empty_spike_cont),maximum(empty_spike_cont)
end

function NMNIST_pre_process_spike_data(temp_container_store)
    spike_packet_lists = Vector{Any}([])
    labelsl = Vector{Any}([])
    packet_window_boundaries = Vector{Any}([])
    maxt = 0
    empty_spike_cont =  []
    @inbounds for i in collect(1:1220)
        push!(empty_spike_cont,[])
    end
    cnt = 0


    @inbounds @showprogress for (ind,s) in enumerate(temp_container_store)
        (times,labels,nodes) = (s[1],s[2],s[3]) 
        maxt = maximum(times)
        if length(times) != 0
            if cnt<25

                empty_spike_cont,min_,maxt = expected_spike_format(empty_spike_cont,nodes,times,maxt)
                maxt += maxt

                push!(labelsl,labels)
                push!(packet_window_boundaries,(min_,maxt))
                cnt+=1
            end
            #push!(spike_packet_lists,spike_packet_labeled)

        end
    end
    return empty_spike_cont,labelsl,packet_window_boundaries
end


#=



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
=#
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
    #replace!(v, Inf=>(Vr+Vt)/2.0)
    #replace!(v, NaN=>(Vr+Vt)/2.0)   
    #replace!(v,-Inf16=>(Vr+Vt)/2.0)
    #replace!(v,-Inf32=>(Vr+Vt)/2.0)
    #replace!(v, NaN32=>(Vr+Vt)/2.0)   
    #replace!(v, NaN16=>(Vr+Vt)/2.0)       
end

"""
# impinge a current proportional to weight on post synaptic cell
# membrane.
"""

#forwards_euler_weights!(post_targets::IFNF{Int64, Vector{Bool}, Vector{Float32}, Vector{Any}}, W::Vector{Any}, fireJ::Vector{Bool}, g::Vector{Float64})

function forwards_euler_weights!(post_targets::IFNF{Int64, Vector{Bool}, Vector{Float32}, Vector{Any}},W::Vector{Any}, fireJ::Vector{Bool},g::Vector{Float64})    
    @inline for (ind,cell) in enumerate(W)
        if fireJ[ind]
            for (s,w) in enumerate(cell)
                if w>0
                    post_targets.ge[ind] = w 
                else
                    post_targets.gi[ind] = w

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

function forwards_euler_weights!(post_targets,post_target_weights, fireJ::Vector{Bool},g::Vector)    
    @inline for (ind,cell) in enumerate(post_target_weights)
        if fireJ[ind]
            @inline for s in cell
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

#=
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
=#

function sim!(pp,dt)
    W = pp.post_synaptic_weights
    pp.fire = Vector{Bool}([false for i in 1:length(pp.fire)])
    integrate_neuron!(pp.N, pp.v, dt, pp.ge, pp.gi, pp.fire, pp.u, pp.tr)
    record!(pp)
    pre_synaptic_cell_fire_map = copy(pp.fire)
    g = zeros(size(pp.fire))
    forwards_euler_weights!(pp,W,pre_synaptic_cell_fire_map,g) 
    #@show(g)        
end 

function sim!(pp,dt,spike_stim_slice,external_layer_indexs)
    W = pp.post_synaptic_weights
    pp.fire = Vector{Bool}([false for i in 1:length(pp.fire)])
    if length(spike_stim_slice)!=0
        @inline for ind in external_layer_indexs[spike_stim_slice]
            pp.ge[ind] = 10.0125
        end
    end
    #println("gets here?")
    integrate_neuron!(pp.N, pp.v, dt, pp.ge, pp.gi, pp.fire, pp.u, pp.tr)
    record!(pp)
    g = zeros(size(pp.fire))
    forwards_euler_weights!(pp,W,copy(pp.fire),g)      
    #@show(pp.v)   
    #@show(pp.ge)   
    #@show(pp.gi)   

end 

#ERROR: LoadError: MethodError: no method matching 
#sim!(::IFNF{Int64, Vector{Bool}, Vector{Float32}, Vector{Any}}; dt::Float64, duration::Int64)

function sim!(P::IFNF{Int64, Vector{Bool}, Vector{Float32}}; dt::Real = 1ms, duration::Real = 10ms)#;current_stim=nothing)
    @inline  for _ in 0:dt:duration
        sim!(P, dt)
        # TODO Throttle maximum firing rate
        # at physiologically plausible levels
    end
end


function sim!(P::IFNF{Int64, Vector{Bool}, Vector{Float32}}; dt::Real = 1ms, duration::Real = 10ms,spike_stim,external_layer_indexs,onset)#;current_stim=nothing)
    prevt=0.0
   @showprogress for t in 0:dt:duration
       #@show(t,onset)
       if t>=onset
            #println("gets here?")

            spike_stim_slice = divide_epoch(spike_stim,prevt,t)
        
            sim!(P, dt,spike_stim_slice,external_layer_indexs)
        else
            sim!(P, dt)
        end
        prevt=t
        # TODO Throttle maximum firing rate
        # at physiologically plausible levels
    end
end


function divide_epoch(vector_times::AbstractVector,start::Real,stop::Real)
    spike_cell_id=Vector{UInt32}([])
    #@show(vector_times)
    @inbounds for (n,tvec) in enumerate(vector_times)
        for t in tvec
            
            if start<=t && t<=stop
                #@show(start,t,stop)
                #@show(start<=t && t<=stop)
                #print("hit??",n)
                push!(spike_cell_id,n)
            end
        end
    end
    #@show(spike_cell_id)
    spike_cell_id
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
