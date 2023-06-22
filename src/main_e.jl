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

function NMNIST_pre_process_spike_data(temp_container_store;duration=25)
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
            if cnt<duration

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
function integrate_neuron!(N::Integer,dt::Real,pp)
    τm = 20ms
    τe = 5ms
    τi = 10ms
    Vt = -50mV
    Vr = -60mV
    El = Vr
    tref = 10.0/dt

    @inbounds for i = 1:N

        pp.v[i] += dt * (pp.ge[i] + pp.gi[i] - (pp.v[i] - El) + pp.u[i]) / τm

        pp.ge[i] += (dt * -pp.ge[i]) / τe
        pp.gi[i] += (dt * -pp.gi[i]) / τi

        # decay conductances after application of them        
        #for seeable problem these equations may only work for a particular value of dt
        if pp.tr[i] > 0  # check if in refractory period
            pp.v[i] = Vr  # set voltage to reset
            pp.tr[i] = pp.tr[i] - dt # reduce running counter of refractory period

        end
        if pp.tr[i]<0
            pp.tr[i] = 0.0
        end
        if pp.tr[i] == 0
            if pp.v[i] >  Vt
                pp.fire[i] = pp.v[i] >  Vt
                pp.tr[i] = Int(round(tref*dt))  # set refractory time
        
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

function forwards_euler_weights!(post_targets::IFNF{Int64, Vector{Bool}, Vector{Float32}, Vector{Vector{Any}}},W::Vector{Vector{Any}})    
    @inline for (ind,cell) in enumerate(W)
        if post_targets.fire[ind]
            @inline for (s,w) in enumerate(cell)

                if w>0

                    post_targets.ge[s] = w*12.5
                else
                    post_targets.gi[s] = w*12.5 
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


function forwards_euler_weightsSDTP!(pop::IFNF{Int64, Vector{Bool}, Vector{Float32}, Vector{Vector{Any}}},W::Vector{Vector{Any}}, t::Float32))    
    τpre = 20ms
    τpost  = 20ms
    Wmax  = 0.01
    ΔApre  = 0.01 * Wmax
    ΔApost  = -ΔApre * τpre / τpost * 1.05
    @inline for (ind,cell) in enumerate(W)
        if pop.fire[ind]
            @inline for (post_syn,w) in enumerate(cell)

                Apre[s] *= exp32(- (t - tpre[post_syn]) / τpre)
                Apost[s] *= exp32(- (t - tpost[post_syn]) / τpost)
                Apre[s] += ΔApre
                tpre[s] = t
                W[ind,post_syn] = clamp(W[ind,post_syn] + Apost[post_syn], 0f0, Wmax)
            end
        end
    end
    WX = sparse(length(W),)
    @inline for (ind,cell) in enumerate(W)
        if pop.fire[ind]
            @inline for (post_syn,w) in enumerate(cell)

            #for st in rowptr[i]:(rowptr[i+1] - 1)
            s = index[st]
            Apre[pre_syn] *= exp32(- (t - tpre[pre_syn]) / τpre)
            Apost[pre_syn] *= exp32(- (t - tpost[pre_syn]) / τpost)
            Apost[pre_syn] += ΔApost
            tpost[pre_syn] = t
            W[pre_syn,ind] = clamp(W[pre_syn,ind] + Apost[pre_syn], 0f0, Wmax)
        end
    end
      
end

function sim!(pp,dt)
    W = pp.post_synaptic_weights
    pp.fire = Vector{Bool}([false for i in 1:length(pp.fire)])
    integrate_neuron!(pp.N, dt ,pp)
    record!(pp)
    forwards_euler_weights!(pp,W) 
end 

function sim!(pp,dt,spike_stim_slice,external_layer_indexs)
    W = pp.post_synaptic_weights
    pp.fire = Vector{Bool}([false for i in 1:length(pp.fire)])
    if length(spike_stim_slice)!=0
        @inline for ind in external_layer_indexs[spike_stim_slice]
            pp.ge[ind] = 50.9125
        end
    end
    integrate_neuron!(pp.N, dt ,pp)
    record!(pp)
    forwards_euler_weights!(pp,W)      
end 


function sim!(P::IFNF{Int64, Vector{Bool}, Vector{Float32}}; dt::Real = 1ms, duration::Real = 10ms)#;current_stim=nothing)
    @inline  for _ in 0:dt:duration
        sim!(P, dt)
        # TODO Throttle maximum firing rate
        # at physiologically plausible levels
    end
end


function sim!(P::IFNF{Int64, Vector{Bool}, Vector{Float32}}; dt::Real = 1ms, duration::Real = 10ms,spike_stim,external_layer_indexs,onset)#;current_stim=nothing)
   cnt = 1
   cnt_stim = 1
   stim_length=length(spike_stim.times_versus_neuron_activations)
   important_length=length(spike_stim.indexs_of_times)

   @showprogress for t in 0:dt:duration
        if t>=onset && cnt<important_length-1    
            
            if spike_stim.indexs_of_times[cnt] 
                spike_stim_slice = spike_stim.times_versus_neuron_activations[cnt_stim]
                cnt_stim+=1                        
                sim!(P, dt,spike_stim_slice,external_layer_indexs)
            else
                sim!(P, dt)
            end
        
        else
            sim!(P, dt)
        end
        cnt+=1
    end
end
struct stim_container{P,R}
    times_versus_neuron_activations::P # row pointer of sparse W
    indexs_of_times::R      # postsynaptic index of W
    function stim_container(times_versus_neuron_activations,indexs_of_times)
        new{typeof(times_versus_neuron_activations),typeof(indexs_of_times)}(times_versus_neuron_activations,indexs_of_times)
    end
end

function restructure_stim(;dt::Real = 1ms, duration::Real = 10ms,spike_stim=spike_stim,onset=onset)#;current_stim=nothing)
    prevt=0.0
    times_ = []
    indexs_of_times = []
    times_versus_neuron_activations = []
    cnt=1
    @showprogress for t in 0:dt:duration
        if t>=onset
            spike_stim_slice = divide_epoch(spike_stim,prevt,t)
            if length(spike_stim_slice)!=0
                push!(times_versus_neuron_activations,spike_stim_slice)
                push!(times_,t)
                push!(indexs_of_times,true)
            else
                push!(indexs_of_times,false)
            end
        else
            push!(indexs_of_times,false)
        end
       prevt=t
       cnt+=1
    end
    stim_container(times_versus_neuron_activations,times_,indexs_of_times)
end

function divide_epoch(vector_times::AbstractVector,start::Real,stop::Real)
    spike_cell_id=Vector{UInt32}([])
    @inbounds for (n,tvec) in enumerate(vector_times)
        for t in tvec
            
            if start<=t && t<=stop
                push!(spike_cell_id,n)
            end
        end
    end
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
