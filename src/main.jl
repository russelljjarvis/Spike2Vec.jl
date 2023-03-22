
function set_syn_values!(container::SpikingSynapse, new_values::CuArray{Bool})
    @set  container.fireJ = new_values
end

function set_syn_values!(container::SpikingSynapse, new_values::Array{Bool})
    @set  container.fireJ = new_values
end

function count_syn(C::Vector{SpikingSynapse},testval::SpikingNeuralNetworks.SpikingSynapse{SparseMatrixCSC})
    cnt_synapses=0
    for sparse_connections in C
        cnt_synapses+=length(sparse_connections.W.nzval)
    end    
    println("synapses to be simulated: ",cnt_synapses)
end

function count_syn(C,testval::SpikingNeuralNetworks.SpikingSynapse{CuArray})
    cnt_synapses=0
    for sparse_connections in C
        cnt_synapses+=length(sparse_connections.W)
    end    
    println("synapses to be simulated: ",cnt_synapses)
end
function sim!(P, C, dt)
    for p in P
        integrate!(p, dt)
        record!(p)
    end
    ##
    # Necessary to update the firing state of used synapses
    # Synaptic gain is updated as the states of these variables change.
    # scalar indexing slow down
    ##
    for (ind,c) in enumerate(C)
        if ind <=2 
            set_syn_values!(c, P[1].fire)
        else
            set_syn_values!(c, P[2].fire)
        end
  
        forward!(c)
        record!(c)
    end
end

function sim!(P, C; dt = 0.1ms, duration = 10ms)
    #count_syn(C,C[1])
    @showprogress for t = 0ms:dt:(duration - dt)
        sim!(P, C, Float32(dt))
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
