using ProgressMeter

function set_syn_values!(container::SpikingSynapse, new_values::CuArray{Bool})
    container.fireJ[] = new_values[] # "reassign" different array to Ref
end

function count_syn(C)
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
    count_syn(C)
    @showprogress for t = 0ms:dt:(duration - dt)
        sim!(P, C, Float32(dt))
                ##
        # Throttle maximum firing rate
        ##
        #for p in P

        #    if sum(p.fire)>10
        #        @show(sum(p.fire))
        #        temp = zeros(Bool, p.N)
        #        set_syn_values!(P[1].fire,temp)
        #    end
        #end
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
#Base.show(io::IO, ::MIME"text/plain") =
#    print(io, """LIF:
#                     voltage: $(neuron.v)
#                     current: $(neuron.u)
#                     τm:      $(neuron.τm)
#                     Vr    :  $(neuron.Vr)
#                     R:       $(neuron.R)""")

function show_net(C)
    for sparse_connections in C
        display(sparse_connections.W)
    end    
    
end
Base.show(io::IO, network::Vector{SpikingNeuralNetworks.IFNF{UInt64, Vector{Bool}, Vector{Float16}}}) = show_net(network)
#show_net(C)
