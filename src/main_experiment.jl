using Revise
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
#=
function integrate_here!(N::Integer,v::Vector,dt::Real,ge::Vector,gi::Vector,fire::Vector{Bool},u::Vector{<:Real},tr::Vector{<:Integer})
    τe, τi = 5.0,10.0
    #,0.2,0.0,-60.0,10.0)    
    #{'V_th': -55.0, 'V_reset': -75.0, 'tau_m': 10.0, 'g_L': 10.0, 'V_init': -75.0, 'E_L': -75.0, 'tref': 2.0, 'T': 400.0, 'dt': 0.1, 'range_t': array([0.000e+00, 1.000e-01, 2.000e-01, ..., 3.997e+02, 3.998e+02,
    #3.999e+02])}
    τ::Real = 8.         
    R::Real = 10.      
    θ::Real = -50.     
    vSS::Real =-55.
    v0::Real = -100. 
    tref = 10.0
    @inbounds for i = 1:N

        ##
        # maybe these should be at the bottom
        ##
        g = ge[i] + gi[i]           
        ge[i] += dt * -ge[i] / τe
        gi[i] += dt * -gi[i] / τi        
       
        #v[i] += dt * (ge[i] + gi[i] - (v[i] - El) + I[i]) / τm
        v[i] = v[i] + (g+u[i]) * R / τ
        v[i] += (dt/τ) * (-v[i] + vSS)
        if abs(tr[i]) > 0  # check if in refractory period
            v[i] = vSS  # set voltage to reset
            tr[i] = tr[i] - dt # reduce running counter of refractory period
            #print("fire lif")
        elseif v[i] >  θ
            fire[i] = v[i] >  θ
            tr[i] = Int(round(tref*dt))  # set refractory time
        end


    end
end
=#
function integrate_here2!(N::Integer,v::Vector,dt::Real,ge::Vector,gi::Vector,fire::Vector{Bool},u::Vector{<:Real},tr::Vector{<:Integer})

    τm = 20ms
    τe = 5ms
    τi = 10ms
    Vt = -50mV
    Vr = -60mV
    El = Vr
    tref = 10.0
    #print("from inside model")
    #@show(sum(ge))
    #@show(sum(u))
    #@show(sum(gi))

    @inbounds for i = 1:N
        v[i] += dt * (ge[i] + gi[i] - (v[i] - El) + u[i]) / τm
        ge[i] += dt * ge[i] / τe

        gi[i] += dt * -gi[i] / τi
    end
    @inbounds for i = 1:N

        fire[i] = v[i] > Vt
        v[i] = ifelse(fire[i], Vr, v[i])
        #@show(fire[i])

    end
    #@show(mean(v))

    #print("from inside model")

    #@show(sum(v))

end

function forwards_here!(colptr::Vector{<:Real}, I, W,fireJ::Vector{Bool},syn_polarity)#,g::Vector)
    g = zeros(sizeof(fireJ))
    fill!(g, zero(Float32))

    @inbounds for j in 1:(length(colptr) - 1)
        if fireJ[j]
            for s in colptr[j]:(colptr[j+1] - 1)
                g[I[s]] += abs(W[s])*syn_polarity
            end
            @assert sign(mean(g)) == sign(syn_polarity)
 
        end
    end
    replace!(g, Inf=>0.0)
    replace!(g, NaN=>0.0)   
    replace!(g,-Inf16=>0.0)

    g

end




function sim!(P, C, dt,conn_map,verbose=false)
    for (xx,ii) in enumerate(conn_map)
        for (ind,c) in enumerate(ii)
            pre_synaptic_population = c[1]
            p = pre_synaptic_population
            if true#xx==1
                p.u .= 10.4*randn(P[1].N)
                p.ge .= 10.4*randn(P[1].N)
                p.u = abs.(p.u)
                p.ge = abs.(p.ge)

                @assert sum(p.ge) >= 0.0
            end
            integrate_here2!(p.N, p.v, dt, p.ge, p.gi, p.fire, p.u, p.tr)
            record!(p)

            if verbose 
                @show(minimum(p.v))
                @show(maximum(p.v))

                @show(mean(p.v))
            end
                #@show(sum(p.fire))
        end
    end
    #exc_connection_map = conn_map[1]
    for ii in conn_map
        for (ind,c) in enumerate(ii)

            #for (ind,c) in enumerate(exc_connection_map)
            pre_synaptic_population = c[1]
            pre_fire_map = pre_synaptic_population.fire
            post_synaptic_cell_population = c[4]

            syn_polarity = c[3]
            projmap = c[2] # pre_synaptic_cell to post synaptic_projection_map



            g_weights = forwards_here!(projmap.colptr,projmap.I,projmap.W, pre_fire_map,syn_polarity)
            if verbose
                @show(mean(g_weights))
                @show(syn_polarity)
            end
            if syn_polarity>= 0
                if verbose
                    @show(mean(g_weights))
    
                end
                @assert mean(g_weights) >=0.0
    
                #@show(sign.(g_weights))# <=0.0

                #@assert sign.(g_weights) >=0.0
                for (ind,value) in enumerate(1:length(g_weights))
                    @assert g_weights[ind] >= 0.0
                    post_synaptic_cell_population.ge[ind] = g_weights[ind] 
                end

            elseif syn_polarity <= 0
                if verbose
                    @show(mean(g_weights))
                    @show(syn_polarity)
                    
                end
                    @assert mean(g_weights) <=0.0
                    #@show(sign.(g_weights))# <=0.0

                for (ind,value) in enumerate(1:length(g_weights))
                    @assert g_weights[ind] <= 0.0
   
                    post_synaptic_cell_population.gi[ind] = g_weights[ind] 
                end
            end
            #@show(sum(post_synaptic_cell_population.fire))
            record!(c[2])
        end
        #integrate_here2!(p.N, p.v, dt, p.ge, p.gi, p.fire, p.u, p.tr)
    end



end

    #=
    inh_connection_map = conn_map[2]
    for (ind,c) in enumerate(inh_connection_map)
        post_synaptic_cell = c[1]
        pspm = c[2] # pre_synaptic_cells_projection_map
        temp0 = pspm.fireJ
        #@show(sum(temp0))
        weights = forwards_here!(pspm.colptr,pspm.I,pspm.W, temp0, post_synaptic_cell.ge)
        #@show(weights)
        for (ind,i) in enumerate(weights)
            post_synaptic_cell.ge[ind] = i   
            #@show(post_synaptic_cell.ge[ind])             
        end
       
        record!( c[2])

        p = post_synaptic_cell
        #integrate_here2!(p.N, p.v, dt, p.ge, p.gi, p.fire, p.u, p.tr)

    end
    =#

    ##
    # Necessary to update the firing state of used synapses
    # Synaptic gain is updated as the states of these variables change.
    # scalar indexing slow down
    #=
    for c in C
        for p in P
            if typeof(p) !=  typeof(SNN.Poisson(1, 20Hz))
                if maximum(c.g) >0.0
                    @set p.ge = c.g

                end
                if maximum(c.g) <0.0
            
                    @set p.gi = c.g

                end

                integrate!_here!(p, dt)

                println("from main")

                @show(p.ge)
                @show(p.gi)

                #end
            end
        end
        forwards_here!(c)
    end
    =#

function sim!(P, C;conn_map, dt = 0.1ms, duration = 10ms)
    #count_syn(C,C[1])
    @showprogress for t = 0ms:dt:(duration - dt)
        sim!(P, C, Float32(dt),conn_map)
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
