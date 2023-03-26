using Revise
using StatsBase
#using SetField
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
function integrate_neuron_was_working!(N::Integer,v::Vector,dt::Real,ge::Vector,gi::Vector,fire::Vector{Bool},u::Vector{<:Real},tr::Vector{<:Number})
    #= works when tr is integer countdown system=#
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
        # decay conductances after application of them
        
        #for seeable problem these equations may only work for a particular value of dt
        ge[i] += 0.1* (dt * -ge[i]) / τe
        gi[i] += 0.1* (dt * -gi[i]) / τi
    end
    @inbounds for i = 1:N
        if abs(tr[i]) > 0  # check if in refractory period
            v[i] = Vr  # set voltage to reset
            tr[i] = tr[i] - dt # reduce running counter of refractory period
        
        elseif v[i] >  Vt
            fire[i] = v[i] >  Vt
            tr[i] = Int(round(tref*dt))  # set refractory time
        end
        #@show(tr[i])
        
        #fire[i] = v[i] > Vt
        #v[i] = ifelse(fire[i], Vr, v[i])
        #@show(fire[i])

    end
    #@show(mean(v))

    #print("from inside model")

    #@show(sum(v))

end

##
# TOdo pass function into neurons,
# functions deal with if 
##

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


function forwards_euler_weights!(colptr::Vector{<:Real}, I, W, fireJ::Vector{Bool},syn_polarity,g::Vector)
    fill!(g, zero(Float32))
    @inbounds for j in 1:(length(colptr) - 1)
        if fireJ[j]
            for s in colptr[j]:(colptr[j+1] - 1)
                g[I[s]] += abs(W[s])*syn_polarity
            end 
        end
    end
    replace!(g, Inf=>0.0)
    replace!(g, NaN=>0.0)   
    replace!(g,-Inf16=>0.0)
    replace!(g, NaN32=>0.0)   
    #g
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

#using DataStructures.CircularBuffer

function sim!(P, C, dt,conn_map=nothing,verbose=true;current_stim=0.0)
    #=
    for (xx,ii) in enumerate(conn_map)
        for (ind,c) in enumerate(ii)
            pre_synaptic_population = c[1]
            p = pre_synaptic_population
            p.fire = Vector{Bool}([false for i in 1:length(p.fire)])
            integrate_neuron!(p.N, p.v, dt, p.ge, p.gi, p.fire, p.u, p.tr)
            record!(p)
        end
    end
    =#
    minus(indx, x) = setdiff(1:length(x), indx)
    for ii in conn_map
        for (ind,c) in enumerate(ii)
            pre_synaptic_population = c[1]
            post_synaptic_cell_population = c[4]
            syn_polarity = c[3]
            projmap = c[2] # pre_synaptic_cell to post synaptic_projection_map
            p = pre_synaptic_population
            
            #@set 
            p.fire = Vector{Bool}([false for i in 1:length(p.fire)])

            samp = sample(1:length(p.fire),Int(round(3*length(p.fire)/4)),replace=true)
            #@set 
            p.u[samp] .= current_stim#Vector{Float32}([current_stim for i in 1:length(samp)])
            #@show(p.u[samp])
            p.u[minus(samp, p.u)] .= 0.0

            integrate_neuron!(p.N, p.v, dt, p.ge, p.gi, p.fire, p.u, p.tr)
            record!(p)
            pre_fire_map = copy(pre_synaptic_population.fire)
            g = zeros(sizeof(pre_fire_map))
            g_weights = forwards_euler_weights!(projmap.colptr, projmap.I, projmap.W, pre_fire_map, syn_polarity,g)
            
            # Insert from front and remove from back.
            #push!(cb, c[2].delays)        # add an element to the back and overwrite front if full
            #pop!(cb)             # remove the element at the back
            #pushfirst!(cb, 10)   # add an element to the front and overwrite back if full
            #popfirst!(cb)        # remove the element at the front
            #push!(cb, g_weights)        
            # add an element to the back and overwrite front if full
            #pop!(cb)             # remove the element at the back
            #=if tr[i] <= projmap.delay  # check if in refractory period
                tr[i] = tr[i] - dt # reduce running counter of refractory period

            end
            if tr[i]<0
                tr[i] = 0.0
            end
            if tr[i] == 0
                if v[i] >  Vt
                    tr[i] = Int(round(tref*dt))  # set refractory time            
                end
            end
            =#


            
            #@set 
            c[2].g = g_weights
            for (ind,value) in enumerate(1:length(g_weights))
                if syn_polarity>= 0
                    #cnt_d = c[2].delays - dt

                    #@set 
                    post_synaptic_cell_population.ge[ind] = g_weights[ind] 
                else
                    #@set 
                    post_synaptic_cell_population.gi[ind] = g_weights[ind] 
                end
            end
            pre_fire_map = Vector{Bool}([false for i in 1:length(pre_fire_map)])
            record!(c[2])

            #=
            if syn_polarity>= 0
                @assert mean(g_weights) >=0.0
                for (ind,value) in enumerate(1:length(g_weights))
                    @assert g_weights[ind] >= 0.0
                    post_synaptic_cell_population.ge[ind] = g_weights[ind] 
                end
            elseif syn_polarity <= 0
                @assert mean(g_weights) <=0.0
                for (ind,value) in enumerate(1:length(g_weights))
                    @assert g_weights[ind] <= 0.0
                    post_synaptic_cell_population.gi[ind] = g_weights[ind] 
                end
            end
            =#
        end
    end
end 
function sim!(P, C;conn_map=nothing, dt = 0.1ms, duration = 10ms,current_stim=nothing)
    #count_syn(C,C[1])
    #delays[]
    @showprogress for (ind,t) in enumerate(0ms:dt:(duration - dt))
        sim!(P, C, Float32(dt),conn_map,current_stim=current_stim[ind])
        #for c in C
        #    c[2].cnt_d = c[2].delays - dt
        #end
                ##
        # TODO Throttle maximum firing rate
        # at physiologically plausible levels
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
