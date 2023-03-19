function sim!(P, C, dt)
    for p in P
        integrate!(p, Float32(dt))
        record!(p)
        #if sum(p.fire)  > 0

        #    @show(p.fire)
        #end
    end
    for (ind,c) in enumerate(C)
        if ind <3 
            c.fireJ = P[1].fire
            
        else
            c.fireJ = P[2].fire
        end
        forward!(c)#, c.param)
        record!(c)
        #if sum(c.fireJ)> 0
        #    @show(c.fireJ)
        #    @show(c.fireI)
       # end
    end
end

function sim!(P, C; dt = 0.1ms, duration = 10ms)
    for t = 0ms:dt:(duration - dt)
        sim!(P, C, dt)
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
