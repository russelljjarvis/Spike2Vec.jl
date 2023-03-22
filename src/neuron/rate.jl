

struct Rate{X<:AbstractArray}
    N::Int32 # = 100
    x::X # = # 0.5randn(N)
    r::X # = #  tanh.(x)
    g::X # = #zeros(N)
    I::X # = zeros(N)
    records::Dict # = Dict()
    
    function Rate(N)
        x = 0.5randn(N)
        r = tanh.(x)
        g = zeros(N)
        I = zeros(N)  
        dict = Dict()  
        new{typeof(x)}(N,x,r,g,I,dict)
    end
end

"""
[Rate Neuron](https://neuronaldynamics.epfl.ch/online/Ch15.S3.html)
"""
Rate

function integrate!(p::Rate, dt::Float32)
    @unpack N, x, r, g, I = p
    @inbounds for i = 1:N
        x[i] += dt * (-x[i] + g[i] + I[i])
        r[i] = tanh(x[i]) #max(0, x[i])
    end
end
