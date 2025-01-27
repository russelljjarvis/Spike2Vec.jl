"""
Get the mean Interspike Interval and all other ISIs as a per cell dictionary, and as a "bag of words",pooled collection.
"""
function get_isis(times,nodes)
    spike_dict = Dict()
    isi_dict = Dict()
    all_isis = []
    for n in unique(nodes)
        spike_dict[n] = []
	isi_dict[n] = []
    end
    for (st,n) in zip(times,nodes)
        append!(spike_dict[n],st)
    end
    for (k,v) in pairs(spike_dict)
        time_old = 0
        for time in spike_dict[k][1:end-1]
            isi = time - time_old
	    append!(isi_dict[n],isi)
            append!(all_isis,isi)
            time_old = time
        end
    end
    (StatsBase.mean(all_isis),isi_dict,all_isis)
end



"""
Divide up epochs, this would normally be called in a loop
For example:
```julia    
    for (ind,toi) in enumerate(end_window)
        self_distances = Array{Float32}(zeros(maximum(nodes)+1))
        sw = start_windows[ind]
        neuron0 = divide_epoch(nodes,times,sw,toi)    
    end
```
"""
function divide_epoch(nodes,times,sw,toi)
    t1=[]
    n1=[]
    t0=[]
    n0=[]
    @assert sw< toi
    third = toi-sw
    @assert third==300
    for (n,t) in zip(nodes,times)
        if sw<=t && t<toi
            append!(t0,t-sw)
            append!(n0,n)            
        elseif t>=toi && t<=toi+third
	    @assert t-toi>=0
            append!(n1,n)
        end
    end
    neuron0 =  Array{}([Float32[] for i in 1:maximum(nodes)+1])
    for (neuron,t) in zip(n0,t0)
        append!(neuron0[neuron],t) 
    end
    neuron0
end
"""
spike2vector algorithm to demonstrate object recognition in spike trains. 
Neural activity in a window can be represented as a coordinate in high-dimensional space; when a network enacts object recognition. 
When a cortical neural network enacts object recognition encoded memories guide the network to replay neural states that correspond to familiar scenes. 
The trajectory of the network between familiar and unfamiliar states can be tracked using a high-dimensional coordinate scheme. 
A network’s ability to revisit an encoded coordinate is testable, and so a spike2vector test of object recognition could be construed as a formal hypothesis test.
"""
function get_vector_coords(neuron0::Vector{Vector{Float32}}, neuron1::Vector{Vector{Float32}}, self_distances::Vector{Float32})
    for (ind,(n0_,n1_)) in enumerate(zip(neuron0,neuron1))        
        if length(n0_) != 0 && length(n1_) != 0
            pooledspikes = vcat(n0_,n1_)
            maxt = maximum(sort!(unique(pooledspikes)))
            t1_ = sort(unique(n0_))
            t0_ = sort(unique(n1_))
            t, S = SPIKE_distance_profile(t0_,t1_;t0=0,tf = maxt)
            self_distances[ind]=sum(S)
        else
            self_distances[ind]=0
        end
    end
    return self_distances
end
#=
function looped!(times,t0,spk_counts,segment_length,temp)
    doonce = LinRange(0.0, segment_length, temp)[:]
    for (neuron, t) in enumerate(t0)
        times[neuron] = doonce
    end
end
function surrogate_to_uniform(times_,segment_length)
    times =  Array{}([Float32[] for i in 1:length(times_)])
    spk_counts = []
    for (neuron, t) in enumerate(times_)
        append!(spk_counts,length(t))
    end
    temp = 4
    looped!(times,times_,spk_counts,segment_length,temp)
    return times

end
=#

function get_plot(;plot_time_chunked_train=false)
    times,nodes = get_()
    division_size = 10
    step_size = maximum(times)/division_size
    end_window = collect(step_size:step_size:step_size*division_size)
    spike_distance_size = length(end_window)
    start_windows = collect(0:step_size:step_size*division_size-1)
    mat_of_distances = zeros(spike_distance_size,maximum(unique(nodes))+1)
    n0ref = divide_epoch(nodes,times,start_windows[3],end_window[3])
    segment_length = end_window[3] - start_windows[3]
    t0ref = surrogate_to_uniform(n0ref,segment_length)
	PP = []
    
    for (ind,toi) in enumerate(end_window)
        self_distances = Array{Float32}(zeros(maximum(nodes)+1))
        sw = start_windows[ind]
        neuron0 = divide_epoch(nodes,times,sw,toi)    
        self_distances = get_vector_coords(neuron0,t0ref,self_distances)
        mat_of_distances[ind,:] = self_distances
    end
    cs1 = ColorScheme(distinguishable_colors(spike_distance_size, transform=protanopic))
    if plot_time_chunked_train
	    for (ind,toi) in enumerate(iters)
		sw = start_windows[ind]
		(t0,n0,t1,n1) = divide_epoch(nodes,times,sw,toi)
		self_distances = get_vector_coords(nodes,t0ref,n0ref,t0,n0)
		mat_of_distances[ind,:] = self_distances
		o1 = HeatMap(zip(minimum(t0):maximum(t0)/1000.0:maximum(t0),minimum(n0):maximum(n0/1000.0):maximum(n0)) )
		fit!(o1,zip(t0,convert(Vector{Float64},n0)))
		p0 = plot(o1, marginals=false, legend=false) 
		o2 = HeatMap(zip(minimum(t0ref):maximum(t0ref)/1000.0:maximum(t0ref),minimum(n0ref):maximum(n0ref/1000.0):maximum(n0ref)) )
		fit!(o2,zip(t0ref,convert(Vector{Float64},n0ref)))
		p1 = plot(o2, marginals=false, legend=false)
		#push!(PP,p0)
		#push!(PP,p1)
	    end
	    normalize!(mat_of_distances[:,:])
	    p = plot()#title = "Plot 1")
    end
    for (ind,_) in enumerate(eachrow(mat_of_distances))
        temp = (mat_of_distances[ind,:].- mean(mat_of_distances[ind,:]))./std(mat_of_distances[ind,:])
        n = length(temp)
        θ = LinRange(0, 2pi, n)
	if plot_time_chunked_train		
		#if ind==1
		#    p = plot(θ, temp, proj=:polar,color=cs1[ind]) |>display#, layout = length(mat_of_distances))
		#@else 
		plot(θ,temp, proj=:polar,color=cs1[ind]) |>display
		#end
		#plot(θ,mat_of_distances[ind,:], proj=:polar,color=cs1[ind])
		#savefig("radar_categories_$ind.png")
		#ind+=1 
	end
    end

    prev = mat_of_distances[1,:].- mean(mat_of_distances[1,:])./std(mat_of_distances[1,:])
 
    n = length(prev)
    θ = LinRange(0, 2pi, n)
    p = plot(θ,prev, proj=:polar,color=cs1[1]) |>display

    for (ind,_) in enumerate(eachrow(mat_of_distances))
        
        temp = mat_of_distances[ind,:].- mean(mat_of_distances[ind,:])./std(mat_of_distances[ind,:])
        diff = temp.-prev
        #n = length(prev)
        #θ = LinRange(0, 2pi, n)
        p = plot(diff) |>display
        prev = temp

        #if ind==1
            
        
        #    p = plot(θ, temp, proj=:polar,color=cs1[ind]) |>display#, layout = length(mat_of_distances))
        #@else 
        #end
        #plot(θ,mat_of_distances[ind,:], proj=:polar,color=cs1[ind])
        #savefig("radar_categories_$ind.png")
        #ind+=1 
    end
    #current() |>display
    for (ind,self_distances) in enumerate(eachrow(mat_of_distances))
        if ind>1
            @show(angle(mat_of_distances[ind,:],mat_of_distances[1,:]))
        end

    end

    for (ind,self_distances) in enumerate(eachrow(mat_of_distances))
        if ind>1
            @show(angle(mat_of_distances[ind,:],mat_of_distances[ind-1,:]))
        end

        #ind+=1 
    end

    plot!(plots_)|>display

    if 1==0
        @time o = IndexedPartition(Float64, KHist(100), 100)
        @time fit!(o,zip(convert(Vector{Float64},nodes),times))
        @time plot(o) |> display
    end
    return mat_of_distances,cs1,PP,end_window
end
@time mat_of_distances,cs1,PP,end_window = get_plot()
function dont_do()
  n = length(PP)
  plot(PP...; marginals=false, legend=false, layout = (length(PP))) |> display

  println("compiled times")
  @time get_plot()
  o = fit!(IndexedPartition(Float64, KHist(40), 40), zip(x, y))
  using Plots

  xy = zip(randn(10^6), randn(10^6))
  xy = zip(1 .+ randn(10^6) ./ 10, randn(10^6))
  0:Float64(maximum(nodes))/10.0:maximum(nodes)), 0.0:maximum(times)/100.0:maximum(times), xy)
  o1 = fit!(HeatMap(zip(1:maximum(nodes)/10.0:maximum(nodes)),0.0:maximum(times)/100.0:maximum(times)), xy)
  plot(o)
  
  plot(o1)
end
  function getstuff(times,nodes,o,o1)
      tempx = []
      tempy = []
      done = 0
      for ci=1:length(nodes) 
          push!(tempx, times[ci])
          push!(tempy, ci)
          if ci%10000==0
              convtempy = convert(Vector{Float64},tempy)
              store_conv_tempy = zip(convtempy, tempx)
              for (i,j) in store_conv_tempy
                  fit!(o,zip(i,j))
                  fit!(o1,zip(i,j))
              end
              plot(o1, marginals=false, legend=true) |> display
              plot(o) |> display

          end
      end
      return tempx,tempy
  end
  println("done")
  tempx,tempy = getstuff(times,nodes,o,o1)
  println("done")
  


  
  o = CCIPCA(2, length(unique(nodes))) 
  fit!(o, zip(xs, ys))
  OnlineStats.transform(o, zip(xs, ys))     # Project u3 into PCA space fitted to u1 and u2 but don't change the projection
  u4 = rand(10)
  OnlineStats.fittransform!(o, zip(xs, ys)) # Fit u4 and then project u4 into the space
  sort!(o)                         # Sort from high to low eigenvalues
  @show(OnlineStats.relativevariances(o))   
  
end
