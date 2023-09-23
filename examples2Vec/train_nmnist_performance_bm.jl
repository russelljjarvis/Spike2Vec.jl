using PyCall
#using PythonCall
using Revise
using Odesa
using Random
using ProgressMeter
using JLD
using NumPyArrays
using DrWatson
using LoopVectorization
using Plots


function build_data_set_native(events,storage,cnt,input_shape,l_change_cnt,l_old)
    xx = Vector{Int32}([])
    yy = Vector{Int32}([])
    tts = Vector{Float32}([])
    polarity = Vector{Int8}([])
    label = Vector{Int32}([])
    A = zeros((35,35))
    I = LinearIndices(A)
    pop_stimulation= Vector{Int32}([])#Vector{UInt32}([])
    @inline for (ind_,ev) in enumerate(events)      
        cnt+=1
        (x,y,ts,p,l) = ev
        push!(pop_stimulation,Int32(I[CartesianIndex(convert(Int32,x),convert(Int32,y))]))
        push!(xx,convert(Int32,x))
        push!(yy,convert(Int32,y))
        ts = Float32(convert(Float32,ts)/1000.0)
        push!(tts,ts)
        push!(polarity,convert(Int8,p))
        l = convert(Int32,l)
        push!(label,l)
    end
    did_it_exec::Tuple{Vector{Int32}, Vector{Int32}, Vector{Float32}, Vector{Int8}, Vector{Int32}, Vector{Any}} = (xx,yy,tts,polarity,label,pop_stimulation)
    (cnt,did_it_exec,l_change_cnt,l_old)
end


function bds!()

    pushfirst!(PyVector(pyimport("sys")."path"), "")
    nmnist_module = pyimport("batch_nmnist_motions")
    dataset::PyObject = nmnist_module.NMNIST("./")
    training_order = 0:dataset.get_count()-1
    #storage = Array{Array}([])
    storage::Array{Tuple{Vector{Int32}, Vector{Int32}, Vector{Float32}, Vector{Int8}, Vector{Int32}, Vector{Any}}} = []
    storage = []
    input_shape = dataset.get_element_dimensions()
    cnt = 0
    l_change_cnt = 0
    l_old = 4
    @inbounds @showprogress for batch in 1:200:length(training_order)
        events = dataset.get_dataset_item(training_order[batch:batch+1])
        cnt,did_it_exec,l_change_cnt,l_old = build_data_set_native(events,storage,cnt,input_shape,l_change_cnt,l_old)
        @save "part_mnmist_$cnt.jld" did_it_exec
    end
end


bds!()

@load "all_mnmist.jld" storage
(x,y,times,p,l,nodes) = (storage[1][1],storage[1][2],storage[1][3],storage[1][4],storage[1][5],storage[1][6])

for (ind,s) in enumerate(storage)
    (x,y,times,p,l,nodes) = (storage[s][1],storage[s][2],storage[s][3],storage[s][4],storage[s][5],storage[s][6])
    @show(unique(l)[1])#,ind)
end


println("rendering slow")
@show(length(unique(l)))
display(Plots.scatter(times,nodes,markersize=0.1))
#=
better_type_info = Vector{Tuple{Int32,Int32,Int32,Int32,Int32}}(zeros(length(storage)))
for (cnt,(xx,yy,tts,pp,ll)) in storage
    @show(typeof(xx),yy,tts,pp,ll)
    better_type_info[cnt] = (xx,yy,tts,pp,ll)
end


"""
Let us say we want to know the average cost of just one batch conversion. timing profile_time once will do that.
"""


function profile_time(dataset,batch,events_cache,training_order)        
    events_ = dataset.get_dataset_item(training_order[batch:batch+100-1])
    @show(pytypeof(events_))
    #x_ = reinterpret(UInt32, events_[1])
    #e1 = pyconvert(Vector{UInt32}, events_[1])
    #@show(typeof(e1))

    """
    x = Vector{Int32}()
    y = Vector{Int32}()
    ts = Vector{Int32}()
    p = Vector{Int32}()
    label = Vector{Int32}()

    for e in events
        push!(x,trunc(Int32,e[1]))
        push!(y,trunc(Int32,e[2]))
        push!(ts,trunc(Int32,e[3]))
        push!(p,trunc(Int32,e[4]))
        push!(label,trunc(Int32,e[5]))

    end
    @show(length(x))
    @show(length(events))
    events_ = (x,y,ts,p,label) 
    """
    #@show(typeof(events_cache))
    #@show(length(events_cache[1]))
    return events_
end
"""
Pre-convert NMNMIST to Julia native types, this ensures that performance evaluation of convuliotional Odessa is not distorted by type conversion and language call artifacts.
"""

function pre_conv(dataset::PyObject)
    events_cache = []
    input_shape = dataset.get_element_dimensions()
    training_order = shuffle(0:dataset.get_count()-1)
    cnt = 1
    @inbounds @showprogress for batch in 1:100:length(training_order)
        batch = profile_time(dataset,batch,events_cache,training_order)
        
        append!(events_cache,batch)
        #batch = events_cache[cnt]
        @show(length(batch[1]))
        #x = [convert(UInt32,i) for i in batch[1]]
        #,y,ts,p,label 
        #, convert(Vector,batch[2]) , convert(Vector,batch[3]) , convert(Vector,batch[4]) , convert(Vector,batch[5])
        #@show(length(x))
        cnt+=1
        #events_cache[cnt] = x,y,ts,p,label
       # @show(length(events_cache[1]))
    end
    #@inbounds @showprogress for batch in events_cache
    #    @show(length(events_cache[1]))
    #end

    (events_cache,input_shape)
end






"""
Make it easier for compiler optimizer to garbage collect everyting that is not the model
Model is consumed by the subsequent Odessa training loop.    
"""
function prealloc(input_shape::Tuple{Int64, Int64, Int64})
    nClass = 10
    input_rows::Int32          = 360
    input_cols::Int32          = 360

    first_layer_input_rows::Int32 = input_rows
    first_layer_input_cols::Int32 = input_cols
    first_layer_input_ch::Int32 = convert(Int32,input_shape[3])
    first_layer_cxt_diam::Int32 = 5
    first_layer_pool_diam::Int32 = 3

    # div(first_layer_input_cols, first_layer_pool_diam) + 1
    second_layer_input_rows::Int32 = div(first_layer_input_rows, first_layer_pool_diam) + 1
    second_layer_input_cols::Int32 = div(first_layer_input_cols, first_layer_pool_diam) + 1

    second_layer_cxt_diam::Int32 = 5
    second_layer_pool_diam::Int32 = 3

    output_layer_input_rows::Int32 = div(second_layer_input_rows, second_layer_pool_diam) + 1
    output_layer_input_cols::Int32 = div(second_layer_input_cols, second_layer_pool_diam) + 1

    output_layer_cxt_rows::Int32 = 5
    output_layer_cxt_cols::Int32 = 5

    first_layer_out_cxt_rows::Int32 = second_layer_cxt_diam
    first_layer_out_cxt_cols::Int32 = second_layer_cxt_diam
    first_layer_n_neurons::Int32 = 30
    first_layer_eta::Float32 = 0.0001
    first_layer_threshold_open::Float32 = 0.001
    first_layer_tau::Float32 = 1e3
    first_layer_trace_tau::Float32 = 2*1e3
    first_layer_thresh_eta::Float32 = first_layer_eta

    first_layer = Odesa.ConvOdesa.Conv(first_layer_input_rows,
                                    first_layer_input_cols,
                                    first_layer_input_ch,
                                    first_layer_pool_diam,
                                    first_layer_pool_diam,
                                    first_layer_out_cxt_rows,
                                    first_layer_out_cxt_cols,
                                    first_layer_cxt_diam,
                                    first_layer_cxt_diam,
                                    first_layer_n_neurons,
                                    first_layer_eta,
                                    first_layer_thresh_eta,
                                    first_layer_threshold_open,
                                    first_layer_tau,
                                    first_layer_trace_tau)

    second_layer_out_cxt_rows::Int32 = output_layer_cxt_rows
    second_layer_out_cxt_cols::Int32 = output_layer_cxt_cols
    second_layer_input_ch::Int32 = first_layer_n_neurons
    second_layer_n_neurons::Int32 = 180
    second_layer_eta::Float32 =  0.0005
    second_layer_threshold_open::Float32 = 0.001
    second_layer_tau::Float32 = 2*1e3 
    second_layer_trace_tau::Float32 = 3*1e3
    second_layer_thresh_eta::Float32 = second_layer_eta


    second_layer = Odesa.ConvOdesa.Conv(second_layer_input_rows,
                                    second_layer_input_cols,
                                    second_layer_input_ch,
                                    second_layer_pool_diam,
                                    second_layer_pool_diam,
                                    second_layer_out_cxt_rows,
                                    second_layer_out_cxt_cols,
                                    second_layer_cxt_diam,
                                    second_layer_cxt_diam,
                                    second_layer_n_neurons,
                                    second_layer_eta,
                                    second_layer_thresh_eta,
                                    second_layer_threshold_open,
                                    second_layer_tau,
                                    second_layer_trace_tau)

    output_layer_input_ch::Int32 = second_layer_n_neurons
    output_layer_n_classes::Int32 = nClass
    output_layer_n_neurons_per_class::Int32 = Int32(round(second_layer_n_neurons/nClass))
    output_layer_eta::Float32 = 0.001
    output_layer_threshold_open::Float32 = 0.5
    output_layer_tau::Float32 = 3*1e3
    output_layer_thresh_eta::Float32 = output_layer_eta


    output_layer = Odesa.ConvOdesa.ConvClassifier(output_layer_input_rows,
                                        output_layer_input_cols,
                                        output_layer_input_ch,
                                        output_layer_cxt_rows,
                                        output_layer_cxt_cols,
                                        output_layer_n_neurons_per_class,
                                        output_layer_n_classes,
                                        output_layer_eta,
                                        output_layer_thresh_eta,
                                        output_layer_threshold_open,
                                        output_layer_tau)


    hidden_layers = Vector{Odesa.ConvOdesa.Conv}()
    push!(hidden_layers, first_layer, second_layer)
    model = Odesa.ConvOdesa.ConvModel(hidden_layers,output_layer)
    winners::Array{Tuple{Int32,Int32, Int32}} = [(Int32(1), Int32(1), Int32(1)) for i in model.hidden_layers]::Array{Tuple{Int32,Int32, Int32}}
    return model
end

function iter_over(Odesa_ConvOdesa::Module,events::Vector{Any},correct_class::Float64,wrong_class::Float64,no_class::Float64,model::Odesa.ConvOdesa.ConvModel)
    
    @inbounds for event in events      
        (x,y,ts,p,label) = event
        @inbounds for (xx,yy,tts,pp,ll) in zip(x,y,ts,p,label)
            winners, output_winner, output_class = Odesa_ConvOdesa.forward(model,x,y,p,ts,label)
            if output_winner != -1
                if output_class == label
                    correct_class += 1.0
                else
                    wrong_class += 1.0
                end
            else
                no_class +=1.0
            end
        end
    end
    (correct_class::Float64,wrong_class::Float64,no_class::Float64)
end

function train!(dataset::PyObject,events_cache::Array{Array{Any}},model::Odesa.ConvOdesa.ConvModel)
    output_winner::Int32 = 0
    output_class::Int32 = 0
    best_accuracy::Float64 = 0
    correct_class::Float64 = 0
    wrong_class::Float64 = 0
    no_class::Float64 = 0
    correct_percent::Float64 = 0
    wrong_percent::Float64 = 0
    no_percent::Float64 = 0    
    correct_class = 0
    wrong_class = 0
    no_class = 0
   # @time @inbounds for _ in 1:20      
    #Odesa.ConvOdesa.reset(model)
    #@inbounds @showprogress for _ in 1:10  
    #println("batch time")
    @time @inbounds for events in events_cache            
            (correct_class,wrong_class,no_class) = iter_over_new(Odesa.ConvOdesa,events,correct_class,wrong_class,no_class,model)
    end
    @show(correct_class)

    # end
    correct_percent = correct_class/(correct_class+no_class+wrong_class)
    no_percent = no_class/(correct_class+no_class+wrong_class)
    wrong_percent = wrong_class/(correct_class+no_class+wrong_class)

    #end


    if correct_percent >= best_accuracy
        best_accuracy = correct_percent
    end
    @show(correct_percent, wrong_percent, no_percent)
    return correct_class,wrong_class,no_class

    
end

model = prealloc(input_shape)


function bds(dataset::PyObject,model::Odesa.ConvOdesa.ConvModel)
    output_winner::Int32 = 0
    output_class::Int32 = 0
    best_accuracy::Float64 = 0
    correct_class::Float64 = 0
    wrong_class::Float64 = 0
    no_class::Float64 = 0
    correct_percent::Float64 = 0
    wrong_percent::Float64 = 0
    no_percent::Float64 = 0    
    correct_class = 0
    wrong_class = 0
    no_class = 0
   # @time @inbounds for _ in 1:20      
    #Odesa.ConvOdesa.reset(model)
    #println("inside previoustrain")
    training_order = shuffle(0:dataset.get_count()-1)
    storage::Array{Tuple{Vector{UInt32}, Vector{UInt32}, Vector{Float32}, Vector{Int8}, Vector{UInt32}}} = []
    cnt = 1

    @time @inbounds for batch in 1:100:length(training_order)

        # println("before pycall")

        events = dataset.get_dataset_item(training_order[batch:batch+100-1])
        #@show(length(events))
        # println("after pycall")

        # @time @inbounds for ev in events
            # println("before iterover")

        #(correct_class,wrong_class,no_class) = 
        cnt = build_data_set_native(Odesa.ConvOdesa,events,storage,cnt)
            # println("after iterover")

        #end
        # @time profile_time(dataset,batch,events_cache,training_order)
    end
    @save "all_mnmist.jld" storage
    #println("gets here")
    # @inbounds @showprogress for _ in 1:10  
    #     println("batch time")
    #     @time @inbounds for events in events_cache            
    #          (correct_class,wrong_class,no_class) = iter_over(Odesa.ConvOdesa,events,correct_class,wrong_class,no_class,model)
    #     end
    #     @show(correct_class)

    # end
    #correct_percent = correct_class/(correct_class+no_class+wrong_class)
    #no_percent = no_class/(correct_class+no_class+wrong_class)
    #wrong_percent = wrong_class/(correct_class+no_class+wrong_class)

    #end


    #if correct_percent >= best_accuracy
    #    best_accuracy = correct_percent
    #end
    #@show(correct_percent, wrong_percent, no_percent)
    #return correct_class,wrong_class,no_class

    
end



# @time (correct_class,wrong_class,no_class) = train!(dataset,events_cache,model)
#@time (correct_class,wrong_class,no_class) = train!(dataset,events_cache,model)

#end
#
#for ev in events_cache @show(size(ev)) end
function iter_over_new(Odesa_ConvOdesa::Module,events,storage,cnt)
    @show(length(events))
    xx = Vector{UInt32}([])
    yy = Vector{UInt32}([])
    tts = Vector{Float32}([])
    polarity = Vector{Int8}([])
    label = Vector{UInt32}([])
    @inbounds for ev in events      
        cnt+=1
        if cnt<13977584
            if cnt!=283249 & cnt!=228845
                (x,y,ts,p,l) = ev
                push!(xx,trunc(UInt32,x))
                push!(yy,trunc(UInt32,y))
                push!(tts,Float32(ts/1000.0))
                push!(polarity,trunc(Int8,p))
                push!(label,trunc(UInt32,l))
            end
        end
    end
    #@show(typeof((xx,yy,tts,polarity,label)))
    did_it_exec::Tuple{Vector{UInt32}, Vector{UInt32}, Vector{Float32}, Vector{Int8}, Vector{UInt32}} = (xx,yy,tts,polarity,label)
    push!(storage,did_it_exec)
    #@assert length(last(storage)) == length(events)

    #@show(length(xx))
        #@show(length(x))
        #@inbounds for (ind,(xx,yy,tts,pp,ll)) in enumerate(zip(x,y,ts,p,label))
            #@show(length(ev),ind)
            #if ind==length(ev)
            #    println("hit")
            #end
            #@show(xx,yy,tts,pp,ll)
            #@show(typeof(xx),typeof(yy),typeof(tts),typeof(pp),typeof(ll))
            #@show(ind)
            #=
            winners, output_winner, output_class = Odesa_ConvOdesa.forward(model,Int32(xx),Int32(yy),Int32(tts),Int32(pp),Int32(ll))
            if output_winner != -1
                if output_class == label
                    correct_class += 1.0
                else
                    wrong_class += 1.0
                end
            else
                no_class +=1.0
            end
            =#
        #end
    #end
    #(correct_class::Float64,wrong_class::Float64,no_class::Float64)
    cnt
end
=#
