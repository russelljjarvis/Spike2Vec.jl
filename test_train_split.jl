using JLD2
@load "all_mnmist_complete.jld" storage
using ScikitLearn
using ScikitLearn.CrossValidation: cross_val_score
#@load "all_mnmist_complete.jld" temp_container_store
#(x,y,times,p,labels,nodes) = load_datasets(storage)
#@show(length(labels))
#@show(length(nodes))
(train,test)=ScikitLearn.CrossValidation.train_test_split(storage)
