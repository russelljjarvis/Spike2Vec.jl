@load "all_mnmist_complete.jld" temp_container_store

#(x,y,times,p,labels,nodes) = load_datasets()
get_plot_uniform(temp_container_store)
