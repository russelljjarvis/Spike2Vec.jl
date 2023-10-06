
#__precompile__(false) 
module SpikeTime

    #export ST
    #const ST = SpikeTime
    #using OhMyREPL
    using LinearAlgebra
    using SparseArrays
    using Requires
    using UnPack
    #using CUDA
    using ProgressMeter
    using Setfield
    using DrWatson
    #initialize_project(@__DIR__)
    quickactivate(@__DIR__)
    using StaticArrays
    #using Colors
    #using StatsPlots
    using Mmap
    #using StatsBase
    #using Plots
    #using Statistics

    #using StaticArrays
    #include("genPotjans.jl")
    include("neuron/uniform_param_lif.jl")
    export IFNF
    include("unit.jl")
    include("models/genPotjansWiring.jl")
    export potjans_layer
    export build_neurons_connections
    #include("plot.jl")
    #include("neuron/if.jl")
    #include("neuron/16bit_if.jl")
    #include("neuron/if2.jl")
    #include("neuron/noisy_if.jl")
    include("ISI.jl")
    export bag_of_isis
    include("input_readers.jl")
    export get_all_exempler_of_days
    export load_datasets_calcium_jesus
    export load_zebra_finche_nmc_dataset
    include("spike2vec.jl")
    export create_spikes_ragged
    export compute_metrics_on_matrix_divisions
    export compute_metrics_on_divisions
    export return_spike_item_from_matrix
    export horizontal_sort_into_tasks
    export divide_epoch
    export get_vector_coords
    export get_vector_coords_uniform!
    export surrogate_to_uniform
    export label_exhuastively_distmat!
    export label_exhuastively_distmat
    export label_online_distmat
    export cluster_distmat
    export cluster_distmat!
    export get_division_via_recurrence
    export spike_matrix_divided!#_no_displacement
    export get_division_scatter_identify
    export get_division_scatter_identify2
    export get_state_transitions
    export state_transition_trajectory
    export get_division_scatter_identify_via_recurrence_mat
    export get_repeated_scatter
    export spike_matrix_divided
    #export post_proc_viz
    #export final_plots
    export get_divisions
    export plot_umap_of_dist_vect
    export create_ISI_histogram
    export bag_of_isis
    include("util.jl")
    include("neuron/poisson.jl")

    include("synapse/spiking_synapse.jl")
    include("main_e.jl")
    export NMNIST_pre_process_spike_data
    export sim!
    export simx!
    export get_appropriate_cellid
    export restructure_stim

    #include("main_experiment2.jl")
    include("neuron/rate.jl")
    include("synapse/rate_synapse.jl")

    #=
    include("neuron/iz.jl")
    include("neuron/hh.jl")


    include("synapse/fl_synapse.jl")
    include("synapse/fl_sparse_synapse.jl")
    include("synapse/pinning_synapse.jl")
    include("synapse/pinning_sparse_synapse.jl")
    =#
    include("plot.jl")
    export get_mean_isis
    export plot_umap
    export get_ts!
    export get_ts
    export hist2dHeat
    export internal_validation1
    export internal_validation0
    export internal_validation_dict
    export ragged_to_uniform

end
