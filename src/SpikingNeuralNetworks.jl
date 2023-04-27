module SpikingNeuralNetworks

export SNN
const SNN = SpikingNeuralNetworks

using LinearAlgebra
using SparseArrays
using Requires
using UnPack
using CUDA
using ProgressMeter
using Setfield

using StatsPlots
#using StatsBase
#using Plots

#using StaticArrays
#include("genPotjans.jl")
include("neuron/noise_free_lif.jl")
export IFNF
include("unit.jl")
include("models/genPotjansWiring.jl")
export potjans_layer
#include("plot.jl")
#include("neuron/if.jl")
#include("neuron/16bit_if.jl")
#include("neuron/if2.jl")
#include("neuron/noisy_if.jl")
include("spike2vec.jl")
export divide_epoch
export get_vector_coords
export surrogate_to_uniform
export post_proc_viz
export final_plots
include("util.jl")
include("neuron/poisson.jl")

include("synapse/spiking_synapse.jl")
include("main_e.jl")

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
#function __init__()
#    @require Plots="91a5bcdd-55d7-5caf-9e0b-520d859cae80" include("plot.jl")
#end

end
