# SpikingNeuralNetworks

[![CI](https://github.com/russelljjarvis/SpikingNeuralNetworks.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/russelljjarvis/SpikingNeuralNetworks.jl/actions/workflows/ci.yml)


## Installation

```julia
using Pkg
pkg"add https://github.com/russelljjarvis/SpikingNeuralNetworks.jl"
```

Heavily based and derived from work by @AStupidBear, the main contributions and differences are explained below.

Takes 30 seconds after forced compilation to simulate `20,000` LIF neurons with millions of synapses.

Supports millions of cells and billions of synapses.

![image](https://user-images.githubusercontent.com/7786645/228764232-b6818524-ea31-461f-913d-5e50196a2a6f.png)

![image](https://user-images.githubusercontent.com/7786645/228764191-10262134-8602-4c7c-81ae-57e0c7ca871c.png)

Time surfaces of the neural activity (similar to a membrane potential heatmap across the network)

![image](https://user-images.githubusercontent.com/7786645/228764258-4da67dfe-1e8b-4a30-97eb-724a9e7dd683.png)

## Older Plots


The simulated networks are capable of producing rich and unusual dynamics, furthermore dynamics are expected to become more sophisticated as STDP learning rules are re-introduced (in a CUDA compatible way), and also as more elaborate models will be included, such as the [Potjan's model]([https://github.com/social-hacks-for-mental-health/SpikingNeuralNetworks.jl/tree/potjans](https://github.com/RJsWorkatWSU/SpikingNeuralNetworks.jl/blob/potjans_model/src/genPotjansInPlace.jl)).

![image](https://user-images.githubusercontent.com/7786645/227809116-d7180fbd-e937-4bdb-bb0d-77645c1eb284.png)

Caption: Top half of plot excitatory population activity, bottom half inhibitory population activity. The plot is interesting because 


![image](https://user-images.githubusercontent.com/7786645/228695786-d496ce45-8df2-401f-a72c-ec48b8281d83.png)

Caption another UMAP of spike trains

# Completed Upgrades:

* Models can be simulated as 16Bit Float (to simulate the effect of reduced precision) 
* Low precision models have a tendency to cause Inf, and NaN values, so these are cleaned during simulation.
* Adaptive Exponential Integrate and Fire Neuron.
* More unit tests.
* Time Surface Plots
* UMAP of spike trains
* Constructors with parametric types.
* Progress Bars, because people like to know how long they should expect to wait.
![image](https://user-images.githubusercontent.com/7786645/227809077-b7b19bf0-cffc-493f-9d28-2034d1bdf038.png)

# Pending Upgrades
* Robust CUDA compilitability including STDP and 16bit float.
* Bio Network Consumption of the ```NMNIST.NmnistMotion("./nmnist_motions.hdf5")``` dataset
* Use Block Array with display to visualize the connection topology of the network.
* Make axonal delays possible using something similar to the refractory period counter inside neural model population structs.
* Demonstration of Potjans network https://github.com/RJsWorkatWSU/SpikingNeuralNetworks.jl/blob/potjans_model/src/genPotjansInPlace.jl
*
## Similar Simulators
This work is different from other simulators in the sense that it only intends to scale well (and read well at the same time), This work is less concerned with code generation and robust solutions to DEs. Furthermore another goal of this work is neural analysis and visualization that scales well.

* https://github.com/FabulousFabs/Spike.jl (looking very inspired by Brian2 except in Julia)
* https://github.com/SpikingNetwork/TrainSpikingNet.jl (nice use of CUDA and reduced precision types)
* https://github.com/leaflabs/WaspNet.jl (nice use of block arrays)
* https://github.com/darsnack/SpikingNN.jl (nice use of multidispatch)
* https://github.com/wsphillips/Conductor.jl (nice use of DiffEq.jl)
* https://github.com/FabulousFabs/AdEx (interesting and different)


## Documentation

The documentation of this package is not finished yet. For now, just check `examples` folder.
