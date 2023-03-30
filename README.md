# SpikingNeuralNetworks

[![Build Status](https://ci.appveyor.com/api/projects/status/github/AStupidBear/SpikingNeuralNetworks.jl?svg=true)](https://ci.appveyor.com/project/AStupidBear/SpikingNeuralNetworks-jl)
[![Coverage](https://codecov.io/gh/AStupidBear/SpikingNeuralNetworks.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/AStupidBear/SpikingNeuralNetworks.jl)

## Installation

```julia
using Pkg
pkg"add https://github.com/russelljjarvis/SpikingNeuralNetworks.jl"
```

Takes 30 seconds after forced compilation to simulate `20,000` LIF neurons with millions of synapses.

Supports millions of cells and billions of synapses.

![image](https://user-images.githubusercontent.com/7786645/227809077-b7b19bf0-cffc-493f-9d28-2034d1bdf038.png)

file:///home/rjjarvis/git/SpikingNeuralNetworks.jl/test/heatmap_normalized.png![image](https://user-images.githubusercontent.com/7786645/228708652-74b1a5d3-811d-418e-870a-c27b7315e815.png)


![image](https://user-images.githubusercontent.com/7786645/227809116-d7180fbd-e937-4bdb-bb0d-77645c1eb284.png)

[image](https://user-images.githubusercontent.com/7786645/228695786-d496ce45-8df2-401f-a72c-ec48b8281d83.png)

Caption: Top half of plot excitatory population activity, bottom half inhibitory population activity. The plot is interesting because 

Completed Upgrades:

* Models can be simulated as 16Bit Float (to run bigger networks with more speed or the same sized network with less memory allocation).
*   Low precision models have a tendency to cause Inf, and NaN values, so these are cleaned during simulation.
* Adaptive Exponential Integrate and Fire Neuron.
* More unit tests.
* UMAP of spike trains

TODO.
* CUDA compilitability
* Use Block Array with display to visualize the connection topology of the network.
* Make axonal delays possible using something similar to the refractory period counter inside neural model population structs.

## Documentation

The documentation of this package is not finished yet. For now, just check `examples` folder.
