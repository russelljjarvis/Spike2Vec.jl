# SpikingNeuralNetworks

[![Build Status](https://travis-ci.com/AStupidBear/SpikingNeuralNetworks.jl.svg?branch=master)](https://travis-ci.com/AStupidBear/SpikingNeuralNetworks.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/AStupidBear/SpikingNeuralNetworks.jl?svg=true)](https://ci.appveyor.com/project/AStupidBear/SpikingNeuralNetworks-jl)
[![Coverage](https://codecov.io/gh/AStupidBear/SpikingNeuralNetworks.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/AStupidBear/SpikingNeuralNetworks.jl)

## Installation

```julia
using Pkg
pkg"add SpikingNeuralNetworks"
```

## Documentation

The documentation of this package is not finished yet. For now, just check `examples` folder.

Takes 30 seconds after forced compilation to simulate `20,000` LIF neurons with millions of synapses.
julia> include("if_net_original_tests.jl")
Progress: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| Time: 0:00:31
simulation done !

Supports millions of cells and billions of synapses.

![image](https://user-images.githubusercontent.com/7786645/227808860-97da2f13-d22a-47f9-8d09-85950b2952de.png)

