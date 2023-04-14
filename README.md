
[![CI](https://github.com/russelljjarvis/SpikingNeuralNetworks.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/russelljjarvis/SpikingNeuralNetworks.jl/actions/workflows/ci.yml)

## Spurious Spikes.
This project started life as a rogue project, or a shadow and has since become legitimate. The project consists of three parts:

* a simulation part, which was largely inherited and has since been improved for speed and efficiency.
* a structural analysis part, which also helps for things like load balancing the network.
* a temporal analysis part.

### Temporal Analysis
Neural activity in a window can be represented as a coordinate in high-dimensional space. When mammal brains enact object recognition, encoded memories guide cortical neurons to “replay” previously seen neural states, which happen to be states that correspond to familiar scenes. 

The spike2vec algorithm will enable reseachers to track the trajectory of the network between familiar and unfamiliar states using a high-dimensional coordinate scheme. A network’s ability to revisit an encoded coordinate is testable, and so a spike2vector test of object recognition could be construed as a formal hypothesis test. 


## Installation

```julia
using Pkg
pkg"add https://github.com/russelljjarvis/SpikingNeuralNetworks.jl"
```
This work is different from other simulators in the sense that it mainly intends to scale well and read well at the same time. This work is less concerned with code generation and garunteed robust solutions to DEs (the forward Euler method is used for synapse and Vm updates because it is fast). Furthermore another goal of this work is neural analysis and spike data visualizations that scale to billions of neurons. This work is directly derived from work by @AStupidBear (https://github.com/AStupidBear/SpikingNeuralNetworks.jl), and new contributions and differences in this work are explained below.

This simulation framework only takes `30` seconds after being pre-compiled (in the background) to simulate `20,000` LIF neurons with millions of synapses.
The framework supports scalling to millions of cells and billions of synapses given sufficient computer memory is available. In my experience 64GB can support very large cell count simulations.

![image](https://user-images.githubusercontent.com/7786645/228764232-b6818524-ea31-461f-913d-5e50196a2a6f.png)

Time surfaces of the neural activity (similar to a membrane potential heatmap across the network). Note that time surface heatmaps are often used in relation to the FEAST algorithm:  https://www.mdpi.com/1424-8220/20/6/1600

![image](https://user-images.githubusercontent.com/7786645/228764258-4da67dfe-1e8b-4a30-97eb-724a9e7dd683.png)

![image](https://user-images.githubusercontent.com/7786645/228764191-10262134-8602-4c7c-81ae-57e0c7ca871c.png)


## Older Plots


The simulated networks are capable of producing rich and unusual dynamics, furthermore dynamics are expected to become more sophisticated as STDP learning rules are re-introduced (in a CUDA compatible way), and also as more elaborate models will be included, such as the ![Potjan's and Diesmann model](https://github.com/russelljjarvis/SpikingNeuralNetworks.jl/blob/master/examples/genPotjansWiring.jl).

![image](https://user-images.githubusercontent.com/7786645/227809116-d7180fbd-e937-4bdb-bb0d-77645c1eb284.png)

Caption: Top half of plot excitatory population activity, bottom half inhibitory population activity. The plot is interesting because 


![image](https://user-images.githubusercontent.com/7786645/228695786-d496ce45-8df2-401f-a72c-ec48b8281d83.png)


file:///home/rjjarvis/Desktop/Screenshot%20from%202023-04-12%2015-02-38.png
Caption another UMAP of spike trains
file:///home/rjjarvis/Desktop/Screenshot%20from%202023-04-12%2015-02-24.png![image](https://user-images.githubusercontent.com/7786645/231696958-d51206c1-825b-45f0-8b5e-e17af6a66c22.png)


file:///home/rjjarvis/Desktop/Screenshot%20from%202023-04-12%2015-02-24.png![Uploading image.png…]()

[image](https://user-images.githubusercontent.com/7786645/231696944-f13de2dd-5bf6-4ee6-aaf7-718840626215.png)
Community structure of the structural wiring corresponding to the Potjan's network.


# Completed Upgrades:

* Models can be simulated as 16Bit Float (to simulate the effect of reduced precision). Note 16Bit precision on CPU is only "emulated" as it is not supported by real CPU hardware, 16bit precision can feasibly speedup GPU simulation, because NVIDIA GPUs do provide Float16 and Int8 support in hardware.
* Low precision models have a tendency to cause Inf, and NaN values, so these values are now cleaned during simulation.
* Adaptive Exponential Integrate and Fire Neuron.
* More unit tests.
* Time Surface Plots
* UMAP of spike trains
* Constructors with parametric types.
* Progress Bars, because people like to know how long they should expect to wait.
![image](https://user-images.githubusercontent.com/7786645/227809077-b7b19bf0-cffc-493f-9d28-2034d1bdf038.png)

# Pending Upgrades
* GLIF neurons.
* Robust CUDA compilitability including STDP and 16bit float.
* Bio Network Consumption of the ```NMNIST.NmnistMotion("./nmnist_motions.hdf5")``` dataset
* Use Block Array with display to visualize the connection topology of the network.
* Make axonal delays possible using something similar to the refractory period counter inside neural model population structs.
* Demonstration of Potjans network, I have written the majority of the code here, some minor bugs need addressing: https://github.com/russelljjarvis/SpikingNeuralNetworks.jl/blob/master/examples/genPotjansWiring.jl
The Potjan's Diesman model is an important network model that researchers re-implement again and again (implementations: Nest, Brian2, PyNN etc)
https://www.biorxiv.org/content/10.1101/248401v1

## Similar Simulators With Different Goals

* https://github.com/FabulousFabs/Spike.jl (looking very inspired by Brian2 except in Julia, nice use of code generation)
* https://github.com/SpikingNetwork/TrainSpikingNet.jl (nice use of CUDA and reduced precision types)
* https://github.com/leaflabs/WaspNet.jl (nice use of block arrays)
* https://github.com/darsnack/SpikingNN.jl (nice use of multidispatch abstract types and type restriction)
* https://github.com/wsphillips/Conductor.jl (nice use of DiffEq.jl and code generation)
* https://github.com/FabulousFabs/AdEx (interesting and different)
* https://github.com/ominux/pub-illing2019-nnetworks (research orientated code)


## Documentation
### FEAST analysis support example:
 Since Odesa is not an officially Julia registered package, running this example would require:
```julia 
 using Pkg
 Pkg.add(url=https://github.com/russelljjarvis/Odesa.jl-1")
 ] resolve
 ] activate
 ] instantiate
```
* https://github.com/russelljjarvis/SpikingNeuralNetworks.jl/blob/master/test/if_net_odessa.jl
The documentation of this package is not finished yet. For now, just check `examples` and `tests` folder.
