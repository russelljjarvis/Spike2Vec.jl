-----
title: 'Spike Time: Exploiting Modern Language Features for High Throughput Spiking Network Simulations, at a Lower Tech Debt '
tags:
  - Simulation of Spiking Neural Networks.
  - Computational Neuroscience
  - Large Scale Modelling and Simulation

authors:
  - name: Russell Jarvis
    affiliation: International Centre for Neuromorphic Systems, MARCS Institute, Western Sydney University
date: June  2023

Bibliography: paper.bib

### Summary
Some gains in biologically faithful neuronal network simulation can be achieved by applying recent computer language features. For example, the Julia language supports [Sparse Compressed Arrays](https://docs.julialang.org/en/v1/stdlib/SparseArrays/), [Static Arrays](https://juliaarrays.github.io/StaticArrays.jl/stable/), furthermore Julia provides very extensive support for CUDA GPU, as well as a plethora of reduced precision types. Julia also provides a high-level syntax that facilitates high code reuse while simplifying plotting and data analysis. These features lend themselves towards high-performance large-scale Spiking Neural Network simulation. Therefore, we are using Julia to develop an open-source software package that enables the simulation of networks with millions to billions of synapses on a computer with a minimum of 64GB of memory and an NVIDIA GPU.  

Some other important advantages of choosing to implement SNN simulations in the Julia language are: technical debt, and the ability to minimise total energy consumption of simulations. The simulation code we are developing at ICNS is both faster and less complicated to read compared with some other simulation frameworks. The simplicity of the code base encompasses a simple installation process. Ease of installation is an important part of neuronal simulators that is often overlooked when evaluating merit, GPU simulation environments are notoriously difficult to install. The Julia language facilitates the ease of installation to solve the “two language problem” of scientific computing. The simulator encompasses a singular language environment, which includes a reliable, versatile, and monolithic package manager. The simulator installation includes no external compilation tools or steps. 

To demonstrate the veracity and performance of this new simulation approach, I compare the Brunel model and the Potjans and Diesmann model as implemented in the NEST and GENN simulators. In a pending analysis, we compare simulation execution speeds and spike train raster plots to NEST and GENN using the discussed models as benchmarks. 

### References
B. Illing, W. Gerstner & J. Brea, Biologically plausible deep learning - but how far can we go with shallow networks?, Neural Networks 118 (2019) 90-101
