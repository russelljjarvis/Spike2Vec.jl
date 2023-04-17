-----
title: 'an Abundance of Spikeliness: Exploiting Modern Language Features for Larger and Easier Simulations of Spiking Neural Networks.'
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
Some gains in biologically faithful neuronal network simulation can be achieved merely by selecting and recombining recently presented computer language features. For example the Julia language supports [Sparse Compressed Arrays](https://docs.julialang.org/en/v1/stdlib/SparseArrays/), [Static Arrays](https://juliaarrays.github.io/StaticArrays.jl/stable/), furthermore Julia provides very extensive support for CUDA GPU a plethora of reduced precision types, and high level syntax which facilitates high code re-use all the while making plotting and data analysis easier. These particular features lend themselves towards large scale Spiking Neural Network simulation, therefore I am using Julia to develop an open source software package that enables the simulation of networks with millions to billions of synapses on a computer with a minimum of 64GB of memory an NVIDIA GPU. The simulation code I am developing is both faster and easier and less complicated to read compared with some other simulation frameworks, and it requires just one language, and no external compilation tools or steps. In pending graphs I will compare simulation execution speeds to other simulator speed benchmarks. 

### References
B. Illing, W. Gerstner & J. Brea, Biologically plausible deep learning - but how far can we go with shallow networks?, Neural Networks 118 (2019) 90-101
