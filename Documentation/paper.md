-----
title: 'Spike2Vec'

tags:
  - Neuroinformatics
  - Simulation of Spiking Neural Networks.
  - Computational Neuroscience
  - Big Data

authors (in no order or undecided order):
  - name: Russell Jarvis
    affiliation: International Centre for Neuromorphic Systems, MARCS Institute, Western Sydney University
  - name: Yeshwanth Bethi
    affiliation: International Centre for Neuromorphic Systems, MARCS Institute, Western Sydney University
  - name: Pablo de Abreu Urbizagastegui
    affiliation: International Centre for Neuromorphic Systems, MARCS Institute, Western Sydney University

date: June  2023

Bibliography: paper.bib

### Summary
A scalable algorithm that can detect fine grained repetitions quickly across large spiking datasets is desirable, as it provides a means to test for the tendency of activity to revisit states. By quantifying repetitions large spiking datasets, using geometric representations of complex spike patterns, we can quantify the frequency of repitition, and achieve a better understanding of a networks ability to revisit states. To this end we represented time bound neural activity as simple geometric coordinates in a highdimensional space. Working with geometric representations of chaotic spike train recordings may enable researchers to interrogate the state-fullness of both biologically recorded spike trains and their digitally simulated counterparts. Furthermore, there is reason to believe that when mammal brains enact visual object recognition encoded memories guide cortical neurons to “replay” previously observed neural states, as replayed spiking states may cohere with the visual brains perceptual recognition of a familiar scene.  

Multivariate approaches to spike train network analysis often involves the computation of some kind of statistic between each possible pair of neurons in the network. To analyse causality in networks, spike train recordings are divided into time windows, and analysis compares previous (lagged time), with current time. Exhaustive pairwise iteration of multivariate statistics is not computationally tractible at the scale of billions of neurons, and adding time lagged analysis of network cross-correlation, or transfer entropy makes the prospoect of scaled temporal analysis even worse. Auto-covariance acts on anolog signals (dense vectors), however autocovariance analysis of continuous membrane potentials would be another way to arrive at a network state description.

Two common models of cortical spiking networks are the, Potjan's and Diesmon model and the Brunel model, both of these models are said exist within a fluctuation driven regime, when these are simulated, observed spike times are typically chaotic and random, but some fine grained recognizable repeating patterns also occur. Under the dynamic systems view of the brain neuronal memories are analogous to attractor basins [Hopfield,Lin, Hairong, et al]. If the view of memories as basins is correct then it should be possible to demonstrate synaptic learning as the mechanism that encodes memories as basins. Network attractor basins may be derived from the interleaved application of Spike Timing Dependent Plasticity (STPD) and sleep when synapses are able to change in a way that strongly biases some future spiking activities towards stereotyped patterns. 

The application of STDP learning within the fluctuation driven regime necessitates a simple method to optimise network parameters a way that maximises the networks capacity to encode and revisit attractor states. A spike2vec algorithm will enable researchers to investigate the state-fullness of spike trains, the corruption of information caused by STDP in the absence of sleep and resistance to the degradation of memories that may be concomitant with neuronal death and synaptic pruning, as many of these network level phenonemana can be re-construed as network parameters: for example neuronal death relates to synaptic count and neuron count.

### Statement of Need

Scalable methods for representing the transient behavior of large populations of neurons are needed. The spike2vec algorithm will enable researchers to track the trajectory of the network between familiar and unfamiliar states using a high-dimensional coordinate scheme. A network’s ability to revisit an encoded coordinate is testable, and so a spike2vector test of object recognition could be construed as a formal hypothesis test.

### Reproducibility
Some preliminary code that performs the Spike2Vec analysis is avaialble at the following [link](https://github.com/russelljjarvis/SpikingNeuralNetworks.jl/blob/master/src/analysis.jl#L29-L83). the code is implemented in Julia, a modern language alternative to Python that makes large-scale model visualization and analysis more computationally tractable. A docker file is included.

### References
@ARTICLE{Eliasmith:2007,
AUTHOR = {Eliasmith, C. },
TITLE   = {{A}ttractor network},
YEAR    = {2007},
JOURNAL = {Scholarpedia},
VOLUME  = {2},
NUMBER  = {10},
PAGES   = {1380},
DOI     = {10.4249/scholarpedia.1380},
NOTE    = {revision \#91016}
}

@article{illing2019biologically,
  title={Biologically plausible deep learning—but how far can we go with shallow networks?},
  author={Illing, Bernd and Gerstner, Wulfram and Brea, Johanni},
  journal={Neural Networks},
  volume={118},
  pages={90--101},
  year={2019},
  publisher={Elsevier}
}

@article{kim2020dynamics,
  title={Dynamics of multiple interacting excitatory and inhibitory populations with delays},
  author={Kim, Christopher M and Egert, Ulrich and Kumar, Arvind},
  journal={Physical Review E},
  volume={102},
  number={2},
  pages={022308},
  year={2020},
  publisher={APS}
}
