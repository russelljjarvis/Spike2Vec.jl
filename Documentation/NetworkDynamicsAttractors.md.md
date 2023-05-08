http://www.scholarpedia.org/article/Attractor_network
In general, an attractor network is a network of nodes
(i.e., neurons in a biological network), often recurrently connected,
whose time dynamics settle to a stable pattern. That pattern may be stationary, 
time-varying (e.g. cyclic), or even stochastic-looking (e.g., chaotic).
The particular pattern a network settles to is called its ‘attractor’.
In theoretical neuroscience, different kinds of attractor neural networks have been associated with different functions, 
such as memory, motor behavior, and classification. Describing networks as attractor networks allows researchers to employ 
methods of dynamical systems theory to quantitatively analyze their characteristics (e.g. stability, robustness, etc.).

Trajectories.
When simulated with two different initial conditions, 
the synaptic drive to neurons deviated strongly from each other (Figure 4a),
and the spiking activity of single neurons was uncorrelated across trials and 
the trial-averaged spiking rate had little temporal structure (Figure 4b). 
The network activity was also sensitive to small perturbation;
the microstate of two identically prepared networks diverged rapidly if one spike was deleted from one of the networks (Figure 4c). 
It has been previously questioned as to whether the chaotic nature of an excitatory-inhibitory 
network could be utilized to perform reliable computations (London et al., 2010; Monteforte and Wolf, 2012).

 is updated, the network states evolve freely with no constraints and can thus diverge from the desired trajectory. This allows the network to visit different network states in the neighborhood of the target trajectories during training, and the trained network becomes resistant to relatively small perturbations from the target trajectories. Third, the synaptic update rule is designed to reduce the error between the target and the ongoing network activity each time W
 is updated. Thus, the sequential nature of the training procedure automatically induces stable dynamics by contracting trajectories toward the target throughout the entire path. In sum, robustness to initial conditions and network states around the target trajectories, together with the contractive property of the learning scheme, allow the trained network to generate the target dynamics in a stable manner.

"Of these mechanisms of active information storage the case ofcircular causal interactions in a loop motif,
and the causal, butrepetitive inﬂuence from another part of the system may seemcounterintuitive at ﬁrst,
as we might think that in these casesthere should be information transfer rather than active informa-tion storage.
To see why these interactions serve storage ratherthan transfer, 
it may help to consider that all components of infor-mation processing, i.e., transfer, active storage and modiﬁcation,
ultimately have to rely on causal interactions in physical systems.Hence, the presence of a causal interaction cannot be linked ina one-to-one 
fashion to information transfer, as otherwise therewould be no possibility for physical causes of active 
informationstorage and of information modiﬁcation left, and no consis-tent decomposition of information processing would be possible.Therefore,
the notion of storage that is measurable in a part of thesystem but that can be related to external inﬂuences onto that 
partis to be preferred for the sake of mathematical consistency andultimately, usefulness. We acknowledge that information
transferhas often been used as a proxy for a causal inﬂuence, dating backto suggestions by Wiener (1956) and Granger (1969).
However,now that causal interventional measures and measures of infor-mation transfer can be clearly
distinguished (Ay and Polani, 2008;Lizier and Prokopenko, 2010) it seems no longer warranted to
map causal interactions to information transfer in a one-to-onemanner
"
