#import "template.typ": *
#show: ieee.with(
  title: "Spike2Vec: Converting Spike Trains to Vectors to Analyse Network States and State Transitions",
  abstract:[
   #include "Abstract.typ"
  ] ,

  authors: (
    (name: "Russell Jarvis", affiliation: "Postdoctoral Research International Centre for Neuromorphic Systems, MARCS Institute, Western Sydney University"),
    (name: "Pablo de Abreu Urbizagastegui", affiliation: "International Centre for Neuromorphic Systems, MARCS Institute, Western Sydney University"),
    (name: "Yeshwanth Bethi", affiliation: "International Centre for Neuromorphic Systems, MARCS Institute, Western Sydney University"),
    (name: "Alexandre Marcireau", affiliation: "Postdoctoral Research Fellow International Centre for Neuromorphic Systems, MARCS Institute, Western Sydney University"),
  ),
  bibliography-file: "refs.bib",
)



//= Abstract
//A scalable algorithm that can detect fine grained repetitions quickly across large spiking datasets is desirable, as it provides a means to test for the tendency of activity to revisit states. By quantifying repetitions large spiking datasets, using geometric representations of complex spike patterns, we can quantify the frequency of repitition, and achieve a better understanding of a networks ability to revisit states. To this end we represented time bound neural activity as simple geometric coordinates in a highdimensional space. Working with geometric representations of chaotic spike train recordings may enable researchers to interrogate the state-fullness of both biologically recorded spike trains and their digitally simulated counterparts. Furthermore, there is reason to believe that when mammal brains enact visual object recognition encoded memories guide cortical neurons to “replay” previously observed neural states, as replayed spiking states may cohere with the visual brains perceptual recognition of a familiar scene.

/*
Elife approach.
== abstract
Please provide an abstract of no more than 150 words. Your abstract should explain the main contributions of your article, and should not contain any material that is not included in the main text.
abstract

*/

/*
= Introduction (Level 1 heading)

= Results (Level 1 heading)

== Level 2 Heading

=== Level 3 Heading

=== Level 4 Heading

= Discussion  (Level 1 heading)

= Methods and Materials  (Level 1 heading)

= Introduction (Level 1 heading)
*/

The most basic aspect of approach of the spike2vec framework is very old, under the established and conventional technique a population of spike trains is converted to a series of spike firing rate encoded vectors. This idea has been around since the late 80s @georgopoulos1988primate. However this work departs from the conventional approach as we decided to construct vectors from the neurons instantaneous pattern of spiking variability in contrast to instananeuous firing rate, somewhat analogously to sampling each neurons local variation as it evolves through time. When the neurons instantanous spike time variability is measured it is entered into a population vector. As network behavior evolves through time, we are able to re-examine collections of these vectors to find approximately repeating temporal spatial patterns. When we have located a repeating temporal spatial patterns we are then able to consider each RTSP as a bag of Inter Spike Intervals, where the spatial structure of spike sources is discarded, and RTSPs overall distributions of RTSPs are analysed between different individuals of the same species to find out if there is universal temporal profiles of rodent RTSPs, in approach somewhat inspired by the work @perez2021parallel



//In order to garner evidence for the "replay as network attractor" theory of memory encoding and memory recall faster scalable 

There may be latent evidence for the Network Attractor theory of neuronal learning in older spiking data sets, and it is imperative to uncover any evidence for the attractor network theory in the vast amount of public spiking data on the internet. Methods are needed to transform spike raster plots into attractor trajectories directly into state transition networks, and it is unclear if the existing implementations of SPADE can achieve this in a performant and large scale manner. //and energy landscapes. 

Furthermore there is a desire to perform neuronal data analysis in a manner that lends itself to extensibility with Julia, a modern language with high performance features, when the Julia ecosystem has a new framework, the combinations of performance orientated tools becomes possible. 
//Single spikes drive sequential propagation and routing of activity in a cortical network
//https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9925052/
//"generating a highly combinatorial space of sequence activations.""


An pre-exisiting theory @Eliasmith_2007, the attractor network theory of the mammal cortical dynamics is consistant with phenomological observations about the mind, such that people commonly refer to "circular thinking", in obsessive compulsive disorder. Furthermore action and perception are theorized to occur in alternating cycles, during "action-perception" loops.  Neuronal synaptic weight changes, that happen as a result of STDP, simply bias the brain in a manner which will make salient brain states more likely to occur.

Both temporal and rate codes fail to fully explain how the cortical neurons of mammal brains are able to use synapses to learn about the world. The attractor network approach to understanding the brain is code agnostic. Deterministic chaotic principles are used to explain spike time irregularity. Spike time network attractors can explain why some of the brains activity patterns are able to repeat, and why the not all cortical activity is highly Asychronous Irregular Activity (AI). The network attractor theory is compatible with recent spike train datasets that demonstrate "replay" in the hippocampus and prefrontal cortex of rats. Replay refers to a phenomena were in two time points of a spike train, there is macroscale similarity between spike patterns, and each pattern is  generally reconizable, and regarded as approximately the same by human observers. 
// trajectories as an explanation the dynamic systems view of the brai
//"replay as network attractor" theory of memory encoding and memory recall.// explain how neural patterns 


There is demand for a scalable algorithm that can detect repeating temporal spatial features in biological and synthetic data sets of cortical neuronal networks. An existing tool Spike Pattern Detection and Evaluation (SPADE) #cite("stella20193d", "quaglio2017detection", "shinomoto2005measure"), can detect repeating spatial temporal sequences, however it is only available in the Python language. Application of SPADE so far does not fully reconcile the attractor network theory of neuronal learning with spike train analysis. Reconciling the attractor network theory with in-silico and in-vivo spike trains is a prominant feature of this work.

An un-answered question is their commanality in the size and shape of RPS between similar but different mammal brains. When electrodes are inserted into different brains, the spatial location of recording electrodes, never reaches the same target twice. Variability in the structure of brains is a rule, and when comparing two neurons that occupy roughly the same position, but in different individual brains.  It is rarely justified to consider the identity of cortical neurons would not be considered to be the same entity. Nontheless it remaians viable and desirable to transform vectors derived from spike trains into a common set of coordinates, by finding the exact sequence of vector axis that maximises overlap between repeated patterns. Analogous to word2vec word embedding models that were famously used in recommendation systems. RTSPs can be compared against spike2vec embedding models, however, before this could happen, spike2vec vectors would need to be transformed to a common coordinate system, where spike train vectors from different individuals suddenly become reconcilable with each other. The word2vec approach was found to generalize to other realms such as product recommendations using meta data @vasile2016meta, and human EEG recordings. In this work we show the beggining contributions of compiled Vectorized library of neuronal spike train recordings, that contains recordings from different individuals and also individuals belonging to different mammal species. By compiling a Vectorized library of neuronal spike train recordings, we will be able to transform vectors by swapping axis order of the constituent vectors to find the vector axis order that maximises overlap between replayed events from different organisms.  


//A scalable algorithm that can detect fine-grained repetitions quickly across large spiking datasets is desirable, as such a framework would provide a means to test for the tendency of neuronal activity to revisit states. 

Quickly identifying repeated states in large-scale neuronal data and simulation is essential, as the degree of repetition should influence the mindset of scientists analyzing spike trains. For instance, several established cortical network models have assumed that realistic cortical neuronal activity should be Asynchronous and Irregular (AI) in character. Two common models of cortical spiking networks are the, Potjan's and Diesmon @potjans2014cell model and the Brunel model @brunel1996hebbian, both of these models exist within a fluctuation driven regime. When each of these respective network models are simulated, observed spike times are typically appear to be poisson distrubited psuedo random spike times. By design these models make it unlikely that fine grained recognizable repeating patterns also occur. The Potjan's model can be used to make data points seperable. However, new data sets prominently capture replayed states, and previously collected spike trains may, too, have latent and unpublished states of replay. The limited recordings from limited species may have biased previous recordings in a way that underrepresented the prevalence of replay.


 /*Often simulations of such activity use Brunel's balanced model of the cortex  @brunel1996hebbian.
 recordings or otherwise  have been misleading as to their limited ability to capture replay and detect it in analysis. */

//Although the dynamic systems view of the brain is old, a survey of spiking datasets which can detect and labels network attractor states in large spike count data is merited, as this would bolster the dynamic systems view of the neuronal learning. 

/*Direct quote: "Neuronal ensembles, coactive groups of neurons found in spontaneous and evoked cortical activity, are causally related to memories and perception, but it is still unknown how stable or flexible they are over time. We used two-photon multiplane calcium imaging to track over weeks the activity of the same pyramidal neurons in layer 2/3 of the visual cortex from awake mice and recorded their spontaneous and visually evoked responses. Less than half of the neurons were commonly active across any two imaging sessions. These "common neurons" formed stable ensembles lasting weeks, but some ensembles were also transient and appeared only in one single session. Stable ensembles preserved ~68 % of their neurons up to 46 days, our longest imaged period, and these "core" cells had stronger functional connectivity. Our results demonstrate that neuronal ensembles can last for weeks and could, in principle, serve as a substrate for long-lasting representation of perceptual states or memories.*/


/*Direct quote: 

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
 is updated. Thus, the sequential nature of the training procedure automatically induces stable dynamics by contracting trajectories toward the target throughout the entire path. In sum, robustness to initial conditions and network states around the target trajectories, together with the contractive property of the learning scheme, allow the trained network to generate the target dynamics in a stable manner.*/ 


After surveying the literature, we found evidence of two-three major replay detection algorithms, SPADE is written in Python, and FAST, which is advertised as "a spike sorting algorithm" is written in MATLAB @dhawale2017automated.  We also found neuronal spike train recordings which are either long term, or bordering on chronic @perez2021long, @riquelme2023single

/*and there is a general demand for neuronal analysis tools in the Julia ecosystem.*/
Multivariate approaches to spike train network analysis often involve inferrring a causal network diagram from spiking acitivity. Where the idea is that the spiking activity contains many hidden network states that emerge in large scale simulations, and that are not visible in the networks static connectome. To analyse effective connectivity in in silico and in vitro networks, spike train recordings are divided into time windows, and analysis compares previous (lagged time), with current time. Exhaustive pairwise iteration of pair-wise and or multivariate statistics is not computationally tractible at the scale of billions of neurons, and adding time lagged analysis of network cross-correlation, or transfer entropy makes the prospoect of scaled temporal analysis even worse. /* Auto-covariance acts on anolog signals (dense vectors), however autocovariance analysis of continuous membrane potentials would be another way to arrive at a network state description. The computation of a bivariate statistic between each possible pair of neurons in the network. */


Under the dynamic systems view of the brain neuronal memories are analogous to attractor basins @lin2023review, @Eliasmith_2007. Consider replay as a mechanism that is related both memory encoding and memory recall. If the view of memories as basins is correct then it should be possible to demonstrate synaptic learning as the mechanism that encodes memories as basins. Network attractor basins may be derived from the interleaved application of Spike Timing Dependent Plasticity (STPD) and sleep when synapses are able to change in a way that strongly biases some future spiking activities towards repeating observed patterns.

Application of STDP learning within to cortical models of Leaky Integrate and Fire (LIF) neurons a simple method to optimise network parameters a way that maximises the networks capacity to encode and revisit attractor states. Although spike trains are often analysed for information entropy at the spike to spike level. A spike2vec algorithm will enable researchers to investigate the spike trains states for redundancy at the sequential pattern level, in a Julia compliant way, and this will parallel efforts to develop Julia cortical simulators which intend to model prefrontal cortex in a data driven way.



= Theoretical Framework


Representational simalarity @grootswagers2019representational has been applied to decoding visual representations from ECG channels, by analysing the differences between channels, and how these differences evolve over time. Rather than applying representational similarity between pairs of neurons in the network, we instead compare spike train distance across one neurons evolving spiking behavior, by assessing how much the neuron changes its representation.

To compute vectors from spike trains we apply representational disimilarity between different windows of the neurons history, ie we evaluate spike train distance across one neurons evolving spiking behavior, by progressively evaluating how much the neuron  deviates from uniform spiking in incremental windows.

A problem with converting spike train raster plots to attractor trajectories, is the that the most established method  deriving attractor trajectories (and energy landscapes) requires the system under investigation to be encoded as a continuous differentiable function. A dominant approach which satisfys the continuous function requirement is to fit a differential equation that models a networks firing rate(s) in response to current injection the assumption underlying this approach, is that the rate coded information and network states are more important than or even exclude temporal codes.    

Another approach to estimating attractor trajectories involves applying Delay Coordinate Embeddings framework. The advantage of this approach is that
a model equation is not required, and a timeseries of system observations satisfies the algorithms requirements. Spikes time raster plots are sparsely encoded collections of events that are naturally encoded by ragged arrays, and delay coordinate embeddings requires a state space map. Vector matrices that are output from spike2vec are sufficient to satisfy Delay Coordinate Embeddings, however, the frame work is slow to evaluate, and the quality of the output of the algorithm dependent on many parameters (both in parameters of spike2vec and DCE).

//julia recurrence analysis     N. Marwan et al., "Recurrence plots for the analysis of complex systems", Phys. Reports 438(5-6), 237-329 (2007).
/*
    N. Marwan & C.L. Webber, "Mathematical and computational foundations of recurrence quantifications", in: Webber, C.L. & N. Marwan (eds.), Recurrence Quantification Analysis. Theory and Best Practices, Sprin */

Yet another approach is to use Recurrence Analysis. Recurrence Analysis is orders of magnitude faster than DCE, and the results of DCE
usefully describe the network properties of state transition matrices. In order to find an optimal time window we could use consistently between data sets, we swept through a range of window lengths (ms), and found the window length which would maximise the correlation between peristimulus time histograms on exemplar spike raster patterns.

In order to test that the "auto spike train distance", metric lead to more well defined network descriptors than other similar but more common metrics, We compared state vectors that were constructed by applying auto-covariance and local variation to the same spike windows, and we compared the spike2vec algorithms performance across all three metrics.  @illing2019biologically Julia simulation of learning. We simulated NMNIST learning. @kim2020dynamics Dynamics systems view of the brain.

As an experiment we used a Julia package Emeddings.jl to convert spike train sequences to English words, by iterating over word embedding vectors in large word2vec models, and finding closely matching vectors, such that we could apply a statistical analysis of written english to spike train recordings.



// the corruption of information caused by STDP in the absence of sleep and resistance to the degradation of memories that may be concomitant with neuronal death and synaptic pruning, as many of these network level phenonemana can be re-construed as network parameters: for example neuronal death relates to synaptic count and neuron count.

/*It is the authors view, that the fast algorithm described above is functionally similar to the RecurrenceAnalysis approach, and that it leads to a faster and more 
interprebable network transition matrices.*/


//=== Intended caption for spike2vec document.
//or finite observations are novel network states, or repeating states.

//Whatever the case, state, or state transition, detecting periods of repeating patterns in a fast and scalable way, still bolsters the attractor network view of the brain.
//The algorithm may also be helpful for feature reconstruction in Neuromorphic Data sets from event based cameras.


/*
Re-occurance analysis did 

#TODO quantify the complexity of state transition matrices with the non reoccuring states included, as this may give us insight about, information in the brain at a different time scale.

 state transition networks

Delay Embeddings can 

//dynamic systems view of the brain @scholarpedia attractor_network.
caused by the network transitioning to familiar states, 
*/





// each of the vector space are labelled as reoccuring.


//7. Unsupervised clustering is applied to the matrix across columns to find .
= Methodological Framework

Data sources analysed.

#table(
  columns: (auto, auto, auto,auto),
  inset: 10pt,
  align: horizon,
  [*Zebra Finch, song bird*],[*chronic multiday recording*],[*NMNIST*], [*Prefrontal Cortex Replay*],
    [ @mackevicius2019unsupervised],[@perez2021long],[@cohen2016skimming], [@peyrache2009replay],
)

/*
spatio-temporal patterns (STPs).

#table(
  columns: (auto, auto, auto),
  inset: 10pt,
  align: horizon,
  [*Kreuz Distance against uniform ISI reference*],[*Kreuz Distance against noise reference*],[*Auto Covariance*], [*Local Variation*],
    [Kreuz spike distance uniform number $0$], [Kreuz spike distance versus noise number $0$],
  [
    Autocovariance number $1$
  ],
  [local variation number $0$],
)
*/
Spike Train state descriptors tried:
/*
metric = "CV"
complexity_ = 271.10365577203237
metric = "autocov"
complexity_ = 263.8397370724894
metric = "LV"
complexity_ = 459.6779683462953
metric = "kreuz"
complexity_ = 385.0864565742967
*/
#table(
  columns: (auto, auto, auto,auto),
  inset: 10pt,
  align: horizon,
  [*Kreuz Distance against uniform ISI reference*],[*Autocovariance*],[*Coefficient of Variation*], [*Local Variation*],
    [$385.1$], 
    [$263.8$],
    [$271.1$],
    [$459.7$]
)

In order to verify that the our particular application of the Kreuz spike distance metric lead to the most unique network state descriptions, we also constructed population vectors by applying other measurements of spike train variability. Matrices were populated by time varying vectors derived by applying the above metrics to some of the data sources. Complexity of applied spike2vec matrices, was computed by first applying columnwise normalization to the matrices, and then summing the total covariance contribution to get a scalar metric representing the complexity of the heatmap surface. Although the Local Variation metric had the highest metric for the sum of covariances, the LV matrix generally lacked repeating temporal structure, so it was not useful (see relevant figures in results).


=== Spike2Vec Algorithm Details
The spike2vec frame work exists at the meta level. It is a novel mashup of pre-existing algorithms, its steps are as follows:

1. Spike trains are divided into N equally sized time windows.

2. In each window spike times are converted by subtracting the window start time, such that spike time variability is now mapped onto the local time frame in the smaller scope of each window (ie the time each window occured is subtracted from each window, making any variation of spike times inside the window relevant to the windows scale). Each of these windows is then stored in an array.

3. The maximum firing rate of all the windows is found.

4. A single artificial spike train in a window is constructed by taking the maxmimum firing rate from step 3. And constructing a spike train that has regular Inter Spike Intervals (ISIs) occuring at the maximum firing rate. We call this the reference window, or surrogate.

5. For every N windows sampled in 1, the observed spike times is compared to the uniform reference window using the Thomas Kreuz spike Distance algorithm implemented in Julia by George Datseris. https://github.com/JuliaNeuroscience/SpikeSynchrony.jl/commits?author=Datseris


6. The Kreuz spike distance is a way of measuring the cost of converting observed spike train * A * , to a different spike train * B *. By measuring the Kreuz spike distance between a variation free regular spiking window, and a window with observed spike time variability, we get a picture of each neurons current unique local variability at each window (note that the method for comparing reference to observed doesn't have to uniquely encode unique spike sequences, it just has to be sufficiently unique to make states appropatriately distinguishable but also recognizable across a population of multiple cells). As there are  * M * number of neurons we then build a vector of coordinate of * M * dimensions, at each of N time windows. *  Xm *, is an M by * N * tensor consists of M neurons and N time windows.


7. Since each column vector of * Xm * encodes a time window, we get the euclidian distance between each column vector and every other column vector, across the columns of the whole matrix. 

8. We take these new distance values we fill a new matrix, between every window, and every other window at row and column location of the matrix. It's important to recognize that here we are not comparing spike distances between neurons (as has occured in established work, we are commparing spike train distance vectors within the same neurons along time). 

9. We perform unsupervised clustering on this temporaly encoded dissimalirity matrix.

10. We discard all cluster labels that correspond to just a single time window, and retain the set of cluster labels, that have at least one repeating label. We regard these duplicated cluster labels as repeated temporal spatial patterns. 

In the figure below we show the a reference window of uniform spikes described above.

#align(center + bottom)[
  #image("figures/UniformSpikes.png", width: 70%)
  *A plot of the regular periodic spike reference window. A unvarying uniform surrogate spike train is used as a comparison inorder to compute the transformation cost of transforming spike train uniform to spike train varying.*]


==== Common Coordinates of Spike2Vec Algorithm Details

The output of Spike2vec, is a reduced set of vectors. Only replayed events are encoded as vectors, and everything else that can be regarded as a non repeating or a state transition is disregarded.

1. When each individuals recorded spike session is encoded as a matrix of column spike2vec vectors, between pairs of matrices, iterate over pairs of full recording matrices where each matrix is derived from one of two different individuals.

2. Consider pairs of matrices between each of all possible individuals.

3. At each step find the a reorganization of the sequence of vector axis, which maximises the commanility between each pair.

/*
=== Reoccurance Analysis

Reoccurance analysis was used to characterize vector encoded spike train matrices for repeating patterns. Re-currence analysis was able to give us numbers quantify the degree of repitition of states, and the entropy of state transition matrices. Steps convert the spike encoded vector matrices to "state space sets" as defined in the Julia Package Dynamical Systems.jl
*/


= DISCUSSION


When it comes to electrical neuronal recordings of the mammal cortex, there is a risk of underestimating the frequency of neuronal replay events and overestimating the frequency of unrepeating random looking events. Repeating Temporal Spatial Patterns (RTSPs) aka sequences, and motifs are rarely observed in important electrical neuronal cortical recordings of sensory neurons @billeh2020systematic, and older recordings of neocortex in rodents and primates. If the prevalence of repitition and replay was thoroughly characterized, cortical models could then appropriately increase the amount of structured coherent repeating patterns they generate and decrease the amount of Poisson process random noise activity represented in simulation outputs. 

The reasons why there is a risk that RTSP is going unrecognized in classic neuronal recordings of spike trains will be discussed below, as most of these risks are avoidable. The first reason is that many electrical neuronal data recordings are only 3 seconds or less. Three seconds is a tiny window of opportunity, and it is not a reasonable duration for spike patterns to be repeated in a recording.

/*; and yet most of the neuronal recordings the authors have dealt with are 3 seconds in duration.*/

Reason two: Data analysis which was applied to the data, didn't necessitate replay detection, as the replay wasn't relevant to the analysis framework; also, if no significant pattern is perceptible to humans, the lack of curiosity seems justified. Reason three, the final reason: Replay detection might fail because a dedicated detector fails to detect the RTSP. Some types of temporal and spatial patterns can defeat replay detectors . Also, some neuronal recordings may be so big that they render applications with results inaccessible because replay detection could not happen in a human-relevant time scale. There is a caveat that 3 seconds might be plenty of time to see a repeat of the experimental paradigm is construed to elicit replay such as in @peyrache2009replay. If replay corresponds to memory recall, as some authors have suggested, it seems reasonable that RTSPs might take days to years to reoccur.

It is possible that the windows which were disregarded because they didn't repeat, may well repeat given long enough neuronal recordings. This is an unavoidable problem, caused by fact that only a limited duration of recording data is: a. available and b, computationally tractable to analyse. States and state transitions cannot be distinguished from each other.

In the absence of infinite recording data, we can draw no conclusions about wether one of observations would have re-occured given more time. Another caveat is that inside this frame work it is impossible to distinguish between brain states that are relevant to the organism, and state transitions, which may also look similar from trial to trial.

Each conscious recall of a memory may involve the brain approximately repeating a pattern; ie recall may mean re-visiting a previous pattern of neuronal activity. Each recalled memory may retain significant traces of activity that is well correlated with the brain experience that caused the memory to be encoded. Such "replay" is has been observed in the hippocampus and prefrontal cortex in rats during sleep. 

When replayed events are detected a sequential map of states can be derived from a spike train, and unrecognized state transitions and anomolies can also be sorted and labelled in discrete chunks.

Under spike2vec framework, spike patterns which are approximately the same as patterns in previous windows are detected because, in the geometric coordinate representation of vectors, spike trains which are close together will have be seperated by a small Euclidean Distance in the vector space.

The spike2vec frame work can be used to convert spike trains to markov transistion matrices, or simply state transition network maps. In such a state network, we can see that not all spike trains are equally stateful, some emperical recordings may have few replay events. When a spike recording is particularly stateful, there may be certain states which are only entered into given the appropriate sequence of prior states. 




//https://github.com/JuliaText/Embeddings.jl




//Dynamic systems view of the brain from scholar pedia @mackevicius2019unsupervised Julia labelling horinzontal




/*
= Statement of Need

Scalable methods for representing the transient behavior of large populations of neurons are needed. The spike2vec algorithm will enable researchers to track the trajectory of the network between familiar and unfamiliar states using a high-dimensional coordinate scheme. A network’s ability to revisit an encoded coordinate is testable, and so a spike2vector test of object recognition could be construed as a formal hypothesis test.


= Reproducibility

Some preliminary code that performs the Spike2Vec analysis is avaialble at the following link. the code is implemented in Julia, a modern language alternative to Python that makes large-scale model visualization and analysis more computationally tractable. A docker file is included. 
*/
=== Result Analysis

==== Evidence for Network Attractor Theory


#align(center + bottom)[
  #image("figures/final_output_graphs_state_transition_matrix_graph.png", width: 70%)
*The output of the framework is a sequential state transition network of the spike train. Spontaneous network activity which didn't get repeated was simply not included in the state transition diagram. Two state transition diagrams are output, one with non repeating states, and one with repeating states. *]


//patterns of activity. Projection neurons exhibit highly phasic stereotyped firing patterns. X-projecting (HVC((X))) neurons burst zero to four times per motif, whereas RA-projecting neurons burst extremely sparsely--at most once per motif. The bursts of HVC projection neurons are tightly locked to the song and typically have a jitter of <1 ms. Population activity of interneurons, but not projection neurons, was significantly correlated with syllable patterns. Consistent with the idea that HVC codes for the temporal order in the song rather than for sound, the vocal dynamics and neural dynamics in HVC occur on different and uncorrelated time scales. We test whether HVC((X)) neurons are auditory sensitive during singing. We recorded the activity of these neurons in juvenile birds during singing and found that firing patterns of these neurons are not altered by distorted auditory feedback, which is known to disrupt learning or to cause degradation of song already learned. https://pubmed.ncbi.nlm.nih.gov/17182906/

//#align(center + bottom)[
//  #image("figures/final_output_graphs_state_transition_matrix_graph.png", width: 70%)
//*cluster_horizontal_vectors_sort_song_birds.png*]



#align(center + bottom)[
  #image("figures/state_transition_matrixPFC.png.png", width: 70%)
*cluster_horizontal_vectors_sort_song_birds.png*]

#align(center + bottom)[
  #image("figures/state_transition_trajectoryPFC.png.png", width: 70%)
*cluster_horizontal_vectors_sort_song_birds.png*]


#align(center + bottom)[
  #image("figures/sensitiv_to_parameters2_genuinely_repeated_patternPFC.png.png", width: 70%)
*A scatter plot of state transition trajectories in the prefrontal cortex spike train.*]

#align(center + bottom)[
  #image("figures/most_convining_plot_genuinely_repeated_patternsongbird.png.png", width: 70%)
*cluster_horizontal_vectors_sort_song_birds.png*]

#align(center + bottom)[
  #image("figures/convincing_UMAP.png", width: 70%)
*cluster_horizontal_vectors_sort_song_birds.png*]


align(center + bottom)[
  #image("figures/umap_of_NMNIST_Data.png", width: 70%)
  *As a matter of routine UMAP dimensional embedding of spike distances was applied to all spike difference vectors in the matrix, since population level spike train local variation evolves over time. UMAP of the spike distance matrix, UMAP of the spike2vec matrix should allows data points to cluster in a time dependent manner.*
]


#align(center + bottom)[
  #image("figures/Normalised_heatmap_kreuz.385.0864565742967.stateTransMat.png.png", width: 70%)
*Kreuz spike distance variation vectors across time. The matrices from assembled vectors, have high variability, and they also do well to represent repeatability of the multivariate signal.*]


#align(center + bottom)[
  #image("figures/Normalised_heatmap_LV.459.6779683462953.stateTransMat.png.png", width: 70%)
*Local variation vectors across time. The matrices from assembled vectors, have highest variability of all the used metrics, but generally they lack structure, especially repeating structure. *]



#align(center + bottom)[
  #image("figures/state_transition_trajectorypfcpfc.png.png", width: 70%)
*cluster_horizontal_vectors_sort_song_birds.png*]

#align(center + bottom)[
  #image("figures/sensitivity_to_parameters.png", width: 70%)
*cluster_horizontal_vectors_sort_song_birds.png*]




#align(center + bottom)[
  #image("figures/MicrosoftTeams-image1.png", width: 70%)
  *In order to test if the spike2vec framework worked as expected, we downloaded a alcium imaging recording from Zebra finch (a song bird's) High Vocal Centre (brain region) source @mackevicius2019unsupervised. Although the actual data source was from (https://github.com/lindermanlab/PPSeq.jl/blob/master/demo/data/songbird_spikes.txt) The downloaded data set was then simply augmented, by duplicating the spike time raster plot in a manner that appended the full repeated recording to the end of the first recording, the process was iterated 3 times yielding a highly repititive data set $4$ times the length of the original. The intention of this exercise was simply to show that spike2vec could identify and label such obvious repeating patterns.
  *
]


#align(center + bottom)[
  #image("figures/MicrosoftTeams-image2.png", width: 70%)
  *We take the single un augmented Zebra finche data and label repeating patterns using the discussed frame work, since there are no obvious explicit repitions caused by duplicating spike patterns, the algorithm is forced to consider the similarity between more disparate population level patterns.*
]


#align(center + bottom)[
  #image("figures/genuinely_repeated_patternpfcpfc.png.png", width: 70%)
  *We took a data set from rat prefrontal cortex. The data set was captured, under experimental conditions explicitly designed to maximise the probability of recording replay ie dreaming and slow wave sleep.*
]

//#align(center + bottom)[
//  #image("both_labelled_mat_of_distances_pablo.png", width: 70%)

 // *Glaciers form an important
//  part of the earth's climate
//  system.*
//]
#align(center + bottom)[
  #image("figures/both_labelled_mat_of_distances_song_bird.png", width: 70%) 
  *Figures from left top to bottom right:A: Top left: 75 NMNIST channels were recorded and time binned in a manner which yielded 85 vectorized time bins. Bottom right: Once the vectorized time bins had been vectorized, a clustering algorithm was applied to the entire matrix of vector coordinates. Cluster centres could then be used as reference points, such that it was possible to compare all*
]


//#align(center + bottom)[
//  #image("figures/cluster_centres_map.png", width: 70%)
//]

#align(center + bottom)[
  #image("figures/clustered_big_model.png", width: 70%)
  *Normalized prefrontal cortex vectors for over 1000 time steps 15 neurons*
]


#align(center + bottom)[
  #image("figures/clustered_train_model.png", width: 70%)
  *We sampled the NMNIST data set more broadly using $6,000$ samples to create vectors. The entire data set consists of $60,000$ samples. 
  Rapid positive alternating deflections are visible in both vectors, because the NMNIST data is caused by pixel activations, when $2D$ pixel derived data sources are converted into a $1D$ vector, sparse clusters of activated pixels, have regular gaps between them. Herein lies a heatmap of dis-simarity matrices constructed using the NMNIST dataset, ie the heatmap above, comes from analysing spike train distance across the NMNIST data set numbers: 0-9 represented as spiking events. There are 300 total presentation number presentations. All nine numbers are incrementally cycled through. Number presentations within the one number are contiguous, (the dataset is not shuffled), and this contiguity is reflected in the heatmap too.*]


//#align(center + bottom)[
//  #image("figures/clustering_NMNIST.png", width: 70%)
//]/
//align(center + bottom)[
 // #image("figures/clustering_NMNIST.png", width: 70%)
 //   *Unormalize NMNINST vectors for 1000 neurons over 10 channels*
//]

align(center + bottom)[
  #image("figures/vector_differences_another_NMNIST.png", width: 70%)
  *Two spike time encoded numerals, were read in to the Julia-lang namespace, then the spiking neuromorphic data were converted to vectors over $1200$ channels. Orange and Blue plots are vectors corresponding to two distinct NMNIST data labels.* ]
