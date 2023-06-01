#import "template.typ": *
#show: ieee.with(
  title: "Spike2Vec: Converting Spike Trains to Vectors to Analyse Network States and State Transitions: ",
  abstract:[
   #include "Abstract.typ"
  ] ,

  authors: (
    (name: "Dr Russell Jarvis", affiliation: "International Centre for Neuromorphic Systems, MARCS Institute, Western Sydney University"),
    (name: "Mr Pablo de Abreu Urbizagastegui", affiliation: "International Centre for Neuromorphic Systems, MARCS Institute, Western Sydney University"),
    (name: "Mr Yeshwanth Bethi", affiliation: "International Centre for Neuromorphic Systems, MARCS Institute, Western Sydney University"),
  ),
  // Insert your abstract after the colon, wrapped in brackets.
  // Example: `abstract: [This is my abstract...]`
 //index-terms: ("A", "B", "C", "D"),
  bibliography-file: "refs.bib",
)


// We generated the example code below so you can see how
// your document will look. Go ahead and replace it with
// your own content!



//= Abstract
//A scalable algorithm that can detect fine grained repetitions quickly across large spiking datasets is desirable, as it provides a means to test for the tendency of activity to revisit states. By quantifying repetitions large spiking datasets, using geometric representations of complex spike patterns, we can quantify the frequency of repitition, and achieve a better understanding of a networks ability to revisit states. To this end we represented time bound neural activity as simple geometric coordinates in a highdimensional space. Working with geometric representations of chaotic spike train recordings may enable researchers to interrogate the state-fullness of both biologically recorded spike trains and their digitally simulated counterparts. Furthermore, there is reason to believe that when mammal brains enact visual object recognition encoded memories guide cortical neurons to “replay” previously observed neural states, as replayed spiking states may cohere with the visual brains perceptual recognition of a familiar scene.

= Introduction

Both temporal and rate codes fail to fully explain how the cortical neurons of mammal brains are able to use synapses to learn about the world. The attractor network approach to understanding the brain is code agnostic. By using chaos attractor trajectories as an explanation the dynamic systems view of the brain can at least explain why some of the brains activity patterns are able to repeat, and why the not all brain activity is high entropy, asychronous irregular activity. The network attractor theory is compatible with recent spike train datasets that demonstrate "replay" in the hippocampus and prefrontal cortex of rats.

//"replay as network attractor" theory of memory encoding and memory recall.// explain how neural patterns 


There is demand for a scalable algorithm that can detect repeating temporal spatial features in biological and synthetic data sets of cortical neuronal networks. An existing tool SPADE #cite("stella20193d", "quaglio2017detection", "shinomoto2005measure"), can detect repeating spatial temporal sequences, however it is only available in the Python language, and it application of SPADE so far does not fully reconcile the attractor network theory of neuronal learning with spike train analysis. Reconciling the attractor network theory with in-silico and in-vivo spike trains is a prominant feature of this work.

//In order to garner evidence for the "replay as network attractor" theory of memory encoding and memory recall faster scalable 

There may be latent evidence for the Dynamic Sytems/Network Attractor theory of neuronal computation, and it is imperative to uncover any evidence for the attractor network theory in the vast amount of public spiking data on the internet. Methods are needed to transform spike raster plots into attractor trajectories directly into state transition networks, and it is unclear if the existing implementations of SPADE can achieve this in a performant and large scale manner. //and energy landscapes. 

Furthermore there is a desire to perform neuronal data analysis in a manner that lends itself to extensibility with Julia, a modern language with high performance features, when the Julia ecosystem has a new framework, the combinations of performance orientated tools becomes possible. 

/*and there is a general demand for neuronal analysis tools in the Julia ecosystem.*/
//@stella20193d SPADE
//@quaglio2017detection SPADE
//@shinomoto2005measure SPADE

Multivariate approaches to spike train network analysis often involve inferrring a causal network diagram from spiking acitivity. Where the idea is that the spiking activity contains many hidden network states, that emerge in large scale simulations, and that are not visible in the networks static connectome. The computation of some kind of statistic between each possible pair of neurons in the network. To analyse effective connectivity in in silico and in vitro networks, spike train recordings are divided into time windows, and analysis compares previous (lagged time), with current time. Exhaustive pairwise iteration of multivariate statistics is not computationally tractible at the scale of billions of neurons, and adding time lagged analysis of network cross-correlation, or transfer entropy makes the prospoect of scaled temporal analysis even worse. Auto-covariance acts on anolog signals (dense vectors), however autocovariance analysis of continuous membrane potentials would be another way to arrive at a network state description.

Two common models of cortical spiking networks are the, Potjan's and Diesmon model and the Brunel model, both of these models are said exist within a fluctuation driven regime, when these are simulated, observed spike times are typically chaotic and random, but some fine grained recognizable repeating patterns also occur. Under the dynamic systems view of the brain neuronal memories are analogous to attractor basins [Hopfield,Lin, Hairong, et al] The authors consider replay as a mechanism that is related both memory encoding and memory recall. If the view of memories as basins is correct then it should be possible to demonstrate synaptic learning as the mechanism that encodes memories as basins. Network attractor basins may be derived from the interleaved application of Spike Timing Dependent Plasticity (STPD) and sleep when synapses are able to change in a way that strongly biases some future spiking activities towards repeating observed patterns.

The application of STDP learning within the fluctuation driven regime necessitates a simple method to optimise network parameters a way that maximises the networks capacity to encode and revisit attractor states. Although spike trains are often analysed for information entropy at the spike to spike level. A spike2vec algorithm will enable researchers to investigate the spike trains states for redundancy at the sequential pattern level, in a Julia compliant way, and this will parallel efforts to develop Julia simulators.

// the corruption of information caused by STDP in the absence of sleep and resistance to the degradation of memories that may be concomitant with neuronal death and synaptic pruning, as many of these network level phenonemana can be re-construed as network parameters: for example neuronal death relates to synaptic count and neuron count.

It is the authors view, that the fast algorithm described above is functionally similar to the RecurrenceAnalysis approach, and that it leads to a faster and more 
interprebable network transition matrices.


//=== Intended caption for spike2vec document.

=== Intended Discusssion

The attractor network view of the mammal cortex is consistant with phenomological observations about the mind, such that people commonly refer to "circular thinking", in obsessive compulsive disorder.
Furthermore action and perception are theorized to occur in alternating cycles, during "action-perception" loops. 

Neuronal synaptic weight changes, that happen as a result of STDP, simply bias the brain in a manner which will make salient brain states more likely to occur.

It is possible that the windows which were disregarded because they didn't repeat, may well repeat given long enough neuronal recordings. This is an unavoidable problem, caused by fact that only a limited duration of recording data is: a. available and b, computationally tractable to analyse.

In the absence of infinite recording data, we can draw no conclusions about wether one of observations would have re-occured given more time. Another caveat is that inside this frame work it is impossible to distinguish between brain states that are relevant to the organism, and state transitions, which may also look similar from trial to trial.

//or finite observations are novel network states, or repeating states.

//Whatever the case, state, or state transition, detecting periods of repeating patterns in a fast and scalable way, still bolsters the attractor network view of the brain.
//The algorithm may also be helpful for feature reconstruction in Neuromorphic Data sets from event based cameras.

== Evidence for Network Attractor Theory

Reoccurance analysis was used to characterize vector encoded spike train matrices for repeating patterns. Re-currence analysis was able to give us numbers quantify the degree of repitition of states, and the entropy of state transition matrices. Steps convert the spike encoded vector matrices to "state space sets" as defined in the Julia Package Dynamical Systems.jl

/*
Re-occurance analysis did 

#TODO quantify the complexity of state transition matrices with the non reoccuring states included, as this may give us insight about, information in the brain at a different time scale.

 state transition networks

Delay Embeddings can 

//dynamic systems view of the brain @scholarpedia attractor_network.
caused by the network transitioning to familiar states, 
*/


== Discussion

Each conscious recall of a memory may involve the brain approximately repeating a pattern; ie recall may mean re-visiting a previous pattern of neuronal activity. Each recalled memory may retain significant traces of activity that is well correlated with the brain experience that caused the memory to be encoded. Such "replay" is has been observed in the hippocampus and prefrontal cortex in rats during sleep. 

When replayed events are detected a sequential map of states can be derived from a spike train, and unrecognized state transitions and anomolies can also be sorted and labelled in discrete chunks.

Under spike2vec framework, spike patterns which are approximately the same as patterns in previous windows are detected because, in the geometric coordinate representation of vectors, spike trains which are close together will have be seperated by a small Euclidean Distance in the vector space.

The spike2vec frame work can be used to convert spike trains to markov transistion matrices, or simply state transition network maps. In such a state network, we can see that not all spike trains are equally stateful, some emperical recordings may have few replay events. When a spike recording is particularly stateful, there may be certain states which are only entered into given the appropriate sequence of prior states. 

// each of the vector space are labelled as reoccuring.

= Theoretical Framework

=== Algorithm Details
The spike2vec frame work exists at the meta level. It is a novel mashup of pre-existing algorithms, its steps are as follows:

1. Spike trains are divided into N equally sized windows time windows.

2. In each window spike times are converted by subtracting the window start time, such that spike time variability is now mapped onto the local time frame in the smaller scope of each window (ie the time each window occured is subtracted from each window, making any variation of spike times inside the window relevant to the windows scale). Each of the converted time, time windows is stored in an array.

3. The maximum firing rate of all the windows is found.

4. A single artificial spike train in a window is constructed by taking the maxmimum firing rate from step 3. And constructing a spike train that has regular Inter Spike Intervals (ISIs) occuring at the maximum firing rate. We call this the reference window, or surrogate.

5. For every N windows sampled in 1, the observed spike times is compared to the uniform reference window using the Thomas Kreuz spike Distance algorithm implemented in Julia by George Datseris. https://github.com/JuliaNeuroscience/SpikeSynchrony.jl/commits?author=Datseris


6. The Kreuz spike distance is a way of measuring the cost of converting observed spike train A, to a different spike train B. By measuring the Kreuz spike distance between a variation free regular spiking window, and a window with observed spike time variability, we get a picture of each neurons current unique local variability at each window (note that the method for comparing reference to observed doesn't have to uniquely encode unique spike sequences, it just has to be unique enough). There is M number of neurons we can build a vector of coordinate of *M* dimensions, at each of N time windows. An M by *N* matrix consists of M neurons and N time windows.

7. Since each column vector encodes a time window, we get the euclidian distance between each column vector and every other column vector, across the columns of the whole matrix. 

8. We take these new distance values we fill a new matrix, between every window, and every other window at row and column location of the matrix. It's important to recognize that here we are not comparing spike distances between neurons (as has occured in established work, we are commparing spike train distance vectors within the same neurons along time). 

9. We perform unsupervised clustering on this temporaly encoded dissimalirity matrix.

10. We discard all cluster labels that correspond to just a single time window, and retain the set of cluster labels, that have at least one repeating label. We regard these duplicated cluster labels as repeated temporal spatial patterns. 

//7. Unsupervised clustering is applied to the matrix across columns to find .
= Methodological Framework

A problem with converting spike train raster plots to attractor trajectories, is the that the most established method  deriving attractor trajectories (and energy landscapes) requires the system under investigation to be encoded as a continuous differentiable function. A dominant approach which satisfys the continuous function requirement is to fit a differential equation that models a networks firing rate(s) in response to current injection the assumption underlying this approach, is that the rate coded information and network states are more important than or even exclude temporal codes.    

Another approach to estimating attractor trajectories involves applying Delay Coordinate Embeddings framework. The advantage of this approach is that
a model equation is not required, and a timeseries of system observations satisfies the algorithms requirements. Spikes time raster plots are sparsely encoded collections of events that are naturally encoded by ragged arrays, and delay coordinate embeddings requires a state space map. Vector matrices that are output from spike2vec are sufficient to satisfy Delay Coordinate Embeddings, however, the frame work is slow to evaluate, and the quality of the output of the algorithm dependent on many parameters (both in parameters of spike2vec and DCE).

//julia recurrence analysis     N. Marwan et al., "Recurrence plots for the analysis of complex systems", Phys. Reports 438(5-6), 237-329 (2007).
/*
    N. Marwan & C.L. Webber, "Mathematical and computational foundations of recurrence quantifications", in: Webber, C.L. & N. Marwan (eds.), Recurrence Quantification Analysis. Theory and Best Practices, Sprin */

Yet another approach is to use Recurrence Analysis. Recurrence Analysis is orders of magnitude faster than DCE, and the results of DCE
usefully describe the network properties of state transition matrices. In order to find an optimal time window we could use consistently between data sets, we swept through a range of window lengths (ms), and found the window length which would maximise the correlation between peristimulus time histograms on exemplar spike raster patterns.

In order to test that the "auto spike train distance", metric lead to more well defined network descriptors than other similar but more common metrics, We compared state vectors that were constructed by applying auto-covariance and local variation to the same spike windows, and we compared the spike2vec algorithms performance across all three metrics.  @illing2019biologically Julia simulation of learning. We simulated NMNIST learning. @kim2020dynamics Dynamics systems view of the brain.

Finally we used a Julia package Emeddings.jl to convert spike train sequences to English words, by iterating over word embedding vectors in large word2vec models, and finding closely matching vectors, we did this in order to make patterns in spike train vectors more intuitive and accessible, by giving repeated spike sequence elements familiar word labels. By converting spike trains to English word sequences we can compare the statistics of written language to spike train statistics. The word2vec approach was found to generalize to other realms such as product recommendations using meta data @vasile2016meta, and human EEG recordings.


//https://github.com/JuliaText/Embeddings.jl


#table(
  columns: (auto, auto, auto),
  inset: 10pt,
  align: horizon,
  [*Kreuz Distance from Uniform ISI*], [*Auto Covariance*], [*Local Variation*],
    [Kreuz spike distance number $0$],
  [
    Autocovariance number $1$
  ],
  [local variation number $0$],
)

//@Eliasmith Dynamic systems view of the brain from scholar pedia @mackevicius2019unsupervised Julia labelling horinzontal




/*
= Statement of Need

Scalable methods for representing the transient behavior of large populations of neurons are needed. The spike2vec algorithm will enable researchers to track the trajectory of the network between familiar and unfamiliar states using a high-dimensional coordinate scheme. A network’s ability to revisit an encoded coordinate is testable, and so a spike2vector test of object recognition could be construed as a formal hypothesis test.


= Reproducibility

Some preliminary code that performs the Spike2Vec analysis is avaialble at the following link. the code is implemented in Julia, a modern language alternative to Python that makes large-scale model visualization and analysis more computationally tractable. A docker file is included. 
*/
= Result Analysis
#align(center + bottom)[
  #image("figures/UniformSpikes.png", width: 70%)
  *A plot of the regular periodic spike reference window. A unvarying uniform surrogate spike train is used as a comparison inorder to compute the transformation cost of transforming spike train uniform to spike train varying.*]

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
  *We sampled the NMNIST data set more broadly using $6,000$ samples to create vectors. The entire data set consists of $60,000$ samples. *
]


//#align(center + bottom)[
//  #image("figures/clustering_NMNIST.png", width: 70%)
//]/
//align(center + bottom)[
 // #image("figures/clustering_NMNIST.png", width: 70%)
 //   *Unormalize NMNINST vectors for 1000 neurons over 10 channels*
//]

align(center + bottom)[
  #image("figures/vector_differences_another_NMNIST.png", width: 70%)
  *Two spike time encoded numerals, where read in to Julia, then the spiking neuromorphic data were converted to vectors over $1200$ channels. Orange and Blue plots are vectors corresponding to two distinct NMNIST data labels. Rapid positive alternating deflections are visible in both vectors, because the NMNIST data is caused by pixel activations, when $2D$ pixel derived data sources are converted into a $1D$ vector, sparse clusters of activated pixels, have regular gaps between them. Herein lies a heatmap of dis-simarity matrices constructed using the NMNIST dataset, ie the heatmap above, comes from analysing spike train distance across the NMNIST data set numbers: 0-9 represented as spiking events. There are 300 total presentation number presentations. All nine numbers are incrementally cycled through. Number presentations within the one number are contiguous, (the data set isn't shuffled), and this contiguity is reflected in the heatmap too.*
]

//align(center + bottom)[
//  #image("figures/UMAP_song_bird.png", width: 70%)
//  *Unormalize NMNINST vectors for 1000 neurons over 10 channels*
//]


/*cluster_sort_MNMIST.png
cluster_sort_song_birds.png
didit_work_NM.png
didit_work.png
everything_includinding_repeated_pattern_pablo.png
everythin_includinding_repeated_pattern_pablo.png
heatmap_after.png
heatmap_before.png
heatmap.png
just_two_pablo_raw_vectors.png
just_two_song_bird_raw_vectors.png
labelled_mat_of_distancesNMINST.png
labelled_mat_of_distancesNMINST_test_train.png
labelled_mat_of_distances_pablo.png
labelled_mat_of_distances.png
LabelledSpikes18.png
LabelledSpikesPartition18.png
Levine13-CD4.png
Normalised_heatmap_pablo.png
Normalised_heatmap_song_bird.png
normal.png
not_cluster_sort_MNMIST.png
pablo_umap.png
reference_labelled_mat_of_distances_pablo.png
relative_to_uniform_referenceNMMIST.png
repeated_pattern_pablo.png
repeated_pattern.png
repeated_pattern_song_bird.png
scatternmnist_angles.png
scatternmnist_distances.png
scatternmnist.png
slice_one_window.png
slice_three_window.png
slice_two_window.png
sorted_train_map.png
test_map.png
train_map.png
umap_of_NMNIST_Data.png
UMAP_song_bird.png
UniformSpikes.png
Unormalised_heatmap_pablo.png
Unormalised_heatmap.png
Unormalised_heatmap_song_bird.png
vector_differences_another.png
*/


== References

//#bibliography("refs.bib")

//https://elifesciences.org/articles/38471

//Test reference @illing2019biologically
