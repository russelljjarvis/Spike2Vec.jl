# "Spike2Vec: Converting Spike Trains to Vectors to Analyse Network States and State Transitions in very large scale data.

#  Abstract
  
  # Introduction:

  * For techical reasons in experimental neuroscience, long duration spike train data sets have not been feasible until very recently, even still these datasets are still only marginally feasible, and they depend on motion correction algorithms.
  
  * These datasets are so big, it is felt that they overwhelm existing data processing capabilities, and this in a large part be due to intrinsic limitations of existing algorithms and their language implementations.
  
  * In hippocampus recordings from CA1 of rats solving mazes, electrode recordings have been able to detect repeated Spatial Temporal Patterns RSTPs which have been dubbed "replay". 
  
  * The prevalence and frequency of Repeated Patterns are of huge significance to our understanding of how the brain works.
  
  * A real time large scale brain modelling platform is currently in development, real time large scale brain simulators will generate a tera bytes of spike train datasets, and it is important to have a way to visualize these data sets.


 # Methods:
  * Approach 
  * Algorithm steps, written
  * Algorithm presented as a flow chart.
  * Find the optimal bin width for data sets, by calculating the amount of variance (dispersion distances) in the spike vector coordinates. High clustering with maximal distances between cluster centres (calculate goodness of clustering fit).

### Dimensionality Reduction of the time evolving Vectors.
* The Number of repeating patterns should match the number of distinct clusters found in the scatter plot of the first two principle component vectors of the low dimensional embedding. This isn't expected to be an integer/ordinal number binary match, but just a propotional match (ie off by 1 or 2 in a group of 10 is explained by stochastic and noisy nature of the algorithm).

### Evaluation of Feature Descriptor Used.
 * Calculate Variance of matrices for different descriptors, this is a way of seeing how much contrast is levarged across the different approaches to representing spike trains as vectors..
 
### Evaluation of Effects of Lossiness.
* Do a sliding window approach to pattern detection, given a fixed repeat threshold, how much extra replay is detected if templates are allowed to slide across different time steps?

* Given that spike trains are asychronous events, the onset of a repeating pattern does not follow a clock but is also asychronous. For this reason state descriptions for each window of events only consider the distribution of ISIs in the window, as opposed to the exact times the spikes occured at in a given window.
  
 when considering the sum of sliding windows, versus
* Benchmark against SPADE on evaluation time, ability to not cause memory failure and accuracy.

* Very often neuronal groups are highly correlated with each other. Allowing the spatial size of replays to be smaller than the whole neuronal count, allows for different sub sections of the network to be doing different things.

# Method demonstration that the algorithm works.

Overlaying detected replays. Plotting the raster plot of two different replays identified as the same pattern ontop of each other, color coding for the different temporal occurance.


# Discussion:

* Do repeated states have a preferred or strict sequential order? Markov Decision Matrices, if they have a proven strict sequential order, what is the average pattern length in units of replay.


* Dimensionality reduction on vectors created by spike trains orginating from different organisms, if Dimensionality reduction does not recognize the spike trains originating from different organisms as seperable clusters, it might mean that there is overlap in repeating temporal spatial patterns.

* Dimensionality reduction on only the set of repeating patterns (between all organisms and datasets), might reveal other constraints.

* When comparing replays between different species, what does the within species ISI distribution look like versus the out of species/ pooled global distribution look like?

Is there a way to correct for different firing rates?

Also is there a way to measure the size of spatial patterns in terms of neurons involved? If it was found that multiple different nervous systems from different organisms preferred replays of time duration X, and spatial size in neurons Y.


Should we be applying optimization constraints that cause replay events to be of this nature (in time duration, and in pattern spatial size neurons).


For the smallest replay size, divide that by `dt`, ie get the ratio of vector distance, over temporal distance.


A network that has a lower ratio of vector distance to temporal distance, would take a long time to move between states (state inertia), and a network that has high ratio of vector distance to temporal distance might be erratic.


* In the simplest sense replayed events (repeated spike train states) cohere with the common human intuition that whatever the representational units of the brain, these representations must repeat during sensory recognition, and during the recall of memories.

 (including visual object recognition, and olfactory recognition). 
   

## Conclusion.

The Spike2Vec approach can successfuly find repeating temporal Patterns in large scale data.

## Scraps

* What else can vectors tell us? Summing all of the time evolved vectors togethor would cause a relatively short vector if each vector was going off in a different direction (meandering path). It would cause a long vector if there was high repetition. It would cause a short vector if there was high repitition of two different vector states which represented very different high dimensional coordinates.


  * https://github.com/SpikeAI/2022-11_brainhack_DetecSpikMotifs/blob/main/figures/synthetic_patterns.png

  SPADE has limitations, it is of 
  SPADE limitations. SPADE is unable to detect certain types of patterns, it is of interest if the spike

  New tools required: Finding RSTP in biological and in-silico data quickly and at scale.

  Testing on biological data, and Neuromorphic datasets.
  Rationale 

  Ability to scale.

  Testing on notebook data where spade fails.

  Ability to detect repeats from neuromorphic cameras not scale or translation invariant.

    * In the context of anomoly detection, an anomolous event is simply a brain state that is not repeated. 
  
  
  * In a more complicated sceanario anomolous events may be a state that has not only not previously been observed, but which is also statistically atypical of what has previously been observed. A matrix representation of spike train states, may allow us to characterize variance patterns in observations, in a way that allows us to make inferences to new observations about whether new observations statistically agree with what exists, or if it is in fact anolomylous.

