A scalable algorithm that can detect fine grained repetitions quickly across large spiking datasets is desirable, as such a frame work would provide a means to test for the tendency of neuronal activity to revisit states. 

Quickly identifying repeated states in large scale neuronal data and simulation is important, as the degree of repitition should influence the mindset of the scientists undertaking an analysis of spike trains. Forinstance several established cortical network models have assumed that realistic cortical neuronal activity should be Asynchronous and Irregular Activity (AI) in character such as Brunel's balanced model of cortex @brunel1996hebbian. 

By ascertaining repetitions in large spiking datasets we can quantify the frequency of repitition, and achieve a better understanding of a networks ability to revisit states. To this end we represented time bound neural activity as simple geometric coordinates in a highdimensional space. Working with geometric representations of chaotic spike train recordings may enable researchers to find a common set of RTSPs. In this work we show the beggining contributions of compiled Vectorized library of neuronal spike train recordings, that contains recordings from different individuals and also individuals belonging to different mammal species. By compiling a Vectorized library of neuronal spike train recordings, we will be able to transform vectors by swapping axis order of the constituent vectors to find the vector axis order that maximises overlap between replayed events from different individuals.  






/*, using geometric representations of complex spike patterns,

However, new data sets prominently capture replayed states, and previously collected of spike trains may too have hidden states of replay. The limited recordings from limited species may have biased previous recordings in a way that under represented the prevalence of replay. recordings or otherwise  have been misleading, as too their limited ability to capture replay, and detect it in analysis.



Although the dynamic systems view of the brain is old, a survey of spiking datasets which can detect and labels network attractor states in large spike count data is merited, as this would bolster the dynamic systems view of the neuronal learning. 


 Furthermore, there is reason to believe that when mammal brains enact visual object recognition encoded memories guide cortical neurons to “replay” previously observed neural states, as replayed spiking states may cohere with the visual brains perceptual recognition of a familiar scene.

*/

/*Introduction

This code is associated with the paper from Dhawale et al., "Automated long-term recording and analysis of neural activity in behaving animals". eLife, 2017. http://dx.doi.org/10.7554/eLife.27702

https://github.com/SpikeAI/2022-11_brainhack_DetecSpikMotifs/blob/main/2022-09-13_Bernstein-Copy1.ipynb
https://github.com/SpikeAI/2022-11_brainhack_DetecSpikMotifs/blob/main/2022-09-13_Bernstein-Copy1.ipynb

https://github.com/SpikeAI/2022-11_brainhack_DetecSpikMotifs/tree/main

Data sets. SPADE.


This synthetic ground-truth dataset accurately models long-term, continuous extracellular tetrode recordings from the rodent brain over a time-period of 256 hours. Each "recording" comprises spiking of 8 distinct single-units with firing rates ranging from 0.1 - 6 Hz, superimposed on background multi-unit spiking activity at 20 Hz. The recording sampling rate is 30 kHz. Single-unit spike amplitudes drift over a range of 100 to 400 μV
 based on the drift we observe in our own long-term recordings from the rodent motor cortex and striatum. For more details, please see our paper "Automated long-term recording and analysis of neural activity in behaving animals" ( https://doi.org/10.1101/033266).
These recordings can be used to test the accuracy of spike-sorting algorithms when clustering non-stationary spike waveform data, such as our own Fast Automated Spike Tracker (FAST) outlined in our paper and available at https://github.com/Olveczky-Lab/FAST.
*/