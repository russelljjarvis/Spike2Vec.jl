#using Pkg
#ENV["python"]="/usr/bin/"
#Pkg.build("PyCall")
using PyCall
py"""
import numpy as np
import quantities as pq
import neo
import elephant
#import viziphant
def get_trains():
    np.random.seed(4542)
    # https://github.com/SpikeAI/2022-11_brainhack_DetecSpikMotifs/blob/main/2022-11-28_SPADE_tutorial.ipynb

    spiketrains = elephant.spike_train_generation.compound_poisson_process(
    rate=5*pq.Hz, A=[0]+[0.98]+[0]*8+[0.02], t_stop=10*pq.s)
    len(spiketrains)

    for i in range(90):
        spiketrains.append(elephant.spike_train_generation.homogeneous_poisson_process(
            rate=5*pq.Hz, t_stop=10*pq.s))

    return spiketrains
"""
trains = py"get_trains"()
#py"exec('SPADE.py')"
