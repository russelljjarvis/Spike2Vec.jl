using CSV
using DataFrames
import SpikeTime.spike2vec
df=CSV.read("spikes_for_julia.csv", DataFrame)

nodes = df.i
times = df.t



