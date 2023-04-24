# First add depencies for the example
using Pkg; Pkg.add.(["Plots", "DSP"])
using Weave
weave("spike2vectest.jmd"; out_path=:pwd)