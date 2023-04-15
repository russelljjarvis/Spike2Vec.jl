ARG BASE=11.5.0-devel-ubuntu20.04
FROM nvidia/cuda:${BASE}
RUN apt-get update && \
    apt-get upgrade -y
RUN apt-get install --yes git curl wget
RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.8/julia-1.8.3-linux-x86_64.tar.gz
RUN tar zxvf julia-1.8.3-linux-x86_64.tar.gz
ENV PATH="$PATH:julia-1.8.3/bin"
RUN julia -e 'import Pkg;Pkg.add("UpdateJulia")'
RUN julia -e 'using UpdateJulia;update_julia()'
RUN git clone https://github.com/SpikingNetwork/TrainSpikingNet.jl

WORKDIR TrainSpikingNet.jl



##

# Undocummented step.

##

RUN julia -e 'import Pkg;Pkg.instantiate()'

RUN bash tsn.sh install



##

# Undocummented steps.

##

RUN julia -e 'import Pkg; Pkg.add("NNlibCUDA")'

RUN julia -e 'import Pkg; Pkg.add("NNlib")'

RUN julia -e 'import Pkg; Pkg.add("JLD2")'

RUN julia -e 'import Pkg; Pkg.add("StatsBase")'

RUN julia -e 'import Pkg; Pkg.add("ArgParse")'

RUN julia -e 'import Pkg; Pkg.add("SymmetricFormats")'

RUN julia -e 'import Pkg; Pkg.add("NLsolve")'

RUN julia -e 'import Pkg; Pkg.add("CUDA")'

RUN julia -e 'import Pkg; Pkg.add("BatchedBLAS")'

# optional RUN julia -e 'using Pkg; Pkg.add("MKL")'



RUN bash tsn.sh init -t auto $(pwd)/src

RUN bash tsn.sh train cpu -n2 $(pwd)/src # more of a unit test

RUN bash tsn.sh unittest



# optional

# ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64

# RUN apt-get install --yes nvidia-gds

# ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64{LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# RUN julia -e 'ENV["LD_LIBRARY_PATH"]="/usr/local/cuda/lib64"; import Pkg; Pkg.build("CUDA")'