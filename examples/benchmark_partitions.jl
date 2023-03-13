using Pkg
#ENV["PYTHON"]= "/home/rjjarvis/.julia/conda/3/x86_64/bin/python3"
#Pkg.build("PyCall")
using PyCall
using SparseArrays
using BlockArrays
using UnicodePlots
using Conda

function toinstallonly()
    Conda.add("scipy")
    Conda.add("numpy")
    @assert numpy = pyimport("numpy")
    @assert scipy = pyimport("scipy")
end
py"""
def split(array, nrows, ncols):
    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                .swapaxes(1, 2)
                .reshape(-1, nrows, ncols))

def get_arrays(connectome_size,numBlocks):
    import numpy as np
    my_v = np.random.random((connectome_size,connectome_size))
    return my_v
def get_blocks_py(connectome_size,numBlocks):
    S = get_arrays(connectome_size,numBlocks)
    lists =  split(S, int(connectome_size/numBlocks), int(connectome_size/numBlocks))
    return lists
"""
connectome_size = Int(round(100000/2.5))
numBlocks = 8
function get_blocks(connectome_size,numBlocks)
    partition_format = [Int(round(connectome_size/numBlocks)) for i in 1:numBlocks]
    return BlockArray(sprand(connectome_size,connectome_size,0.1),partition_format,partition_format)
end
@time w = get_blocks(connectome_size,numBlocks)

@show(size(w))
@show(size(blocks(w)[1]))

@time numpy_matrix = py"get_blocks_py"(connectome_size,numBlocks)
