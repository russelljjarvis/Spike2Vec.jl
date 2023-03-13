

using PyCall
using SparseArrays
using BlockArrays
using Plots
using UnicodePlots
using Conda
using Pkg
unicodeplots()
using MatrixNetworks
using Pkg

"""
This code is just to show speed of matrix partitioning versus Python.

"""
function toinstallonly()
    Pkg.add("MatrixNetworks")
    Pkg.add("Conda")
    ENV["PYTHON"]= "/home/rjjarvis/.julia/conda/3/x86_64/bin/python3"
    Pkg.build("PyCall")
    
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
    target_matrix = sprand(connectome_size,connectome_size,0.1)
    return BlockArray(target_matrix,partition_format,partition_format)
end

function do_fast()
    @time w = get_blocks(connectome_size,numBlocks)

    @show(size(w))
#       @show(size(blocks(w)[1]))

    @time numpy_matrix = py"get_blocks_py"(connectome_size,numBlocks)
end
function smart_partition()
    connectome_size = Int(round(10000))
    target_matrix = sprand(connectome_size,connectome_size,0.01)
    A = max.(target_matrix, target_matrix')
    @time result0 = MatrixNetworks.spectral_cut(A)
    #@time result1 = MatrixNetworks.dirclustercoeffs(target_matrix)
    result0
    #result1
    #UnicodePlots.spy(result0)
    #UnicodePlots.spy(result1)
    return result0,target_matrix#,result1
end
result0,target_matrix = smart_partition()
result0
