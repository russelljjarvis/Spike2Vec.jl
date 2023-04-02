

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
This code is to compare the speed of matrix partitioning in Julia versus Python.
Note in Julia its normal to put documentation strings above method definitions.
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
# This is an example of a Python code string that will be plugged into a Python executor from within the Julia namespace.
# The code partitions arrays into equal block sizes.

def get_arrays(connectome_size,numBlocks):
    # First lets declare an array.
    import numpy as np
    my_v = np.random.random((connectome_size,connectome_size))
    return my_v

def split(array, nrows, ncols):
    # now lets declare a method to partition an array into smaller equally sized blocks.
    r, h = array.shape
    return (array.reshape(h//nrows, nrows, -1, ncols)
                .swapaxes(1, 2)
                .reshape(-1, nrows, ncols))

def get_blocks_py(connectome_size,numBlocks):
    S = get_arrays(connectome_size,numBlocks)
    lists =  split(S, int(connectome_size/numBlocks), int(connectome_size/numBlocks))
    return lists
"""


function get_blocks(connectome_size,numBlocks)
    target_matrix = sprand(connectome_size,connectome_size,0.1) # declares a sparse compressed matrix format container.
    partition_format = [Int(round(connectome_size/numBlocks)) for i in 1:numBlocks]
    @time ba = BlockArray(target_matrix,partition_format,partition_format) # This lines partitions the array into smaller equally sized blocks in Julia.
    return ba

end

function do_speed_comparison(connectome_size,numBlocks)
    @time w = get_blocks(connectome_size,numBlocks)
    @show(size(w))
    @time numpy_matrix = py"get_blocks_py"(connectome_size,numBlocks)
end
const connectome_size = Int(round(100000/2.5)) # this was a number that worked in a timely manner with 32GB of RAM
const numBlocks = 8
do_speed_comparison(connectome_size,numBlocks)
            
#=            
# First lets declare an array.
const connectome_size = Int(round(10000))
global target_matrix = sprand(connectome_size,connectome_size,0.01)
             
function smart_partition()
    A = max.(target_matrix, target_matrix')
    @time result0 = MatrixNetworks.spectral_cut(A)
    return result0,target_matrix
end
result0,target_matrix = smart_partition(target_matrix)
result0
=#
