using SparseArrays
using BlockArrays
using UnicodePlots
using PyCall
using Conda

Conda.add("scipy")
Conda.add("numpy")
py"""
def get_arrays(connectome_size,numBlocks)
    import numpy
    from scipy.sparse import random
    from scipy import stats
    from numpy.random import default_rng
    rng = default_rng()
    rvs = stats.poisson(25, loc=10).rvs
    S = random(connectome_size, connectome_size, density=1, random_state=rng, data_rvs=rvs)
    return S
def get_blocks_py(connectome_size,numBlocks)
    S = get_arrays(connectome_size,numBlocks)
    return np.split(S, 8)
"""
connectome_size = 111000
numBlocks = 8
function get_blocks(connectome_size,numBlocks)

    partition_format = [Int(round(connectome_size/numBlocks)) for i in 1:numBlocks]
    
    return BlockArray(sprand(connectome_size,connectome_size,0.1),partition_format,partition_format)

end
@time w = get_blocks(connectome_size,numBlocks)

@show(size(w))
@show(size(blocks(w)[1]))

@time numpy_matrix = "get_blocks_py"(connectome_size,numBlocks)


@show(size(numpy_matrix))
@show(size(numpy_matrix)[1])