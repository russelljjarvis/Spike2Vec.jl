using PyCall
py"""
def get_spikes(): 
    import pickle; 
    return pickle.load(open("pablo_conv.p","rb"))
"""
(n,t)=py"get_spikes"()