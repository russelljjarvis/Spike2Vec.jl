import pickle
import pandas
import pytz
def get_pablo_sim():

    temp = pickle.load(open("spikes.dat","rb"))
    nodes = temp.i.values
    times = temp.t.values
    pickle.dump([nodes,times],open("pablo_conv.p","wb"))
    return None

get_pablo_sim()