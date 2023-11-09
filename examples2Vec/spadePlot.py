import pickle
import viziphant
import matplotlib.pyplot as plt

with open("zebra_spade.p","rb") as f:
     [pat,train] = pickle.load(f)
viziphant.patterns.plot_patterns(train, pat)
plt.show()
