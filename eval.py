import pickle
import sys
from matplotlib import pyplot as plt
path = sys.argv[1]
aname = sys.argv[2]
from matplotlib import pyplot as plt
f = open(path + '/' + aname + '_AUC.pkl', 'rb')
auc = pickle.load(f)
f.close()
plt.plot(range(len(auc)), auc)
plt.savefig(path + '/' + aname + '_100epochs.png')
