import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("csv", help="Result CSV file")
args = parser.parse_args()

results = pd.read_csv(args.csv)

results["l2"] = np.sqrt( (results.hm_x_max - results.shift_x)**2 + (results.hm_y_max - results.shift_y)**2 )

count = []
for t in np.arange(0, results.l2.max(), 0.5):
    count.append(np.sum(results.l2 <= t))

count = np.array(count)
plt.plot(np.arange(0, results.l2.max(), 0.5), count/len(results))
plt.title("Threshold vs % Successful matches")
plt.show() 

plt.scatter(test.l2, test.nlog_match_loss)
plt.title("L2 error vs -log(matching loss)")
plt.show()

results["nnlog_match_loss"] = results.nlog_match_loss

# For each possible matching loss threshold count the number of regions where we managed to match accurately
counts = {1:[], 2:[], 3:[]}
counts2 = {1:[], 2:[], 3:[]}
c = ['r','b','k']
for k in counts.keys():
    for t in np.unique(results.nnlog_match_loss):
        counts[k].append( np.sum(results.loc[results.nnlog_match_loss >= t].l2 <= k)/len(results.loc[results.nnlog_match_loss >= t]) )
        counts2[k].append( np.sum(results.loc[results.nnlog_match_loss >= t].l2 > k)/len(results.loc[results.nnlog_match_loss >= t]) )

counts = {k: np.array(v) for k,v in counts.items()}
counts2 = {k: np.array(v) for k,v in counts2.items()}

[plt.plot(np.unique(results.nnlog_match_loss), counts[k], c[k-1]) for k in counts.keys()]
plt.title("ROCish")
plt.show()