import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from argparse import ArgumentParser
from sklearn.manifold import TSNE

parser = ArgumentParser()
parser.add_argument("src", type=str, help="File of descriptors and image names")
parser.add_argument("--normalise", action="store_true", help="Normalise the descriptors")
parser.add_argument("--metric", type=str, default="euclidean", help="Distance function to use [euclidean, matching, cosine, correlation, hamming]")
args = parser.parse_args()

def normalise(x):
    return x/np.linalg.norm(x)

def string2ndarray(x, dtype=np.float32):
    # Remove BS which pandas adds to numpy array string    
    x = x.replace("\n","").replace("[","").replace("]","").replace(",","")
    x = re.sub('\s+', ' ', x).strip().split(" ")
    return np.asfarray(x, dtype)

def extract_img_id(x):
    return x.rsplit('_', 1)[0]

df = pd.read_csv(args.src)

# Get the data back into the format we want
df['opt_id'] = df['opt'].apply(extract_img_id)
df['sar_id'] = df['sar'].apply(extract_img_id)

# indices = [i for i, s in enumerate(mylist) if 'aa' in s]

df['z_sar'] = df['z_sar'].apply(string2ndarray)
df['z_opt'] = df['z_opt'].apply(string2ndarray)

dfnm = df.copy() 

df = df.loc[df['sar'] == df['opt']]
dfnm = dfnm.loc[dfnm['sar'] != dfnm['opt']]
df = df.reset_index(drop=True)
dfnm = dfnm.reset_index(drop=True)

z_sar = np.stack(df['z_sar'].values)
z_opt = np.stack(df['z_opt'].values)

z_sarnm = np.stack(dfnm['z_sar'].values)
z_optnm = np.stack(dfnm['z_opt'].values)

if args.normalise:
    z_sar = np.apply_along_axis(normalise ,1 , z_sar)
    z_opt = np.apply_along_axis(normalise ,1 , z_opt)
    z_sarnm = np.apply_along_axis(normalise ,1 , z_sarnm)
    z_optnm = np.apply_along_axis(normalise ,1 , z_optnm)

dists = cdist(z_sar, z_opt, metric=args.metric)

plt.imshow(dists, cmap="jet")
plt.show()

idxs = np.zeros(dists.shape[0])

for i, row in enumerate(dists):
    order = np.argsort(row)
    idx = np.argwhere(order == i)[0]
    idxs[i] = idx

top_n = np.zeros(dists.shape[0])

for i in range(25):
    top_n[i] = np.sum(idxs < i+1)
    print(f"Top {i+1}: {np.round(top_n[i]/len(idxs)*100, 2)}")

import code
code.interact(local=locals())

MAX = 100

Zo_2d = TSNE(n_components=2).fit_transform(z_opt[:MAX])
Zs_2d = TSNE(n_components=2).fit_transform(z_sar[:MAX])
# https://www.kaggle.com/gaborvecsei/plants-t-sne
from matplotlib import pyplot as plt
plt.figure(figsize=(6, 5))
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple']
for i, (sar, opt) in enumerate(zip(Zs_2d, Zo_2d)):
    plt.scatter(sar[0], sar[1], c=colors[i%10], label="s_{}".format(df["sar"].values[i]))
    plt.scatter(opt[0], opt[1], c=colors[i%10], label="o_{}".format(df["opt"].values[i]))

plt.legend()
plt.show()

# for i, c, label in zip(target_ids, colors, df["opt"].values):
#     plt.scatter(Z_2d[i, 0], Z_2d[i, 1], c=c, label=label)
# plt.legend()
# plt.show()

# import code
# code.interact(local=locals())

# print(df.head())