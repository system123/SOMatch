from argparse import ArgumentParser
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import math

parser = ArgumentParser()
parser.add_argument("lr", type=str, help="Path to LR CSV file from tensorboard")
parser.add_argument("loss", type=str, help="Path to Loss CSV file from tensorboard")
parser.add_argument("--clip", type=float, help="Only plot until a LR of n")
args = parser.parse_args()

lr = pd.read_csv(args.lr)['Value'].values
loss = pd.read_csv(args.loss)['Value'].values

if args.clip:
    lr = lr[np.where(lr <= args.clip)]

# Solve some common issues
if len(loss) != len(lr):
    l = min(len(loss), len(lr))
    r = math.ceil(len(lr)/len(loss))
    loss = loss if len(lr) > len(loss) else loss[::r]
    lr = lr if len(lr) < len(loss) else lr[::r]

    loss = loss[:l]
    lr = lr[:l]

    print(f"Loss: {len(loss)} LR: {len(lr)}")

plt.semilogx(lr, loss)
plt.show()
