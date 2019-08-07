#!/usr/bin/python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Open log file
if len(sys.argv) != 2:
	print("usage: plotter.py <log_file>")
	sys.exit(2)

file_in = sys.argv[1]
df = pd.read_csv(file_in)

# Sort by addr
df.sort_values("Addr", kind='mergesort', inplace = True)

# Update address, and also check for RMW
addr = df.Addr.to_numpy()
idx = df.index.to_numpy()
r_w = df.R0_W1.to_numpy()

new_addr = 0
prev_addr = 0
prev_type = 1
items = len(idx)
for i in range(items):
    curr_addr = addr[i]
    curr_type = r_w[i]
    if (prev_addr != curr_addr):
        #diff address, just update new_addr
        new_addr = new_addr + 1
        prev_addr = curr_addr
    elif (prev_type == 0 and curr_type == 1):
        #same address and RMW
        r_w[i-1] = 3      #invalid
        r_w[i] = 2        #rmw
    prev_type = curr_type
    addr[i] = new_addr

# Cut read/write/modify values
X = np.zeros((4,items), dtype=int)
Y = np.zeros((4,items), dtype=int)
cnt = np.zeros(4, dtype=int)
for i in range(items):
    tp = r_w[i]
    X[tp][cnt[tp]] = idx[i]
    Y[tp][cnt[tp]] = addr[i]
    cnt[tp] = cnt[tp] + 1

# Plotting
plt.figure(figsize=[15,10])
plt.plot(X[0][0:cnt[0]], Y[0][0:cnt[0]], 'r^', alpha=0.2, label='Read')
plt.plot(X[1][0:cnt[1]], Y[1][0:cnt[1]], 'b.', alpha=0.2, label='Write')
plt.plot(X[2][0:cnt[2]], Y[2][0:cnt[2]], 'gs', alpha=0.2, label='Read-Modify-Write')
plt.legend()
plt.show()

