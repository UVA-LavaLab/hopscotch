#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

res = pd.read_csv('results.csv')
print(res)

# set plot properties
#plt.figure()
fig, ax1 = plt.subplots()
ax1.set_xlabel('Load AI (FLOP/Byte)')
ax1.set_ylabel('Load BW (GB/s)', color='r')
ax1.tick_params(axis='y', colors='red')
ax1.set_xscale('log')
#ax1.set_yscale('log')

plt.grid(which='major', axis='both')
plt.grid(which='minor', axis='both', linestyle=':')

ax1.plot(res['ai'], res['bw'], '-r^', label='Load BW')

ax2 = ax1.twinx()

ax2.plot(res.ai, res.lat, '-bs', label='Latency')
ax2.set_ylabel('Latency (ns)', color='b')
ax2.tick_params(axis='y', colors='blue')

#plt.plot(dp_ai, dp_perf, '-r^', label='Performance (DP)')
mb = max(res.perf) / max(res.bw)
ax1.axvline(x=mb, color='r', linestyle='--', label='Machine Balance')

# save the plot
plt.savefig('loaded_latency.pdf', format='pdf', bbox_inches='tight')









