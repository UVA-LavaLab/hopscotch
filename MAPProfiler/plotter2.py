#!/usr/bin/python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

# Open log file
if len(sys.argv) != 5:
	print("usage: plotter2.py <log_file> <start_addr> <end_addr> <type= R or W or RW>")
	sys.exit(2)

file_in = sys.argv[1]
minAddr = float(sys.argv[2])
maxAddr = float(sys.argv[3])
plotType = sys.argv[4]

#read csv
df = pd.read_csv(file_in)

#cut read and write
if maxAddr < 0:
	dfR = df.loc[(df['R0_W1'] == 0) & (df['Addr'] >= minAddr)]
	dfW = df.loc[(df['R0_W1'] == 1) & (df['Addr'] >= minAddr)]
else:
	dfR = df.loc[(df['R0_W1'] == 0) & (df['Addr'] >= minAddr) & (df['Addr'] < maxAddr)]
	dfW = df.loc[(df['R0_W1'] == 1) & (df['Addr'] >= minAddr) & (df['Addr'] < maxAddr)]

print(dfR)

# Plotting
plt.figure(figsize=[15,10])
if plotType == "R":
	plt.plot(dfR['Addr'], 'r^', alpha=0.2, label='Read')
	plt.legend()
	plt.show()
elif plotType == "W":
	plt.plot(dfW['Addr'], 'b.', alpha=0.2, label='Write')
	plt.legend()
	plt.show()
elif plotType == "RW":
	plt.plot(dfR['Addr'], 'r^', alpha=0.2, label='Read')
	plt.plot(dfW['Addr'], 'b.', alpha=0.2, label='Write')
	plt.legend()
	plt.show()
else:
	print("Wrong type. Should be R/W/RW")
	print("usage: plotter2.py <log_file> <start_addr> <end_addr> <type= R or W or RW>")

