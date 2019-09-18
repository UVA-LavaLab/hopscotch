#!/usr/bin/env python3

###############################################################################
#
# Plots latency with varying working set size.
#
# Author: 	Alif Ahmed
# email: 	alifahmed@virginia.edu
# Updated: 	Aug 06, 2019
#
###############################################################################
import argparse
import json
import subprocess
import sys
import shlex
import matplotlib.pyplot as plt


###############################################################################
# Function definitions
###############################################################################

# Check if sval is a positive integer. Used for argument validity checking.
def positive_integer(sval):
	ival = int(sval)
	if ival <= 0:
		raise argparse.ArgumentTypeError("%s must be a positive integer" % sval)
	return ival


# Build and run the roofline kernel with different configurations
def run_bench(args):
	wss = []
	latency = []
	curr_wss = args.wss_min
	while(curr_wss <= args.wss_max):
		# make
		make_cmd = 	"make kernel USER_DEFS=\"-DWSS_EXP=" + str(curr_wss) + "\""
		subprocess.run(shlex.split(make_cmd), check=True, stdout=subprocess.DEVNULL);

		# run kernel
		results_str = subprocess.check_output(shlex.split("./kernel"), universal_newlines=True)
		print(results_str, end='')
		results_arr = results_str.split()
		wss.append(float(results_arr[0]))
		latency.append(float(results_arr[1]))
		curr_wss = curr_wss + 1
	return wss, latency


###############################################################################
# Arguments
###############################################################################
parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument('--wss-min', default='13', type=positive_integer, metavar='X',
					help='Minimum working set size = (2 ^ wss_min) bytes. (default: %(default)s)')

parser.add_argument('--wss-max', default='32', type=positive_integer, metavar='X',
					help='Maximum working set size = (2 ^ wss_max) bytes. (default: %(default)s)')

parser.add_argument('--plot-file', default='latency.pdf', type=str, metavar='X',
					help='Latency plot will be saved in this file. (default: %(default)s)')

args = parser.parse_args()


###############################################################################
# Validate arguments
###############################################################################
if (args.wss_min > args.wss_max):
	raise argparse.ArgumentTypeError("--wss-min must be <= --wss-max.")


###############################################################################
# Run kernel and plot the results
###############################################################################
print("")
print("================================================================================")
print("       Working Set Size (Byte)                               Latency (ns)       ")
print("================================================================================")


# set plot properties
plt.figure()
plt.xlabel('Working Set Size (Byte)')
plt.ylabel('Latency (ns)')
plt.xscale('log')
plt.yscale('log')
plt.title('Latency Plot')
plt.grid(which='major', axis='both')
plt.grid(which='minor', axis='both', linestyle=':')

# run kernel
wss, latency = run_bench(args)
plt.plot(wss, latency, '-bo')

# save plot
plt.savefig(args.plot_file, format='pdf', bbox_inches='tight')

print("================================================================================")
print("Latency plot saved as " + args.plot_file)
print("")








