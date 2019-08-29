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
		subprocess.run(shlex.split("make clean"), check=True, stdout=subprocess.DEVNULL);
		make_cmd = 	"make kernel USER_DEF=\"-DHS_ARRAY_ELEM=" + str(curr_wss//8) + "\""
		subprocess.run(shlex.split(make_cmd), check=True, stdout=subprocess.DEVNULL);
		#run kernel
		results_str = subprocess.check_output(shlex.split("./kernel"), universal_newlines=True)
		print(results_str, end='')
		results_arr = results_str.split()
		wss.append(float(results_arr[0]))
		latency.append(float(results_arr[1]))
		curr_wss = int(curr_wss * args.wss_rate)
	return wss, latency


###############################################################################
# Arguments
###############################################################################
parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument('--wss-min', default='8192', type=positive_integer, metavar='X',
					help='Minimum working set size in byte. (default: %(default)s)')

parser.add_argument('--wss-max', default='1073741824', type=positive_integer, metavar='X',
					help='Maximum working set size in byte. (default: %(default)s)')

parser.add_argument('--wss-rate', default='2', type=float, metavar='X',
					help='Increament rate of working set size. (default: %(default)s)')

parser.add_argument('--plot-file', default='latency.pdf', type=str, metavar='X',
					help='Latency plot will be saved in this file. (default: %(default)s)')

args = parser.parse_args()


###############################################################################
# Validate arguments
###############################################################################
if (args.wss_min > args.wss_max):
	raise argparse.ArgumentTypeError("--wss-min must be <= --wss-max.")

if (args.wss_rate <= 1):
	raise argparse.ArgumentTypeError("--wss-rate must be greater than 1.0")


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








