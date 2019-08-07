#!/usr/bin/python3

import argparse
import json
import subprocess
import sys
import shlex
import matplotlib.pyplot as plt

def positive_integer(sval):
	ival = int(sval)
	if ival <= 0:
		raise argparse.ArgumentTypeError("%s must be a positive integer" % sval)
	return ival

def run_bench(data_type, args):
	ai = []
	bw = []
	perf = []
	curr_flop = args.flop_min
	while(curr_flop <= args.flop_max):
		make_cmd = 	"make kernel USER_DEFS=\"-DFLOPS_PER_ELEM=" + str(curr_flop) + " -DDATA_T_ENC=" + data_type \
					+ " -DHS_ARRAY_SIZE_MB=" + str(args.working_set_size) \
					+ " -DNTRIES=" + str(args.ntries) + "\""
		subprocess.run(shlex.split(make_cmd), check=True, stdout=subprocess.DEVNULL);
		#run kernel
		results_str = subprocess.check_output(shlex.split("./kernel"), universal_newlines=True)
		print(results_str, end='')
		results_arr = results_str.split()
		ai.append(float(results_arr[2]))
		bw.append(float(results_arr[3]))
		perf.append(float(results_arr[4]))
		curr_flop = int(curr_flop * args.flop_rate)

	return ai, bw, perf

parser = argparse.ArgumentParser(allow_abbrev=False)

parser.add_argument('--working-set-size', default='2048', type=positive_integer, metavar='X',
					help='Working set size in MB. (default: %(default)s)')

parser.add_argument('--ntries', default='10', type=positive_integer, metavar='X',
					help='Number of runs for each configuration (default: %(default)s)')

parser.add_argument('--flop-min', default='1', type=positive_integer, metavar='X',
					help='Minimum value of FLOPs per data element. (default: %(default)s)')

parser.add_argument('--flop-max', default='8192', type=positive_integer, metavar='X',
					help='Maximum value of FLOPs per data element. Actual value may differ depending on --min-flops and --flop-rate. (default: %(default)s)')

parser.add_argument('--flop-rate', default='2', type=float, metavar='X',
					help='Increament rate of FLOPs per data element. Refer to README for detail. (default: %(default)s)')

parser.add_argument('--disable-sp', action='store_true',
					help='Disable roofline plot for single precision floating point. Enabled by default.')

parser.add_argument('--disable-dp', action='store_true',
					help='Disable roofline plot for double precision floating point. Enabled by default.')

parser.add_argument('--plot-file', default='roofline.pdf', type=str, metavar='X',
					help='Roofline plot will be saved in this file. (default: %(default)s)')

args = parser.parse_args()

if (args.flop_min > args.flop_max):
	raise argparse.ArgumentTypeError("--flop-min must be <= --flop-max.")

if (args.flop_rate <= 1):
	raise argparse.ArgumentTypeError("--flop-rate must be greater than 1.0")


print("")
print("Working set size: " + str(args.working_set_size) + " MB")
print("================================================================================")
print("Type         FLOP/elem           AI            BW (GB/s)          Perf (GFLOP/s)")
print("================================================================================")

# run kernel for single precision floating point
plt.figure()
plt.xlabel('Arithmetic Intensity (FLOP/Byte)')
plt.ylabel('Performance (GFLOP/s)')
plt.xscale('log')
plt.yscale('log')
plt.title('Roofline Plot')
plt.grid(which='major', axis='both')
plt.grid(which='minor', axis='both', linestyle='--')

if (args.disable_sp == False):
	sp_ai, sp_bw, sp_perf = run_bench("1", args)
	plt.plot(sp_ai, sp_perf, '-bo', label='Single Precision')


# run kernel for double precision floating point
if (args.disable_dp == False):
	dp_ai, dp_bw, dp_perf = run_bench("0", args)
	plt.plot(dp_ai, dp_perf, '-r^', label='Double Precision')

plt.legend()
plt.savefig(args.plot_file, format='pdf', bbox_inches='tight')

print("================================================================================")
print("Roofline plot saved as " + args.plot_file)
print("")








