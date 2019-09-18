#!/usr/bin/env python3

import subprocess
import shlex

def build():
	cmd = "make"
	subprocess.run(shlex.split(cmd), check=True, stdout=subprocess.DEVNULL);


def start_background(thrds):
	cmd = "./back", str(thrds)
	return subprocess.run(shlex.split(cmd), check=True);


def stop_background(back_proc):
	back_proc.terminate();


def run_workload():
	cmd = "./workload"
	results_str = subprocess.check_output(shlex.split(cmd), universal_newlines=True)



print("=======================================================")
print("  Background          Sequential           Random      ")
print("   Threads            BW (MB/s)           BW (MB/s)    ")
print("=======================================================")


for i in range(1,4):


