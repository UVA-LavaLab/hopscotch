#!/usr/bin/env python3

import sys
import csv
import argparse
import pandas as pd
import matplotlib.pyplot as plt


###############################################################################
# Program parameters
###############################################################################
#current version
version_string = 'v1.0'

#types
READ_TYPE  = 0
WRITE_TYPE = 1
RMW_TYPE   = 2


###############################################################################
# Argument definitions
###############################################################################
parser = argparse.ArgumentParser(allow_abbrev=False, 
                    description='Memory Access Pattern Plotter ' + version_string)

parser.add_argument('--version', action='version', version=version_string)

parser.add_argument('trace_file', metavar='<trace-file>',
                    help='Trace file to plot.')

parser.add_argument('--min-x', type=int, default='-1', metavar='<min-x>',
                    help='Minimum value for x-axis range. Use -1 to ignore. \
                          (default: %(default)s)')

parser.add_argument('--max-x', type=int, default='-1', metavar='<max-x>',
                    help='Maximum value for x-axis range. Use -1 to ignore. \
                          (default: %(default)s)')

parser.add_argument('--read', type=int, choices=[0,1], default='1',
                    help='Controls read operation tracing. 0: disabled, 1: enabled (defalut).')

parser.add_argument('--write', type=int, choices=[0,1], default='1',
                    help='Controls write operation tracing. 0: disabled, 1: enabled (defalut).')

parser.add_argument('--rmw', type=int, choices=[0,1], default='1',
                    help='Controls merging of read and write to read-modify-write. \
                          0: disabled, 1: enabled (defalut).')

parser.add_argument('--const', type=int, choices=[0,1], default='0',
                    help='Controls if const address accesses are plotted. \
                          0: disabled (default), 1: enabled.')

parser.add_argument('--alpha-read', type=float, default='0.15', metavar='<alpha-read>',
                    help='Controls transparency of read points. (default: %(default)s)')

parser.add_argument('--alpha-write', type=float, default='0.15', metavar='<alpha-write>',
                    help='Controls transparency of write points. (default: %(default)s)')

parser.add_argument('--alpha-rmw', type=float, default='0.15', metavar='<alpha-rmw>',
                    help='Controls transparency of read-modify-write points. (default: %(default)s)')

args = parser.parse_args()


###############################################################################
# Read Trace CSV
###############################################################################
df = pd.read_csv(args.trace_file)

#convert to numpy array
addr = df.Addr.to_numpy()
rw_type = df.R0_W1.to_numpy()
num_rows = len(addr)
if (num_rows == 0):
    print("Nothing to plot... Quitting...")
    sys.exit(0)


###############################################################################
# Apply Filters
###############################################################################

#convert arguments to proper range
if (args.min_x < 0):
    args.min_x = 0
if (args.max_x < 0) or (args.max_x > num_rows):
    args.max_x = num_rows

entries = {'R0_W1':[], 'Addr':[]}
curr_idx = args.min_x
last_type = -1
last_addr = -1
while curr_idx < args.max_x:
    curr_addr = addr[curr_idx]
    curr_type = rw_type[curr_idx]
    curr_idx += 1
    
    # apply --read arg filter
    if (args.read == 0) and (curr_type == READ_TYPE):
        continue

    # apply --write arg filter
    if (args.write == 0) and (curr_type == WRITE_TYPE):
        continue
    
    # apply --const filter
    if (args.const == 0) and (last_type == curr_type) and (last_addr == curr_addr):
        continue

    # apply --rmw filter
    if (args.rmw == 1) and (last_addr == curr_addr) and  \
            (last_type == READ_TYPE) and (curr_type == WRITE_TYPE):
        #pop previous entry
        entries['R0_W1'].pop()
        entries['Addr'].pop()
        #add new entry
        entries['R0_W1'].append(RMW_TYPE)
        entries['Addr'].append(curr_addr)
    else:
        entries['R0_W1'].append(curr_type)
        entries['Addr'].append(curr_addr)
    last_addr = curr_addr
    last_type = curr_type

# create a new DataFrame object with filtered entries
df = pd.DataFrame(data=entries)

# sort by address (required by the next step of address compression)
df.sort_values("Addr", kind='mergesort', inplace=True)

# function for reducing the gap between addresses
def comp_addr(curr_addr):
    curr_gap = curr_addr - comp_addr.act
    if(curr_gap > comp_addr.MAX_GAP):
        comp_addr.conv += comp_addr.MAX_GAP
    else:
        comp_addr.conv += curr_gap
    comp_addr.act = curr_addr
    return comp_addr.conv

# static initial values for the comp_addr function 
comp_addr.act = 0
comp_addr.conv = 0
comp_addr.MAX_GAP = 512

# replace previous addresses with compressed addresses
df.Addr = df.Addr.apply(comp_addr)


###############################################################################
# Plot results
###############################################################################
# seperate reads, writes and read-modify-writes
dfR = df.loc[df.R0_W1 == READ_TYPE]
dfW = df.loc[df.R0_W1 == WRITE_TYPE]
dfRMW = df.loc[df.R0_W1 == RMW_TYPE]

# plot
plt.figure()
plt.title("Memory Access Pattern Visualizer")
plt.xlabel("Reference #")
plt.ylabel("Relative Address")
plt.plot(dfR.index, dfR.Addr, 'r^', alpha=args.alpha_read, label='Read')
plt.plot(dfW.index, dfW.Addr, 'b.', alpha=args.alpha_write, label='Write')
plt.plot(dfRMW.index, dfRMW.Addr, 'gs', alpha=args.alpha_rmw, label='Read-modify-write')
plt.legend(loc='upper left')
plt.show()




