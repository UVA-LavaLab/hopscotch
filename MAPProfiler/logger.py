#!/usr/bin/env python3

###############################################################################
#
# Memory access pattern logger. Creates memory trace file by instrumenting with
# Intel Pin tool.
#
# Author:   Alif Ahmed
# email:    alifahmed@virginia.edu
# Updated:  Aug 06, 2019
#
###############################################################################
import sys
import os
import argparse
import subprocess


###############################################################################
# Program parameters
###############################################################################
#current version
version_string = 'v1.0'

#target object file for Pin (path relative to the script)
obj_file = 'obj-intel64/MAPProfiler.so'


###############################################################################
# Argument definitions
###############################################################################
parser = argparse.ArgumentParser(allow_abbrev=False, 
                    description='Memory Access Pattern Logger ' + version_string)

parser.add_argument('--version', action='version', version=version_string)

parser.add_argument('--pin-bin', metavar='<path-to-pin-binary>',
                    help="Path to Intel Pin binary. If not given, derives from \
                          PIN_ROOT environment variable instead. If PIN_ROOT is \
                          also not given, then uses 'pin' (assumes pin is in PATH).")

parser.add_argument('--read', type=int, choices=[0,1], default='1',
                    help='Controls read operation tracing. 0: disabled, 1: enabled \
                          (defalut).')

parser.add_argument('--write', type=int, choices=[0,1], default='1',
                    help='Controls write operation tracing. 0: disabled, 1: enabled \
                          (defalut).')

parser.add_argument('--stack-log', type=int, choices=[0,1], default='0',
                    help='Controls stack based access logging. 0: disabled (default), \
                          1: enabled.')

parser.add_argument('--ip-log', type=int, choices=[0,1], default='1',
                    help='Controls instruction pointer relative access logging. 0: disabled, 1: enabled \
                          (default).')

parser.add_argument('--trace-file', default='mem_trace.csv', metavar='<trace-file>',
                    help='Memory trace will be written to this file by Pin. The \
                          file is in simple human readable csv format. (default: %(default)s)')

parser.add_argument('--trace-limit', type=int, default='1000000', metavar='<limit>',
                    help='Maximum number of accesses to trace. (default: %(default)s)')

parser.add_argument('--max-threads', type=int, default='10000', metavar='<limit>',
                    help='Upper limit of threads for the program being profiled. \
                          (default: %(default)s)')

parser.add_argument('--func', default='main', metavar='<func-to-trace>',
                    help='Function to trace. Fully qualified name should be used for \
                          C++ functions. (default: %(default)s)')

parser.add_argument('cmd', metavar='<cmd>',
                    help='Command for running the program being profiled.')

parser.add_argument('cmd_args', nargs=argparse.REMAINDER, metavar='...')


###############################################################################
# Parse arguments
###############################################################################
args = parser.parse_args()
if args.pin_bin == None:
    pin_root = os.environ.get('PIN_ROOT')
    if pin_root == None:
        args.pin_bin = 'pin'
    else:
        args.pin_bin = pin_root + '/pin'

script_dir = os.path.dirname(os.path.abspath(__file__))
obj_full_path = script_dir + '/' + obj_file


###############################################################################
# Generate Pin command
###############################################################################
pin_cmd = [ args.pin_bin,
            '-t', obj_full_path,
            '-func', args.func,                 
            '-out', args.trace_file,
            '-lim', str(args.trace_limit),
            '-read', str(args.read),
            '-write', str(args.write),
            '-stack', str(args.stack_log),
            '-ip', str(args.ip_log),
            '--', args.cmd] + args.cmd_args
print("\n================================================================================")
print(' '.join(pin_cmd))
print("================================================================================")

###############################################################################
# Invoke Pin
###############################################################################
subprocess.run(pin_cmd)




