import os
import sys
import shlex
import subprocess
import itertools
import argparse
import pdb

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('script_name', type=str, help='Name of script to run')
parser.add_argument('search_space_file', type=str, help='Filename containing search_space')
args = parser.parse_args()
# Initialize the search_space
names = []
values = [] # list of lists
with open(args.search_space_file, 'r') as f:
    for line in f:
        tokens = line.rstrip('\r\n')
        split_tokens = tokens.split(' ')
        names.append(split_tokens[0])
        values.append(split_tokens[1:])


# Initialize the functions
def namer(names, values):
    base = 'run.'

    try:
        idx = names.index('-c')
        val = values[idx]
        base += 'cnn.' + str(val)
    except:
        base += 'no_cnn'

    if '--pos' in values:
        base += '.pos'
    if '--variety' in values:
        base += '.variety'
    if '--punct-count' in values:
        base += '.punct'
    return base


def main(script_name, argument_names, argument_values, function_arguments):
    arguments = argument_names
    combinations = itertools.product(*argument_values)
    # pdb.set_trace()
    for values in combinations:
        # pdb.set_trace()
        command = ['python', script_name]
        for i in range(len(arguments)):
            argument = arguments[i]
            value = values[i]
            command.extend([argument, value])
        for (fn_name, fn) in function_arguments:
            command.extend([fn_name, fn(arguments, values)])
        command = [x for x in command if x!='']
        print('Running: ', ' '.join(command))
        # pdb.set_trace()
        subprocess.call(command)
    return


if __name__ == '__main__':
    main(args.script_name, names, values, [('--nm', namer), ('-o', namer)])
