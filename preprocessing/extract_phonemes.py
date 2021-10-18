#!/usr/bin/python3

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input', help='file with perl script output', required=True)
parser.add_argument('--output', help='the name of the output file', required=True)

args = parser.parse_args()

with open(args.input) as file:
    lines = file.readlines()

string = []
for line in lines:
    try:
        start = line.index('-')
        end = line[start:].index('+')
        if end <= 3:
            phoneme = line[start + 1:start+end]
            string.append(phoneme)
        else:
            print(line)
    except ValueError:  # line.index can't find "-"
        #print(line)
        pass

string = " ".join(string)
with open(args.output, 'w') as file:
     file.write(string)
