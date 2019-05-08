#! /usr/bin/env python
import sys

args = sys.argv[1:]
extracting = args and args[0] == '-e'
if extracting:
    args.pop(0)

for ii, arg in enumerate(args):
    senses = set()
    for line in open(arg, 'r'):
        for sense in line.strip().split(' ')[1:]:
            senses.add(sense)
    sys.stderr.write(f"{len(senses)} {arg}\n")
    if extracting:
        arg = '.'.join(arg.split('.')[:-1])+'.senses'
        with open(arg, 'w') as sf:
            for sense in sorted(list(senses)):
                sf.write(f"{sense}\n")
        sys.stderr.write(f"wrote {arg}\n")

sys.exit(0)
