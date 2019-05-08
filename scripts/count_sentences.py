#! /usr/bin/env python
import sys
for arg in sys.argv[1:]:
    opens = 0
    closes = 0
    nline = 0
    for line in open(arg, 'r'):
        nline += 1
        if line.startswith('<sentence'):
            opens += 1
            if opens - closes != 1:
                sys.stderr.write(f"{opens - closes} at {nline}\n")
                sys.exit(0)
        elif line.startswith('</sentence'):
            closes += 1
            if opens - closes != 0:
                sys.stderr.write(f"{opens - closes} at {nline}\n")
                sys.exit(0)
    print(f"{opens} {closes} {arg}")
