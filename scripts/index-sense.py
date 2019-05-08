#! /usr/bin/env python
# filter sense index to include only senses actually used in input sense list

import sys
err = sys.stderr

PROJ_ROOT = '/home/tony/csci/project/'
uses = frozenset(line.strip() for line in sys.stdin)
entries = dict()
for line in open(PROJ_ROOT+"supWSD/dict/index.sense"):
    elts = line.strip().split()
    if not elts:
        continue
    # lemma, tag = elts[0].split('%')
    if elts[0] in uses:
        sys.stdout.write(line)

sys.exit(0)
