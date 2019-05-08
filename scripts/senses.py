#! /usr/bin/env python
import sys
from collections import Counter

err = sys.stderr


def filter(args):

    ambig = 0
    if args and args[0][0] == '-':
        ambig = int(args.pop(0)[1:])

    if not args:
        raise Exception("No paths to filter")

    if not args:
        args = ['/dev/stdin']

    c = Counter()

    with open(args.pop(0), 'r') as inf:

        if not args:
            args = ['/dev/stdout']

        with open(args.pop(0), 'w') as outf:

            nline = 0

            for line in inf:
                line = line.strip()
                nline += 1
                ftoks = line.split(' ')
                insts = [ii for ii, tok in enumerate(ftoks)
                         if tok.find('\\') > 0]
                for inst, feats in [(ii, ftoks[ii].split('\\'))
                                    for ii in insts]:
                    senses = feats[2].split(',')
                    if not senses:
                        err.write(f"Error in {feats[0]}\n")
                        continue
                    sno = nline - 1
                    nsense = len(senses)
                    if nsense == ambig or not ambig:
                        c[nsense] += 1
                        outf.write(f"{sno}\t{inst}\t{nsense}\t" +
                                   "\t".join(senses)+"\n")
    err.write(str(c)+"\n")
    return nline


if __name__ == "__main__":
    err.write(str(filter(sys.argv[1:])) + "\n")
    sys.exit(0)
