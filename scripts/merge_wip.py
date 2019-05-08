#!/usr/bin/env python

import os,sys

def main(args):
    if not args:
        raise Exception("No arguments")
    if args[0].endswith('.py'):
        args.pop(0)
    
    if len(args) < 2:
        raise Exception("Needs two input file arguments")
    
    
if __name__ == '__main__':
    sys.exit(main(sys.argv))
