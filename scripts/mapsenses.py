#! /usr/bin/env python

import sys
import re

err = sys.stderr

PROJ_ROOT = '/home/tony/csci/project/'
MAP_PATH = PROJ_ROOT + 'data/word_sense_disambigation_corpora/combined_map.txt'
SENSE_MAP = dict()


def load_map():
    n = 0
    with open(MAP_PATH, 'r') as algmap:
        for line in algmap:
            line = line.strip()
            if not line:
                break
            n += 1
            ab = line.split("\t")
            SENSE_MAP[ab[0]] = ab[1]
    err.write(str(n)+" senses mapped\n")


re_word = re.compile(r"text=\"([^\"]*)\".*break_level=\"([A-Z_]*)\"")
re_sense = re.compile(r"lemma=\"([^\"]*)\".*sense=\"([^\"]*)\"")


def dexmlify(args):
    nline = 0
    args.pop(0)
    if not args:
        raise Exception("no arguments")
    skipbad = args[0] in [ '--skip', '-s', '-bad', '--bad', '-b']
    if skipbad:
        args.pop()
    if len(args) != 2:
        raise Exception("Requires input and output file paths")
    
    with open(args.pop(0)) as inf:
        with open(args.pop(0), 'w') as outf:
            text = ''
            for line in inf:
                nline += 1
                line = line.strip()
                if not line.startswith('<word'):
                    continue
                m = re.search(re_word, line)
                if not m:
                    if skipbad:
                        text = ''
                        continue
                    raise Exception(f"Bad word line {nline}")
                word, brk = m.group(1), m.group(2)
                if text:
                    if brk.startswith('SENTENCE'):
                        outf.write(text + "\n")
                        text = ''
                    else:
                        text += ' '
                m = re.search(re_sense, line)
                if m:
                    lemma, senses = m.group(1), m.group(2).replace(' ', ',')
                    for ii, sense in enumerate(senses.split(',')):
                        if sense not in SENSE_MAP:
                            err.write(f"Missing sense {sense} at {nline}\n")
                            if skipbad:
                                text = ''
                                continue
                        else:
                            senses[ii] = SENSE_MAP[sense]
                    sense = ','.join(senses)
                text += f"{word}\\{lemma}\\{sense}" if m else word
            if text:
                outf.write(text + "\n")
        return


if __name__ == "__main__":
    load_map()
    dexmlify(sys.argv)
    sys.exit(0)
