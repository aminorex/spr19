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


re_id = re.compile(r"id=\"([\w\d\.]+)\"")
re_text = re.compile(r"\>([^\<]+)")
re_sid = re.compile(r"[semeval|senseval][\w\d\.]+ ")
re_lemma = re.compile(r"lemma=\"([^\"]*)\"")
re_word = re.compile(r"text=\"([^\"]*)\".*break_level=\"([A-Z_]*)\"")
re_sense = re.compile(r"lemma=\"([^\"]*)\".*sense=\"([^\"]*)\"")
re_deannot = re.compile(r"\\[^ ]* ")


def remove_ids(inpath, outpath, filter_ids=frozenset(), option=''):
    nline = 0
    nsent = 0
    nskip = 0
    enabled = True
    
    with open(inpath, 'r') as inf:
        with open(outpath, 'w') as outf:
            for line in inf:
                nline += 1
                if line.startswith('<sentence'):
                    id = re.search(re_id, line).group(1)
                    enabled = id not in filter_ids:
                    nsent += 1
                    if not enabled:
                        nskip += 1
                elif not enabled:
                    if line.startswith('<\sentence') and not enabled:
                        enabled = True
                        continue
                if enabled:
                    outf.write(line)
        return nline, nsent, nskip


if __name__ == "__main__":
    with open(sys.argv[1], 'r') as idf:
        ids = frozenset(line.strip() for line in idf)
    nline, nsent, nskip = remove_ids(sys.argv[2], sys.argv[3], ids)
    nid = len(ids)
    err.write(f"{nline} {nsent} {nskip} {nid}\n")
    sys.exit(0)

