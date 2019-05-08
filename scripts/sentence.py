#! /usr/bin/env python

import sys
import re

err = sys.stderr

PROJ_ROOT = '/home/tony/csci/project/'
MAP_PATH = PROJ_ROOT + 'data/word_sense_disambigation_corpora/combined_map.txt'
GOLD_KEY = PROJ_ROOT + 'data/WSD_Unified_Evaluation_Datasets/ALL/ALL.gold.key.txt'
SENSE_MAP = dict()


def load_map(map_path,sep='\t'):
    n = 0
    with open(map_path, 'r') as algmap:
        for line in algmap:
            line = line.strip()
            if not line:
                break
            ab = line.split(sep)
            k = ab[0]
            j = ','.join(ab[1:])
            SENSE_MAP[k] = j
            n += 1
    err.write(str(n)+" senses mapped\n")


re_id = re.compile(r"id=\"([\w\d\.]+)\"")
re_text = re.compile(r"\>([^\<]+)")
re_sid = re.compile(r"[semeval|senseval][\w\d\.]+ ")
re_lemma = re.compile(r"lemma=\"([^\"]*)\"")
re_word = re.compile(r"text=\"([^\"]*)\".*break_level=\"([A-Z_]*)\"")
re_sense = re.compile(r"lemma=\"([^\"]*)\".*sense=\"([^\"]*)\"")
re_deannot = re.compile(r"\\[^ ]* ")


def dexmlify(option=''):
    spos = 0
    nline = 0
    nword = 0
    counting = 'c' in option
    with open(sys.argv[1]) as inf:
        outf = open(sys.argv[2], 'w')
        for line in inf:
            nline += 1
            line = line.strip()
            if line.startswith('<word'):
                m = re.search(re_word, line)
                if not m:
                    raise Exception("Bad word line "+nline)
                text, brk = m.group(1), m.group(2)
                m = re.search(re_sense, line)
                if brk.startswith('SENTENCE'):
                    sep = '\n' if spos else ''
                    spos = 0
                else:
                    sep = ' '
                outf.write(sep + text)
                spos += 1
                nword = 1
                if m:
                    lemma, sense = m.group(1), m.group(2)
                    if sense not in SENSE_MAP:
                        err.write("Missing sense "+sense+" @"+str(nline)+"\n")
                        if s in option:
                            continue
                    else:
                        sense = SENSE_MAP[sense]
                    outf.write("\\"+lemma+"\\"+sense)
            elif line.startswith('<sentence'):
                id = re.search(re_id, line).group(1)
                words = []
            elif line.startswith('<wf'):
                text = re.search(re_text, line).group(1)
                words.append(text)
            elif line.startswith('<instance'):
                id = re.search(re_id, line).group(1)
                text = re.search(re_text, line).group(1)
                if not SENSE_MAP:
                    words.append(text)
                    continue
                if id not in SENSE_MAP:
                    err.write("Missing sense "+id+" @"+str(nline)+"\n")
                    continue
                senses = SENSE_MAP[id]
                lemma = senses.split('%')[0]
                words.append(f"{text}\\{lemma}\\{senses}")
            elif line.startswith('</sentence'):
                if counting:
                    nwords = len(words)
                    outf.write(f"{nwords} ")
                outf.write(' '.join(words)+'\n')
        if spos:
            outf.write('\n')
        outf.close()
        return


def merge(option=''):
    nline = 0
    nout = 0
    gold = dict()
    with open(sys.argv[2], 'r') as inf:
        for line in inf:
            nodes = line.strip().split(' ')
            gold[nodes[0]] = ','.join(nodes[1:])
    with open(sys.argv[1], 'r') as inf:
        with open(sys.argv[3], 'w') as outf:
            text = ''
            for line in inf:
                nline += 1
                line = line.strip()
                if line.startswith('<sentence'):
                    if text:
                        nout += 1
                        outf.write(text+"\n")
                    text = ''
                elif line.startswith('<wf'):
                    text += re.search(re_text, line).group(1) + ' '
                elif line.startswith('<instance'):
                    id = re.search(re_id, line).group(1)
                    if id not in gold:
                        err.write(f"Missing {id} at {nline}\n")
                        text = ''
                        continue
                    word = re.search(re_text, line).group(1)
                    lemma = re.search(re_lemma, line).group(1)
                    text += f"{word}\\{lemma}\\{gold[id]} "
            if text:
                nout += 1
                outf.write(text+"\n")
    err.write(f"{nout} output\n")


def sentenceify(mode):
    with open(sys.argv[1]) as inf:
        with open(sys.argv[2], 'w') as outf:
            for line in inf:
                if mode == '2':
                    line = re.sub(re_deannot, ' ', line.strip().lower())
                else:
                    line = ' '.join(x.split('%')[0]
                                    for x in
                                    line.strip().lower().split(' ')[1:])
                outf.write(line+"\n")


if __name__ == "__main__":
    final = sys.argv[-1]
    if 'x' in final:
        if 'n' not in final:
            # include senses
            load_map(GOLD_KEY, ' ')
        dexmlify(sys.argv[-1][1:])
    elif 'm' in final:
        merge(sys.argv[-1][1:])
    else:
        sentenceify(sys.argv[-1])
