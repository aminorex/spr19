#! /usr/bin/env python
import sys
from collections import Counter

PROJ_ROOT = '/home/tony/csci/project/'
BERT_DIR = PROJ_ROOT + 'pytorch-pretrained-BERT'
sys.path.insert(0, BERT_DIR)
from pytorch_pretrained_bert import BertTokenizer

err = sys.stderr

MODEL_DIR = PROJ_ROOT + 'models/uncased_L-24_H-1024_A-16/'


def filter(args):
    tize = None
    strict, ambig, dictionary, matching = False, False, False, False

    # print(f"{len(args)} args {args}\n")
    while args and args[0][0] == '-':
        arg = args.pop(0)[1:]
        while arg:
            cmd = arg[0]
            arg = arg[1:]
            if cmd == 'a':  # disable ambiguity check
                ambig = not ambig
            elif cmd == 't':  # enable bert tokenizer
                # required for all checks except ambiguity
                tize = BertTokenizer.from_pretrained(MODEL_DIR)
            elif cmd == 's':  # disable special token check
                strict = not strict
            elif cmd == 'd':  # disable vocab token check
                dictionary = not dictionary
    if not args:
        raise Exception("No paths to filter")
    vocarg = args[0].endswith('vocab.txt')
    vocpath = args.pop(0) if vocarg else (MODEL_DIR + 'vocab.txt')
    with open(vocpath, 'r') as vf:
        vocab = frozenset([item.strip() for item in vf])
    if not args:
        raise Exception("No input path")

    # print(f"{len(args)} args: {args}\n")
    nfail, nout, nline = 0, 0, 0
    with open(args.pop(0), 'r') as inf:

        if not args:
            args = ['/dev/stdout']

        with open(args.pop(0), 'w') as outf:

            for line in inf:
                line = line.strip()
                nline += 1

                if not line:  # not counted as fail
                    continue

                ftoks = line.split(' ')
                insts = [ii for ii, tok in enumerate(ftoks)
                         if tok.find('\\') > 0]
                feats = [ftoks[ii].split('\\') for ii in insts]
                words = [l[0] for l in feats]

                if ambig and any(len(a.split(',')) > 1
                                 for a in [l[2] for l in feats]): 
                    maxf = max(len(a.split(','))
                               for a in [l[2] for l in feats])
                    print(f"{nline} {maxf}\n")
                    nfail += 1  # skip ambiguity
                    continue

                # deannotate to match btoks
                for ii, word in enumerate(words):
                    ftoks[insts[ii]] = word

                # does not work if word pieces in vocab:
                # if strict and any(ftok not in vocab for ftok in ftoks):
                #   nfail += 1
                #   continue

                if tize:
                    btoks = tize.tokenize(line)

                    # disallow non-dictionary tokens
                    if dictionary and not all(btok in vocab
                                              for btok in btoks):
                        nfail += 1
                        continue

                    # disallow lines with special tokens
                    if strict and any(btok[0] == '[' and btok[-1] == ']'
                                      for btok in btoks):
                        nfail += 1
                        continue

                    # match words to wordpieces or die
                    ii, jj = (0, 0) if matching else (len(ftoks), len(btoks))
                    while ii < len(ftoks) and jj < len(btoks):
                        if ftoks[ii] == btoks[jj]:  # match, advance to next
                            ii += 1
                            jj += 1
                        elif btoks[jj+1].startswith('##'):

                            if len(btoks) < jj + 1:  # broken pieces
                                nfail += 1
                                continue

                            btok = btoks[jj]  #concat pieces
                            for contin in btoks[jj+1:]:
                                if not contin.startswith('##'):
                                    break
                                btok += contin[2:]
                                jj += 1
                                if jj > len(btoks) - 1:
                                    break

                            if btok != ftoks[ii]:  # compare concat to word
                                nfail += 1
                                continue

                        else:  # mismatched
                            nfail += 1
                            continue

                    # after while loop
                    if jj < len(btoks) or ii < len(ftoks):  # tail mismatch
                        nfail += 1
                        continue

                # after for loop
                outf.write(' '.join(ftoks)+"\n")
                nout += 1
    return nline, nout, nfail


if __name__ == "__main__":
    nline, nout, nfail = filter(sys.argv[1:])
    empty = nline - nout - nfail
    err.write(f"{nline} in, {nout} out, {nfail} omitted," +
              f"{empty} empty\n")
    sys.exit(0)
