#! /usr/bin/env python

import sys
from itertools import chain

err = sys.stderr


def filter(filtf, dataf, outf, includes, sents, words, lemmas, anyx):
    filts = set()
    for line in filtf:
        filts.add(line)
    ffilts = frozenset(filts)
    nfilt = len(ffilts)
    nout = 0
    nline = 0
    for line in dataf:
        nline += 1
        if lemmas:
            # extract lemmas from sentence
            lems = [tok.split('\\')[1]
                    for tok in line.strip.split()
                    if tok.find('\\') == -1]
            if words:
                # extract surface forms from sentence
                # and add the lemmas
                wrds = [tok.split('\\')[0]
                        for tok in line.strip.split()] + lems
                found = all(wrd in ffilts
                            for wrd in wrds)
            else:
                found = all(lemma in ffilts
                            for lemma in lems)
        elif words:
            # extract words from sentences
            tmp = [tok.split('\\')[0]
                   for tok in line.strip.split()
                   if tok.find('%') > 0]
            found = all(word in ffilts
                        for word in tmp)
        elif sents:
            # extract sense lists ,-sep
            tmp = [tok.split('\\')[2:]
                   for tok in line.strip().split()
                   if tok.find('%') > 1]
            # split lists into senses
            tmp = [x.split(',') for x in chain.from_iterable(tmp)]
            if anyx:
                found = any(sense in ffilts
                            for sense in chain.from_iterable(tmp))
            else:
                found = all(sense in ffilts
                            for sense in chain.from_iterable(tmp))
        else:
            found = line in ffilts
        if (found and includes) or not (found or includes):
            outf.write(line)
            nout += 1
    return nout, nfilt, nline


if __name__ == "__main__":
    narg = len(sys.argv)
    fps = []
    includes = False
    sentence = False
    words = False
    lemmas = False
    anyx = False
    for arg in sys.argv[1:]:
        if arg.startswith('-i'):
            includes = not includes
            continue
        elif arg.startswith('-a'):
            anyx = not anyx
            continue
        elif arg.startswith('-l'):
            lemmas = not includes
            continue
        elif arg.startswith('-s'):
            sentence = not sentence
            continue
        elif arg.startswith('-w'):
            words = not words
        if len(fps) == 2:
            fps.append(open(arg, 'w'))
            break
        fps.append(open(arg, 'r'))
    if not fps:
        raise Exception("no filter specified")
    if len(fps) == 1:
        fps.append(sys.stdin)
    if len(fps) == 2:
        fps.append(sys.stdout)
    fps += [includes, sentence, words, lemmas, anyx]
    nout, nfilt, nline = filter(*fps)
    err.write(f"{nline} lines, {nfilt} filters, {nout} output.\n")
    sys.exit(0)
