#! /usr/bin/env python

import sys
import re
import random
err = sys.stderr

re_inside = re.compile(r"</*[wis][fne][ sn]")
re_id = re.compile(r"id=\"([^\"]*)\"")


def main(args):
    xml_p, key_p = False, False
    while args and args[0][0] == '-':
        arg = args.pop(0).strip('-')
        if arg[0] == 'x':
            xml_p = True
        elif arg[0] == 'k':
            key_p = True
        else:
            err.write(f'Unknown arg ignored: {arg}\n')

    num = args.pop(0)
    if num.find('.') > -1:
        random.seed(42)
        rate = float(num)
    else:
        random.seed(int(num))
        rate = float(arg.pop(0))
    err.write(f"rate {rate}\n")

    for arg in args:
        if arg.endswith('.xml'):
            xml_p = True
        elif arg.endswith('.key'):
            key_p = True
        elif xml_p and arg.endswith('.txt'):
            key_p = True

    if args[0] == '-':
        args[0] = '/dev/stdin'
        if xml_p and key_p:
            err.write("Use of key files with stdin not supported\n")

    if len(args) != (6 if key_p else 4):
        raise Exception("Incorrect paths (3 needed) on command line")

    nfiles = 4 if key_p else 2
    fps = [open(arg, 'r' if ii < (nfiles//2) else 'w')
           for ii, arg in enumerate(args) if ii <= (nfiles+(nfiles//2))]

    cnt = [0, 0, 0, 0]
    bad = 0

    if not xml_p:
        inf = fps.pop(0)
        args.pop(0)
        for line in inf:
            if not line.strip():
                bad += 1
                continue
            which = 0 if random.random() < rate else 1
            fps[which].write(line)
            cnt[which] += 1
        err.write(f"lines {cnt[0]} {cnt[1]}\n")
        return bad

    xi = fps.pop(0)
    if key_p:
        ki = fps.pop(0)
    xo = [fps.pop(0)]
    if key_p:
        ko = [fps.pop(0)]
        xo.append(fps.pop(0))
        ko.append(fps.pop(0))
    else:
        xo.append(fps.pop(0))
    mi, mo = {}, [{}, {}]
    buf = []
    failed = False
    which = 0
    try:

        if key_p:
            for line in ki:
                elts = line.replace('\t', ' ').replace(',', ' ').split()
                mi[elts[0]] = ' '.join(elts[1:])
        err.write(f"{len(mi)} keys loaded\n")

        while True:
            failed = False
            xline = next(xi)

            while re.search(re_inside, xline):
                if not failed:
                    buf.append(xline)

                if xline.startswith('<sentence'):
                    which = 0 if random.random() < rate else 1
                    xline = next(xi)
                    continue

                if key_p and not failed:
                    if xline.startswith('<instance'):
                        m = re.search(re_id, xline)
                        if not m:
                            err.write(f"failed {xline} {len(buf)}\n")
                            sys.exit(0)
                            senses = None
                        else:
                            sid = m.group(1)
                            senses = mi.get(sid)
                        if senses:
                            mo[which][sid] = senses
                        else:
                            err.write(f"missed '{sid}' '{senses}'\n")
                            sys.exit(0)
                            bad += 1
                            failed = True
                            buf = []
                        xline = next(xi)
                        continue

                if xline.startswith('</sentence'):
                    if not failed:
                        while buf:
                            xo[which].write(buf.pop(0))
                        cnt[which] += 1
                    failed = False
                    buf = []

                xline = next(xi)

            while not re.search(re_inside, xline):
                for x in xo:
                    x.write(xline)
                xline = next(xi)

    except StopIteration:

        if not failed:
            while buf:
                xo[which].write(buf.pop(0))
            cnt[which] += 1

    for xf in xo:
        xf.close()

    for which in (0, 1):
        if not key_p:
            break
        for k in mo[which]:
            v = mo[which][k]
            if isinstance(v, list):
                v = ' '.join(v)
            ko[which].write(f"{k} {v}\n")
            cnt[which+2] += 1
        ko[which].close()

    err.write(f"out: {cnt[0]} {cnt[1]} xml {cnt[2]} {cnt[3]} keys\n")
    return bad


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
