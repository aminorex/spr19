import os
import re
import sys
import math
import torch
import datetime
import numpy as np
from copy import copy
from tqdm import trange
from torch import tensor, argmax, no_grad, nn
from torch.nn import CrossEntropyLoss
from sklearn.metrics import f1_score
from random import randint, seed
from collections import Counter
# from torch.optim import Adam
# from torch.utils.data import TensorDataset
# from torch.utils.data import DataLoader
# from torch.utils.data import RandomSampler
# from torch.utils.data import SequentialSampler
# from sklearn.model_selection import train_test_split

PROJ_ROOT = os.getenv('PROJ_ROOT')
if PROJ_ROOT is None:
    PROJ_ROOT = '/home/tony/csci/project/'
elif PROJ_ROOT[-1] != '/':
    PROJ_ROOT += '/'
BERT_DIR = PROJ_ROOT + 'pytorch-pretrained-BERT'
ENV_PKG_DIR = os.getenv('HOME') + \
    '.conda/envs/ptbert/lib/python3.7/site-packages/'

sys.path.insert(0, BERT_DIR)
sys.path.insert(0, ENV_PKG_DIR)
# import tensorflow as tf
# from keras.preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertAdam
from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertForMaskedLM
from pytorch_pretrained_bert import BertLayerNorm
from pytorch_pretrained_bert import BertPreTrainedModel
from pytorch_pretrained_bert import WEIGHTS_NAME, CONFIG_NAME


err = sys.stderr
out = sys.stdout
inp = sys.stdin


def ctime():
    return datetime.datetime.now().ctime()


RANDOM_SEED = os.getenv('RANDOM_SEED')
if RANDOM_SEED is None:
    RANDOM_SEED = 42

PAD_TEXTS = None  # loaded in  global code, below

MODEL_INFO = {
    'base': {
        'name': 'bert-base-uncased',
        'dir': 'models/uncased_L-12_H-768_A-12/',
        },
    'large': {
        'name': 'bert-large-uncased',
        'dir': 'models/uncased_L-24_H-1024_A-16/',
    }
}
MODEL = 'base'

MODEL_DIR = PROJ_ROOT + 'data/model/'

FILTERED_DIR = PROJ_ROOT + 'data/filtered/'
FILTER_SENSES = FILTERED_DIR + 'filter.senses'

GOOGLE_WSD_DIR = PROJ_ROOT + 'data/word_sense_disambigation_corpora/'

# GOOGLE_WSD_DAT = GOOGLE_WSD_DIR + 'all.xml'
# GOOGLE_WSD_KEY = GOOGLE_WSD_DIR + 'combined_map.txt'
GOOGLE_WSD_DAT = FILTERED_DIR + 'google.xml'
GOOGLE_WSD_KEY = FILTERED_DIR + 'google.key'
GOOGLE_CORPUS = (GOOGLE_WSD_DAT, GOOGLE_WSD_KEY, 37424)

OMSTI_TRAIN_DIR = PROJ_ROOT + 'data/WSD_Training_Corpora/SemCor+OMSTI/'

# OMSTI_TRAIN_DAT = OMSTI_TRAIN_DIR + 'semcor+omsti.data.xml'
# OMSTI_TRAIN_KEY = OMSTI_TRAIN_DIR + 'semcor+omsti.gold.key.txt'
OMSTI_TRAIN_DAT = FILTERED_DIR + 'omsti.xml'
OMSTI_TRAIN_KEY = FILTERED_DIR + 'omsti.key'
OMSTI_CORPUS = (OMSTI_TRAIN_DAT, OMSTI_TRAIN_KEY, 664654)

SEMCOR_TRAIN_DIR = PROJ_ROOT + 'data/WSD_Training_Corpora/SemCor/'

# SEMCOR_TRAIN_DAT = SEMCOR_TRAIN_DIR + 'semcor.data.xml'
# SEMCOR_TRAIN_KEY = SEMCOR_TRAIN_DIR + 'semcor.gold.key.txt'
SEMCOR_TRAIN_DAT = FILTERED_DIR + 'semcor.xml'
SEMCOR_TRAIN_KEY = FILTERED_DIR + 'semcor.key'
SEMCOR_CORPUS = (SEMCOR_TRAIN_DAT, SEMCOR_TRAIN_KEY, 16079)

UNIFIED_TEST_DIR = PROJ_ROOT + 'data/WSD_Unified_Evaluation_Datasets/ALL/'

# UNIFIED_TEST_DAT = UNIFIED_TEST_DIR + 'ALL.data.xml'
# UNIFIED_TEST_KEY = UNIFIED_TEST_DIR + 'ALL.gold.key.txt'
UNIFIED_TEST_DAT = FILTERED_DIR + 'unified.xml'
UNIFIED_TEST_KEY = FILTERED_DIR + 'unified.key'
UNIFIED_CORPUS = (UNIFIED_TEST_DAT, UNIFIED_TEST_KEY, 1129)

FILT_TRAIN_DAT = FILTERED_DIR + 'train.xml'
FILT_TRAIN_KEY = FILTERED_DIR + 'train.key'
ALL_CORPUS = (FILT_TRAIN_DAT, FILT_TRAIN_KEY, 710270)

FILT_LARGE_DAT = FILTERED_DIR + 'large.xml'
FILT_LARGE_KEY = FILTERED_DIR + 'large.key'
LARGE_CORPUS = (FILT_LARGE_DAT, FILT_LARGE_KEY, 568315)

FILT_SMALL_DAT = FILTERED_DIR + 'small.xml'
FILT_SMALL_KEY = FILTERED_DIR + 'small.key'
SMALL_CORPUS = (FILT_SMALL_DAT, FILT_SMALL_KEY, 141955)

FILT_TEST_DAT = FILTERED_DIR + 'test.xml'
FILT_TEST_KEY = FILTERED_DIR + 'test.key'
TEST_CORPUS = (FILT_TEST_DAT, FILT_TEST_KEY, 2138)

FILT_DEV_DAT = FILTERED_DIR + 'dev.xml'
FILT_DEV_KEY = FILTERED_DIR + 'dev.key'
DEV_CORPUS = (FILT_DEV_DAT, FILT_DEV_KEY, 5749)

CORPORA = {
    'google': GOOGLE_CORPUS,
    'omsti': OMSTI_CORPUS,
    'semcor': SEMCOR_CORPUS,
    'unified': UNIFIED_CORPUS,
    'test': TEST_CORPUS,
    'dev': DEV_CORPUS,
    'small': SMALL_CORPUS,
    'large': LARGE_CORPUS,
    'all': ALL_CORPUS
}

TOKEN_MAX = 85
TOKEN_MIN = 5
MAX_SEQUENCE = TOKEN_MAX + 2*TOKEN_MIN + 2

DEFAULT_EPOCHS = 3
BATCH_SIZE = 32
MAX_SHIFTS = 32

NON_DECAY_UNITS = ['bias', 'layernorm.bias', 'layernorm.weight']
BERT_CACHE_DIR = '/'.join((os.getenv('HOME'), '.pytorch_pretrained_bert'))

PAD_FILE = PROJ_ROOT + 'data/pad.txt'

VOCAB_DIR = PROJ_ROOT + MODEL_INFO['large']['dir']
VOCAB_FILE = VOCAB_DIR + 'vocab.txt'
with open(VOCAB_FILE, 'r') as vocabf:
    VOCAB_ELTS = frozenset(elt.strip() for elt in vocabf)

CUDA_DEVS = os.getenv('CUDA_VISIBLE_DEVICES')

WN_NOUN, WN_VERB, WN_ADJ, WN_ADV, WN_EXT, WN_NONE = 1, 2, 3, 4, 5, 0
WN_NTAGS = 6
# POS tags used in WSD xml and their possible WN category numbers
POS_TAGS = {
    '.':	[WN_NONE],
    'PUNCT':	[WN_NONE],
    'ADJ':	[WN_ADJ, WN_EXT],
    'ADP':	[WN_NONE],
    'ADV':	[WN_ADV],
    'CONJ':	[WN_ADV, WN_NONE],
    'DET':	[WN_EXT, WN_ADV, WN_ADJ, WN_NONE],
    'NOUN':	[WN_NOUN],
    'NUM':	[WN_NOUN, WN_EXT],
    'PRON':	[WN_NOUN],
    'PRT':	[WN_NONE],
    'VERB':	[WN_VERB],
    'X':	[WN_ADV],
    'abbreviation': [WN_NOUN]
}

# non-regex html entity fixups
FIX_REPLS = {
    're&apos;t':	'reat',
    'out&apos;n':	'out in',
    '&$39;':		'&apos;',
    '&#225;':		'a',
    '&#231;':		'c',
    '&#233;':		'e',
    '&#243;':		'o',
    '&quot;':		"''",
    '&amp;':		'&',
    'po&apos;k':	'pork',
    's&apos;posin':	'supposing'
}

# regex html entity fixups
FIX_EXPRS = {
    re.compile(r'( *)&apos;d'):		r'\1would',
    re.compile(r'( *)&apos;em'):	r'\1them',
    re.compile(r'( *)&apos;ll'):	r'\1will',
    re.compile(r'( *)&apos;m'):		r'\1am',
    re.compile(r'( *)&apos;re'):	r'\1are',
    re.compile(r'( *)&apos;ve'):	r'\1have',
    re.compile(r'( *)&apos;&apos;.'):	r"\1'' .",
    re.compile(r'( *)([Bb])ird&apos;s-eye'):	r"\1\2ird's-eye",
    re.compile(r'( *)([Bb])ull&apos;s-eye'):	r"\1\2ull's-eye",
    re.compile(r'( *)([Cc])&apos;mon'):	r'\1\2ome on',
    re.compile(r'( *)n&apos;t'):	r"\1not",
    re.compile(r'&apos;'):		r"'"
}


# extraction regexes - supWSD corpus
re_id = re.compile(r"id=\"([^\"]*)\"")
re_text = re.compile(r"\>([^\<]*)")
re_sentid = re.compile(r"([semeval\.|senseval\.]*[^\"]*)[\" ]")
re_inst = re.compile(r" id=\"([^\"]*)\".*lemma=\"([^\"]*)\".*" +
                     r"pos=\"([^\"]*)\">([^<]*)<")
re_lemma = re.compile(r"lemma=\"([^\"]*)\" pos=\"([^\"]*)\"")
# extraction regexes - Google corpus
re_sense = re.compile(r"text=\"([^\"]*)\".*lemma=\"([^\"]*)\".*" +
                      r"pos=\"([^\"]*)\".*sense=\"([^\"]*)\".*" +
                      r"break_level=\"([^\"]*)\"/>")
re_word = re.compile(r"text=\"([^\"]*)\".*break_level=\"([^\"]*)\"/>")
re_name = re.compile(r"name=\"([^\"]*)\"")

# non-english filter regex for WSD XML
JUNK_TEXT = frozenset([
    "&#178;",
    "&amp;&amp|dabhumaksanigalu&apos;ahai",
    "ksu&apos;u&apos;peli&apos;afo",
    "mai&apos;teipa",
    "&apos;Orso"
])


class Tokenizer(BertTokenizer):
    _INST = None

    @classmethod
    def instance(cls):
        if not cls._INST:
            cls._INST = Tokenizer()
        return cls._INST

    def __init__(self):
        BertTokenizer.__init__(self, VOCAB_FILE)
        return

    def merge_pieces(self, toks, tcopy=False):
        if tcopy:
            toks = copy(toks)
        pos = 1
        source_pos = [[0]]
        for ii, tok in enumerate(toks):
            if not ii:
                continue
            if tok[:2] == '##':
                toks[ii-1] += tok[2:]
                source_pos[-1].append(ii)
            else:
                source_pos.append([ii])
            pos += 1
        return toks, source_pos


TIZE = Tokenizer.instance()

#
# load pad sentences selected from unified test data
#
PAD_TEXTS = dict((i, []) for i in range(TOKEN_MIN, TOKEN_MAX + 1))

with open(PAD_FILE, 'r') as padf:
    # move sense marks to first token of a word piece split
    for line in padf:
        line = line.strip()
        rtoks, trails = [], []
        for ii, rtok in enumerate(line.split()):
            parts = rtok.split('\\')
            rtoks.append(parts.pop(0))
            trails.append(parts)
        atoks = TIZE.tokenize(' '.join(rtoks))
        if len(atoks) > TOKEN_MAX:
            continue
        parts = []
        for ii, atok in enumerate(atoks):
            if atok[:2] != '##':
                parts = trails.pop(0) if trails else []
            parts.insert(0, atok)
            atoks[ii] = '\\'.join(parts)
            del parts[0]
        PAD_TEXTS[len(atoks)].append(atoks)

# construct missing pad cases
for k in PAD_TEXTS:
    if not PAD_TEXTS[k]:
        a, b = k//2, k - k//2
        if a == b and k > 10:
            c, d = k//2-1, k//2
        else:
            c, d = b, a
        for b in ((a, b), (c, d)):
            try:
                nu, nl = map(lambda x: len(PAD_TEXTS[x]), b)
            except Exception:
                print(repr(b), repr(k))
            for ul in PAD_TEXTS[b[0]]:
                for ll in PAD_TEXTS[b[1]]:
                    su = PAD_TEXTS[b[0]][randint(0, nu-1)]
                    sl = PAD_TEXTS[b[1]][randint(0, nl-1)]
                    PAD_TEXTS[k].append(sl + su)
                    if len(PAD_TEXTS[k]) > 9:
                        break
    if not PAD_TEXTS[k]:
        err.write(f"Failed to pad at {k}\n")
        sys.exit(0)


def random_pad(ntok):
    """
    select a pad of the desired length at random
    """
    if ntok not in PAD_TEXTS:
        # print(f"missing pad {ntok}")
        return None
    pads = PAD_TEXTS[ntok]
    return pads[randint(0, len(pads)-1)]


def unpack_wnclass(senses, tag=None):
    """
    numeric WN pos classes for each sense
    when tag is provided, that will be force to
    be the first in the list
    """
    # start with class of pos tag if tagged
    cls = POS_TAGS.get(tag, WN_NONE)
    # get class marks from senses
    clss = []
    for sense in senses:
        try:
            parts = sense.split('%')
            if not parts:
                err.write(f"Empty sense error.\n")
                continue
            if len(parts) != 2:
                err.write(f"Failed to %-split sense {sense}\n")
            else:
                parts.pop(0)
            for part in parts:
                tagx = part.split(':')
                clss.append(int(tagx[0]))
        except Exception:
            raise Exception(f"Broken at {sense}")
    # omit WN_NONE and cls
    clss = Counter(c for c in clss if c and c != cls)
    # descending order, strip counts
    clss = [p[0] for p in sorted(clss.items(), key=lambda p: (-p[1], p[0]))]
    # insure cls is first, if not WN_NONE
    if cls or not clss:
        clss.insert(0, cls)
    # must return [WN_NONE] iff no other clss found
    return clss


class TargetCoder(object):
    def __init__(self, sense_cat):
        self.sense_codes = {}   # map sense text to sense number
        self.code_senses = {}   # map sense number to sense text
        with open(FILTER_SENSES, 'r') as sensef:
            for ii, sense in enumerate(sensef):
                sense.strip()
                self.sense_codes[sense] = ii
                self.code_senses[ii] = sense
        self.nsense = len(self.sense_codes)
        self.shift = None
        self.pad_cache = {}
        return

    # TODO
    def decode(self, vocab_code, sense_codes):
        """
        map numeric encoding to string encoding
        """
        return (self.code_vocab.get(vocab_code, ''),
                [self.code_sense.get(x, None) for x in sense_codes])

    def unpack_instance(self, inst):
        """
        a WSD instance from corpus iterator structure
        is unpacked into numerical equivalent
        """
        at, senses, _, tag, tok = inst
        snuml = [self.sense_codes.get(s, None) for s in senses]
        snuml = [s for s in snuml if s is not None]
        wncl = unpack_wnclass(senses, tag)
        return at, wncl, snuml, tok

    def encode_item(self, rtoks, insts):
        """
        a text and the WSD instances within it are unpacked into
        an equivalent numerical encoded structure
        """
        snumlv, wnclv, ptokv = [], [], []
        ipos, wncl, snuml = -1, [WN_NONE], []
        ptokv = TIZE.tokenize(' '.join(rtoks))
        rpos = 0

        for ii, ptok in enumerate(ptokv):
            contin = ptok[:2] == '##'
            if not contin:
                if rpos > ipos and insts:
                    ipos, wncl, snuml, _ = self.unpack_instance(insts.pop(0))
                if ipos > rpos:
                    wncl, snuml = [WN_NONE], []
                rpos += 1
            wnclv.append(wncl)
            snumlv.append(snuml)

        vtokv = TIZE.convert_tokens_to_ids(ptokv)
        return vtokv, snumlv, wnclv, ptokv

    def encode_pad(self, atoks):
        """
        similar to encode_item, but uses generated pad format input
        to be used for pad tokens.  atoks are word piece tokens with
        sense annotations on the first piece
        """
        snumlv, wnclv, ptokv = [], [WN_NONE], []
        ptoks = [atok.split('\\')[0] for atok in atoks]
        snuml, wncl = [], []
        for ii, parts in enumerate(atok.split('\\') for atok in atoks):
            ptokv.append(parts[0])
            if ptokv[-1][2:] != '##':
                try:
                    if len(parts) > 2:
                        senses = parts[-1].split(',')
                        snuml = [self.sense_codes.get(x) for x in senses]
                        snuml = [x for x in snuml if x is not None]
                        wncl = unpack_wnclass(senses)
                    else:
                        snuml = []
                        wncl = [WN_NONE]
                except Exception:
                    err.write(f"Error in parts {parts} senses {senses}\n")
                    err.write(f"tokens: {atoks}\n")
                    sys.exit(0)
            snumlv.append(snuml)
            wnclv.append(wncl)
        vtokv = TIZE.convert_tokens_to_ids(ptoks)
        return vtokv, snumlv, wnclv, ptokv

    def make_pads(self, nattend, max_width, max_shifts):
        """
        nattend includes [CLS] [SEP] tokens
        """
        pads = []
        possible_shifts = max_width - nattend - TOKEN_MIN * 2
        stride = 1
        if possible_shifts > max_shifts:
            stride = (possible_shifts + max_shifts - 1) // max_shifts
        for prelen in range(TOKEN_MIN, possible_shifts+TOKEN_MIN+1, stride):
            postlen = max_width - nattend - prelen
            pre = random_pad(prelen)
            post = random_pad(postlen)
            if pre and post:
                pads.append((pre, post))
        return pads

    def shifts(self, toks, insts, width, max_shifts):
        """
        generate training shifts
        could also use this as an ensemble of classifiers in test
        """
        self.args = [toks, insts, width, max_shifts]
        ntokv, snumlv, wnclv, ptokv = self.encode_item(toks, insts)
        pads = self.make_pads(len(ntokv), width, max_shifts)
        self.shift = [pads, ntokv, snumlv, wnclv, ptokv]
        return self

    def length(self):
        if not self.shift:
            return 0
        return len(self.shift[0])

    def reset(self):
        self.shift = None

    def __iter__(self):
        return self

    def __next__(self):
        pads, ntokv, snumlv, wnclv, ptokv = self.shift
        if not pads:
            self.reset()
            raise StopIteration
        pair = pads.pop(0)
        encodes = []
        for pad in pair:
            text = ' '.join(pad)
            if text not in self.pad_cache:
                self.pad_cache[text] = self.encode_pad(pad)
            encodes.append(self.pad_cache[text])
        ntokv0, snumlv0, wnclv0, ptokv0 = encodes.pop(0)
        ntokv2, snumlv2, wnclv2, ptokv2 = encodes.pop(0)
        ntokv3 = ntokv0 + [-1] + ntokv + [-1] + ntokv2
        snumlv3 = snumlv0 + [[]] + snumlv + [[]] + snumlv2
        wnclv3 = wnclv0 + [[WN_NONE]] + wnclv + [[WN_NONE]] + wnclv2
        # ptokv3 = ptokv0 + ['[CLS]'] + ptokv + ['[SEP]'] + ptokv2
        l0, l1, l2 = len(ptokv0), len(ptokv) + 2, len(ptokv2)
        l1 += l0
        mask = [0 if ii < l0 or ii >= l1 else 1 for ii in range(l2+l1)]
        return ntokv3, snumlv3, wnclv3, mask


class XmlCorpusIter(object):
    """Iterator returns successive acceptable instances
    of an XML WSD corpus"""

    def __init_fp(self):
        dataf = open(self.data_path, 'r')
        if not self.est_n:
            self.est_n = sum(1 for line in dataf
                             if line.startswith('<sentence') or
                             line.find('SENTENCE_BREAK') > -1)
        dataf.seek(0)
        for line in dataf:
            if line.startswith('<SimpleWsdDoc'):
                google_fmt = True
                break
            if line.startswith('<sentence'):
                google_fmt = False
                break
        if not (line.startswith('<SimpleWsdDoc') or
                line.startswith('<sentence')):
            raise Exception(f"No starting tag in {self.data_path} {line}.")
        if google_fmt:
            self.doc_no = 0
            self.sent_no = 0
            self.doc_name = re.search(re_name, line).group(1)
            self.sent_id = 'google-{}.d{}.s{}'.\
                format(self.doc_name, str(self.doc_no).zfill(3),
                       str(self.sent_no).zfill(4))
        else:
            self.sent_id = re.search(re_id, line).group(1)
        return dataf

    def __init__(self, data_path, map_path, *rest, **kwargs):
        self.est_n = kwargs.get('est_n', 0)
        self.data_path = data_path
        self.flags = kwargs

        self.verbose = 1 if 'v' in rest or 'verbose' in rest else 0
        for k in 'v', 'verbose', 'verbosity':
            if k in kwargs:
                self.verbose += int(kwargs[k])

        if self.flags.get('copy'):
            filt = self.flags['copy']
            err.write(f"filter {filt}\n")
            self.copyf = open(self.flags['copy'], 'w')
            self.buffer = []
        else:
            self.copyf = False

        # google_fmt = map_path.find('map.txt') > -1
        self.data = self.__init_fp()
        self.reset(close=False)

        self.sense_map = {}
        with open(map_path, 'r') as mapf:
            for line in mapf:
                if line.find('\t') > -1:
                    line = line.strip().split('\t')
                    self.sense_map[line[0]] = line[1].split(',')
                elif line.find(' ') > -1:
                    line = line.strip().split(' ')
                    self.sense_map[line[0]] = line[1:]
                else:
                    break
        tmp = []
        with open(FILTER_SENSES, 'r') as sensef:
            for sense in sensef:
                sense = sense.strip()
                if sense:
                    tmp.append(sense)
        self.sense_cat = frozenset(tmp)
        self.senses_found = set()

        return

    def length(self):
        return max(self.est_n, 0)

    def reset(self, close=True):
        if self.data and close:
            self.data.close()
        self.senses_found = set()
        self.yielded = 0
        self.dropped = 0
        self.nlookup_fail = 0
        self.nsense_fail = 0
        self.nvocab_fail = 0
        self.njunk_fail = 0
        self.nmax_fail = 0
        self.nmin_fail = 0
        self.doc_no = 0
        self.insts = {}
        self.failed_senses = Counter()
        return

    def __iter__(self):
        self.sent_no = 0
        self.doc_no = 0
        self.doc_name = ''
        if self.data is None:
            self.data = self.__init_fp()
        return self

    def pos_ids(self, tag):
        if tag not in POS_TAGS:
            return [0]
        return POS_TAGS[tag]

    def entity_fix(self, string):
        for k in FIX_REPLS:
            string = string.replace(k, FIX_REPLS[k])
        for k in FIX_EXPRS:
            string = re.sub(k, FIX_EXPRS[k], string)
        return string

    def valid_p(self, tokens, instances, lookup_fails):
        failed = False
        if lookup_fails:
            if self.verbose:
                err.write(f"lookup failure\n")
            self.nlookup_fail += 1
            return False
        sentence = ' '.join(tokens)
        if 'junk' not in self.flags or \
           self.flags['junk']:
            for junk in JUNK_TEXT:
                if sentence.find(junk) > -1:
                    if self.verbose:
                        err.write(f"junk in {sentence}\n")
                    self.njunk_fail += 1
                    return False
        ptoks = TIZE.tokenize(sentence)
        ntoks = len(ptoks)
        if self.flags.get('min') and ntoks < self.flags['min']:
            if self.verbose:
                err.write(f"{ntoks} toks < min\n")
            self.nmin_fail += 1
            return False
        if self.flags.get('max') and ntoks > self.flags['max']:
            if self.verbose:
                err.write(f"{ntoks} toks > max\n")
            self.nmax_fail += 1
            return False
        if 'vocab' not in self.flags or self.flags['vocab']:
            for x in ptoks:
                if x in VOCAB_ELTS:
                    continue
                if not all(not c.isalpha()
                           for c in x):
                    continue
                if x[0] != '[' or x[-1] != ']':
                    if self.verbose:
                        err.write(f"missing vocab {x}\n")
                    self.nvocab_fail += 1
                    return False
        # require all senses to be in the catalog
        if 'sense' not in self.flags or self.flags['sense']:
            for inst in instances:
                for sense in inst[1]:
                    if sense not in self.sense_cat:
                        if self.verbose:
                            err.write(f"missing sense {sense}\n")
                        self.nsense_fail += 1
                        self.failed_senses[sense] += 1
                        failed = True
            if failed:
                return False
        return True

    def __next__(self):
        insts = []
        lemma, tag, brk, name, sense = ['']*5
        toks = []
        bad_lookup = 0
        check = False
        self.inst_no = 0
        self.sent_no = 0
        for line in self.data:
            val = None
            line = line.strip()
            # start with google xml format
            if line.startswith('<SimpleWsdDoc '):
                self.inst_no = 0
                self.doc_name = re.search(re_name, line).group(1)
                self.sent_id = "google-{}.d{}.s{}".\
                    format(self.doc_name, str(self.doc_no).zfill(3),
                           str(self.sent_no).zfill(4))
            elif line.startswith('</SimpleWsdDoc>'):
                self.doc_name = ''
                self.doc_no += 1
                self.sent_no = 0
            elif line.startswith('<word '):
                if self.copyf:
                    if not self.buffer:
                        self.buffer.append('<sentence id="{0}">\n'.
                                           format(self.sent_id))
                m = re.search(re_sense, line)
                if m:
                    txt, lemma, pos, code, brk = m.groups()
                    txt = self.entity_fix(txt)
                    sense = self.sense_map.get(code)
                    check = self.sent_id == 'google-br-r05.d001.s0017'
                    if self.copyf:
                        inst_id = self.sent_id+'.t'+str(self.inst_no).zfill(4)
                        # write this to new gold file later
                        if sense:
                            self.sense_map[inst_id] = sense
                        self.buffer.append(f'<instance id="{inst_id}" ' +
                                           f'lemma="{lemma}" pos="{pos}"' +
                                           f'>{txt}</instance>\n')
                    if sense:
                        insts.append((len(toks), sense, lemma, pos, txt))
                        self.inst_no += 1
                    else:
                        if self.verbose or check:
                            err.write(f"sense missing for {code} in {txt} " +
                                      f"at {self.sent_id}\n")
                        bad_lookup += 1
                if not m:
                    m = re.search(re_word, line)
                    txt, brk = m.groups()
                    txt = self.entity_fix(txt)
                    if self.copyf:
                        self.buffer.append(f'<wf lemma="{txt}" pos="X"' +
                                           f'>{txt}</wf>\n')
                toks.append(txt.lower())
                if brk == 'SENTENCE_BREAK':
                    self.est_n += -1
                    if self.valid_p(toks, insts, bad_lookup):
                        val = (toks, insts, copy(self.sent_id))
                        self.sent_no += 1
                        self.yielded += 1
                        for i in insts:
                            for s in i[1]:
                                self.senses_found.add(s)
                        if self.copyf and self.buffer:
                            self.buffer.append("</sentence>\n")
                            for lout in self.buffer:
                                self.copyf.write(lout)
                    else:
                        if self.verbose:
                            err.write(f"not valid {toks} {insts}\n")
                        self.dropped += 1
                    self.buffer = []
                    self.inst_no = 0
                    insts = []
                    toks = []
                    bad_lookup = 0
                    self.sent_id = "google-{}.d{}.s{}".\
                        format(self.doc_name, str(self.doc_no).zfill(3),
                               str(self.sent_no).zfill(4))
                    if val:
                        return val
            # otherwise, supWSD xml format
            elif line.startswith('<sentence '):
                self.est_n += -1
                if self.valid_p(toks, insts, bad_lookup):
                    val = (toks, insts, self.sent_id)
                    self.sent_no += 1
                    self.yielded += 1
                    for i in insts:
                        for s in i[1]:
                            self.senses_found.add(s)
                    if self.copyf and self.buffer:
                        for lout in self.buffer:
                            self.copyf.write(lout)
                else:
                    if self.verbose:
                        err.write(f"not valid {toks} {insts}\n")
                    self.dropped += 1
                self.buffer = []
                insts = []
                toks = []
                bad_lookup = 0
                m = re.search(re_id, line)
                self.sent_id = m.group(1)
                if self.copyf:
                    self.buffer.append('<sentence id="{0}">\n'.
                                       format(self.sent_id))
                if val:
                    return val
            elif line.startswith('<wf '):
                txt = re.search(re_text, line).group(1)
                toks.append(self.entity_fix(txt).lower())
                if self.copyf:
                    lemma, pos = re.search(re_lemma, line).groups()
                    self.buffer.append(f'<wf lemma="{lemma}" ' +
                                       f'pos="{pos}">{txt}</wf>\n')
            elif line.startswith('<instance '):
                inst_id, lemma, pos, txt = re.search(re_inst, line).groups()
                txt = self.entity_fix(txt)
                sense = self.sense_map.get(inst_id)
                if sense:
                    # char pos, inst id, lemma, pos tag
                    insts.append((len(toks), sense, lemma, pos, txt))
                    toks.append(txt.lower())
                else:
                    if self.verbose:
                        err.write(f"sense missing for {inst_id} in {txt}\n")
                    bad_lookup += 1
                if self.copyf:
                    self.buffer.append(f'<instance id="{inst_id}" ' +
                                       f'lemma="{lemma}" ' +
                                       f'pos="{pos}">{txt}</instance>\n')
            elif line.startswith('</sentence'):
                self.inst_no = 0
                if self.copyf:
                    self.buffer.append("</sentence>\n")
        if self.copyf:
            self.copyf.close()
            with open(self.flags['copy'].split('.')[0]+'.key', 'w') as createf:
                for k in self.sense_map:
                    try:
                        createf.write("{} {}\n".
                                      format(k, ' '.join(self.sense_map[k])))
                    except Exception:
                        err.write("failed at key {} senses {}.\n".
                                  format(k, repr(self.sense_map[k])))
                        sys.exit(0)
        raise StopIteration


class CorpusChainIter(object):

    def __init__(self, *arglist):
        self.paths = arglist
        self.fp = None

    def __iter__(self):
        return self

    def __next__(self):
        if not self.fp:
            if not self.paths:
                raise StopIteration
            self.fp = open(self.paths.pop(0), 'r')


class Predictor(object):

    def __init__(self, _model, _text=''):
        self.model = _model
        self.text = ''
        self.toks = []
        if _text:
            self.add(_text)

    def add(self, _text):
        if not self.toks:
            self.toks = ['[CLS]']
        elif self.toks[-1] == '[SEP]':
            self.toks.pop(-1)
        self.toks += TIZE.tokenize(_text) + ['[SEP]']
        self.text = ' '.join(self.toks)
        self.parse()

    def reset(self):
        self.text = ''
        self.toks = []

    def parse(self, text=''):
        if text:
            self.reset()
            self.add(text)
        if not self.text or not self.toks:
            self.reset()
            raise Exception('No text to parse')
        self.itoks = TIZE.convert_tokens_to_ids(self.toks)
        self.ttoks = tensor([self.itoks])
        # seps = [i for i in range(len(self.toks)) if self.toks[i] == '[SEP]']
        self.masks = [i for i in range(len(self.toks))
                      if self.toks[i] == '[MASK]']
        self.tsegs = tensor([[0] * len(self.toks)])
        return len(self.toks)

    def predict(self, text=''):
        if not text:
            if not self.text:
                raise Exception('No text to predict')
        else:
            self.parse(text)
        ctoks = self.ttoks.to('cuda')
        csegs = self.tsegs.to('cuda')
        with no_grad():
            predicts = self.model(ctoks, csegs)
        return predicts

    def top_predictions(self, predicts, nway=0):
        if nway < 2:
            pidxs = [argmax(predicts[0, i]).item() for i in self.masks]
            ptoks = TIZE.convert_ids_to_tokens(pidxs)
            toks = self.toks.copy()
            for ii, ti in zip(pidxs, ptoks):
                toks[ii] = ti
            return [toks]
        else:
            return []  # TODO implement


class MaskedGenerator(object):
    def __init__(self):
        self.model, self.predictor, self.predictions = None, None, None

    def _load(self):
        if not self.model:
            self.model = \
                BertForMaskedLM.from_pretrained(MODEL_INFO[MODEL]['name'])
            self.model.eval()
            self.model.to('cuda')
            self.predictor = Predictor(self.model)

    def predict(self, _text):
        self._load()
        self.predictions = self.predictor.predict(_text)
        return self.predictions

    def alternates(self, predictions=[], max_n=1):
        self._load()
        if not predictions:
            predictions = self.predictions
        alts = dict()
        for ii in self.predictor.masks:
            logits = list(predictions[0, ii])
            pairs = sorted([(logit, jj)
                            for jj, logit in enumerate(logits)])[:max_n]
            # total = sum(paiairpr[0] for pair in pairs)
            # itoks = [x[1] for x in pairs]
            alts[ii] = TIZE.convert_ids_to_tokens(it[1]
                                                  for it in pairs)
        return alts

    def generate(self, _text=None, max_n=1):
        self._load()
        if _text is not None:
            self.predict(_text)
        # TODO: implement max_n > 1
        alts = self.alternates(max_n)
        # ntok = len(alts)
        toks = self.predictor.toks.copy()
        for k, v in alts:
            toks[k] = v[0]
        return ' '.join(toks)


def one_hot(positions, length):
        """
        a numerical format target list is expanded
        to one-hot coding feature vector
        """
        vec = [0] * length
        for pos in positions:
            vec[pos] = 1
        return vec


def gelu(x):
    """
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertForWSD(BertPreTrainedModel):

    def __init__(self, config, num_labels=None, num_classes=0, deep=0):
        super(BertForWSD, self).__init__(config)
        self.deep = deep
        self.num_labels = num_labels
        self.num_classes = num_classes
        self.bert = BertModel(config)
        extended_size = config.hidden_size + num_classes
        if deep:
            # insert cls here if deep
            self.dense = nn.Linear(extended_size, extended_size)
            self.transform = ACT2FN[config.hidden_act]
            self.layernorm = BertLayerNorm(extended_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # insert cls here if not deep
        self.classifer = nn.Linear(extended_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, tokvv, maskvv, labels=None, classes=None):
        seq, pooled = self.bert(tokvv, maskvv,
                                output_all_encoded_layers=False)
        if self.num_classes and classes:
            ext = torch.cat((seq, classes), 3)
        else:
            ext = seq
        if self.deep:
            ext = self.dense(ext)
            ext = self.transform(ext)
            ext = self.layernorm(ext)
        ext = self.dropout(ext)
        logits = self.classifier(ext)

        lm_loss = None
        if labels:
            loss_fn = CrossEntropyLoss(ignore_index=-1)
            lm_loss = loss_fn(logits.view(-1, self.num_labels),
                              labels.view(-1))

        return logits, lm_loss


def init_seeds(n_gpu=0):
    """
    initialize all operational random number generators
    """
    seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if n_gpu > 1:
        torch.cuda.manual_seed_all(RANDOM_SEED)


def init_cuda():
    """
    get cuda environment
    """
    device = 'cpu'
    n_gpu = 0
    if CUDA_DEVS is None or CUDA_DEVS != '':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            n_gpu = torch.cuda.device_count()
    return device, n_gpu


def param_groups(model):
    """
    build optimizer parameter groups for a given BertModel
    """
    param_opt = list(model.named_parameters())
    return [
        {'params': [p for n, p in param_opt
                    if not any(nd in n for nd in NON_DECAY_UNITS)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_opt
                    if any(nd in n for nd in NON_DECAY_UNITS)],
         'weight_decay': 0.0}
    ]


def train(coder, model, items, est_n, epochs,
          scalar=False, grad_steps=1):
    """
    perform training on items from iterator "items"
    est_n is an estimate of the number of items produced by the iterator
    """
    device, n_gpu = init_cuda()
    init_seeds(n_gpu)
    model.train()
    model.to(device)

    learn_rate = 5e-5
    warm_ups = 0.1
    est_batches = est_n * MAX_SHIFTS // BATCH_SIZE
    optimizer = BertAdam(param_groups(model),
                         lr=learn_rate, warmup=warm_ups,
                         t_total=est_batches)
    g_steps, t_steps, t_loss, n_examples = 0, 0, 0, 0

    try:
        for ii in trange(items.length(), desc='cases'):
            rtoks, insts, sent_id = next(items)
            batch = [[], [], [], []]
            shifts = coder.shifts(rtoks, insts, MAX_SEQUENCE, MAX_SHIFTS)
            nlabels = coder.nsense
            for jj in range(shifts.length()):
                try:
                    data = list(next(shifts))
                    data[1] = [one_hot(snuml, nlabels) for snuml in data[1]]
                    for ii, column in enumerate(data):
                        batch[ii].append(column)
                except StopIteration:
                    shifts.reset()
                rem = items.length() + shifts.length()
                if batch and (len(batch) == BATCH_SIZE or not rem):
                    tokvv = torch.tensor(batch[0]).to(device)
                    hotvvv = torch.tensor(batch[1]).to(device)
                    clsvv = torch.tensor(batch[2]).to(device)
                    maskvv = torch.tensor(batch[3]).to(device)
                    for i in range(epochs):
                        """
                        # classifier style loss
                        logits = model(tokvv, clsvv, maskvv, hotvvv)
                        step_loss, nexamples, nsteps = 0, 0, 0
                        """
                        _, lm_loss = model(tokvv, maskvv,
                                           labels=hotvvv, classes=clsvv)
                        t_steps += 1
                        t_loss += lm_loss.item()
                        n_examples += tokvv.size(0)
                        lm_loss.backward()

                        if (i+1) % grad_steps == 0:
                            optimizer.step()
                            optimizer.zero_grad()
                            g_steps += 1
    except StopIteration:
        pass

    return


def rollup_batch(gold, logits):
    minl = logits.min()
    maxl = logits.max()
    logits = (logits - minl) / (maxl - minl)
    batchsz, maxseq, nsense = gold.size()
    g2 = gold.view(-1, nsense).numpy()
    l2 = logits.view(-1, nsense).numpy()
    shp = np.shape(l2)
    ones = np.ones(shp)
    zeros = np.zeros(shp)
    active = g2.sum(1) > 0
    minp = np.where(np.logical_and(active, g2 > 0), l2, ones).min()
    maxn = np.where(np.logical_and(active, g2 < 1), l2, zeros).max()
    threshold = (minp + maxn) * 0.5
    b2 = (l2 > threshold) * 1
    f1_micro = f1_score(g2, b2, average='micro')
    f1_macro = f1_score(g2, b2, average='macro')
    cases = [(g2[ii, :], l2[ii, :]) for ii in range(shp[0])
             if active[ii]]
    return [cases, minp, maxn, f1_micro, f1_macro]


def rollup_total(batches):
    b = batches.pop(0)
    cases, minp, maxn, f1_micro, f1_macro = b[0]

    thresholds = [(minp + maxn) * 0.5]
    case = cases[1]
    g2 = case[0]
    l2 = case[1]
    for ii in range(len(cases)-1):
        np.concatenate(g2, case[ii+1], out=g2)
        np.concatenate(l2, case[ii+1], out=l2)
    f1a = [f1_micro]
    f1b = [f1_macro]
    sizes = [len(cases)]
    for b in batches:
        sizes.append(len(b[0]))
        for c in b[0]:
            np.concatenate(g2, c, out=g2)
            np.concatenate(l2, c, out=l2)
            cases.append(c)
        threshold = (b[1] + b[2]) * 0.5
        thresholds.append(threshold)
        minp = min(minp, b[1])
        maxn = max(maxn, b[2])
        f1a.append(b[3])
        f1b.append(b[4])

    threshold = (minp + maxn) * 0.5
    out.write(f"threshold {threshold}\n")
    tnp = np.array(thresholds)
    tm, td = tnp.mean(), tnp.std()
    out.write(f"batch threshold mean, std: {tm} {td}\n")
    del thresholds
    del tnp

    b2 = (l2 > threshold) * 1
    f1_micro = f1_score(g2, b2, average='micro')
    f1_macro = f1_score(g2, b2, average='macro')
    out.write(f"f1_micro total {f1_micro} f1_macro total {f1_macro}\n")

    f1a = np.array(f1a)
    f1b = np.array(f1b)
    m1a = f1a.mean()
    m1b = f1b.mean()
    d1a = f1a.std()
    d1b = f1b.std()
    out.write(f"batch mean, std: f1_micro {m1a} {d1a}, f1_macro {m1b} {d1b}\n")

    snp = np.array(sizes)
    sm, sd = snp.mean(), snp.std()
    out.write(f"batch size mean, std: {sm} {sd}\n")
    del snp
    del sizes

    tp = (b2 * g2).sum()
    fp = (b2 * (1-g2)).sum()
    tn = ((1-b2) * (1-g2)).sum()
    fn = ((1-b2) * g2).sum()
    out.write(f"t,f pos: {tp}, {fp} - t,f neg: {tn} {fn}\n")
    return


class Result(object):
    def __init__(self):
        self.batch = []

    def batch(self, gold, logits, loss):
        self.batch.append(rollup_batch(gold, logits))
        return

    def summary(self):
        rollup_total(self.batch)
        return


def test(coder, model, items, max_shifts):
    device, n_gpu = init_cuda()
    init_seeds(n_gpu)
    model.eval()
    model.to(device)
    seqlen = MAX_SEQUENCE
    max_shifts = min(max_shifts, BATCH_SIZE)
    result = Result()

    for ii in trange(items.length(), desc="cases"):
        rtoks, insts, sent_id = next(items)
        batch = [[], [], [], []]
        shifts = coder.shifts(rtoks, insts, seqlen, max_shifts)
        nlabels = coder.nsense

        for jj in range(shifts.length()):
            try:
                # breakpoint()
                data = list(next(shifts))
                data[1] = [one_hot(snuml, nlabels) for snuml in data[1]]
                data[2] = [x if isinstance(x, list) else [x] for x in data[2]]
                data[2] = [one_hot(x, WN_NTAGS) for x in data[2]]
                for ii, column in enumerate(data):
                    batch[ii].append(column)
            except StopIteration:
                shifts.reset()
        tokvv = torch.tensor(batch[0]).to(device)
        clsvv = torch.tensor(batch[2]).to(device)
        maskvv = torch.tensor(batch[3]).to(device)
        with no_grad():
            logits, loss = model(tokvv, maskvv, classes=clsvv)
        result.batch(torch.tensor(batch[1]), logits, loss)

    result.summary()
    return


def main(corpus, task, flags):
    # UNIFIED_TEST_DAT, UNIFIED_TEST_KEY,
    dat_path, key_path, est_n = corpus
    imin, imax = TOKEN_MIN, TOKEN_MAX
    if flags.get('test') or flags.get('train'):
        imax -= TOKEN_MIN
    items = XmlCorpusIter(dat_path, key_path, est_n, v=flags.get('v'),
                          copy=flags.get('filter'), min=imin, max=imax)

    if task == 'scan':
        outf = flags.get('outf')
        # just run the xml processor
        nsents = 0
        uniqs = set()
        total_insts = 0
        total_senses = 0
        for item in items:
            toks, insts, sent_id = item
            nsents += 1
            if outf:
                outf.write(f"{sent_id}\n")
                outf.write(f"{toks}\n")
            ninst = len(insts)
            total_insts += ninst
            for ii, inst in enumerate(insts):
                at, sense, lemma, tag, tok = inst
                if outf:
                    outf.write("{}/{}: {} lem {} pos {} @{}\n".
                               format(ii, ninst, tok, lemma, tag, at))
                nsense = len(sense)
                total_senses += nsense
                for jj, s in enumerate(sense):
                    uniqs.add(s)
                    if outf:
                        out.write(f"\t{jj}/{nsense}: {s}\n")
        nfound = len(items.senses_found)
        nlabs = len(items.sense_cat)
        valid, invalid = items.yielded, items.dropped
        err.write("{0} valid, {1} invalid lines with {2} of {3} senses\n".
                  format(valid, invalid, nfound, nlabs))
        err.write("{0} unique {1} senses {2} instances {3} sentences\n".
                  format(len(uniqs), total_senses, total_insts, nsents))
        err.write(("fails {0} lookup {1} sense {2} vocab {3} " +
                   "junk {4} max {5} min\n").
                  format(items.nlookup_fail, items.nsense_fail,
                         items.nvocab_fail, items.njunk_fail,
                         items.nmax_fail, items.nmin_fail))
        if outf:
            for sense in items.failed_senses:
                num = items.failed_senses[sense]
                outf.write(f"{num} {sense}\n")
            for sense in items.senses_found:
                outf.write(f"found {sense}\n")

    if task in ('train', 'test'):
        coder = TargetCoder(items.sense_cat)

    wide = flags.get('wide', 0)
    deep = flags.get('deep', 0)

    if task == 'train':
        select = MODEL_INFO[MODEL]['name']
        nclass = WN_NTAGS if wide else 0
        model = BertForWSD.from_pretrained(select,
                                           num_classes=nclass,
                                           num_labels=coder.nsense,
                                           deep=deep,
                                           cache_dir=BERT_CACHE_DIR)
        train(coder, model, items, est_n,
              flags.get('epochs', DEFAULT_EPOCHS),
              scalar=flags.get('scalar', False))

        sub_dir = f"{deep}_{wide}"
        model_dir = flags.get('save')
        if sub_dir not in os.listdir(model_dir):
            model_dir = model_dir + '/' + sub_dir
            model_dir.replace('//', '/')
            os.mkdir(model_dir)
            model_dir += '/'
        else:
            model_dir += '/' + sub_dir + '/'
            model_dir.replace('//', '/')
        weights_path = model_dir + WEIGHTS_NAME
        config_path = model_dir + CONFIG_NAME

        torch.save(model.state_dict(), weights_path)
        with open(config_path, 'w') as fp:
            fp.write(model.config.to_json_string())

    if task == 'test':
        model_dir = flags.get('load') + '/' + f"{deep}_{wide}" + '/'
        model_dir.replace('//', '/')
        nclass = WN_NTAGS if wide else 0
        model = BertForWSD.from_pretrained(model_dir,
                                           num_classes=nclass,
                                           num_labels=coder.nsense,
                                           deep=flags.get('deep', 0),
                                           cache_dir=BERT_CACHE_DIR)
        test(coder, model, items, 1)

    items.reset()
    return


if __name__ == '__main__':

    flags = {
        'v': 0,
        'outf': None,
        'train': False,
        'test': False,
        'scan': False,
        'save': MODEL_DIR,
        'load': MODEL_DIR,
        'filter': None,
        'corpus': None
    }

    for arg in sys.argv[1:]:
        if arg == 'v':
            flags['v'] += 1
        elif arg == 'training':
            flags['train'] = True
        elif arg == 'testing':
            flags['test'] = True
        elif arg == 'scan':
            flags['test'] = True
        elif arg.startswith('save='):
            flags['save'] = arg[5:]
        elif arg.startswith('load='):
            flags['load'] = arg[5:]
        elif arg.startswith('filt='):
            flags['filter'] = arg[5:]
        elif arg.startswith('epochs='):
            flags['epochs'] = int(arg[7:])
        elif arg == '-':
            flags['outf'] = sys.stdout
        elif arg[:4] == 'deep':
            flags['deep'] = 1
        elif arg[:4] == 'wide':
            flags['wide'] = 1
        elif arg == 'list-corpora':
            for tag in CORPORA:
                print(tag)
            sys.exit(0)
        elif arg in CORPORA:
            flags['corpus'] = arg
        elif all(x.isdigit() for x in arg):
            RANDOM_SEED = int(arg)
        elif all(x.isalpha() for x in arg):
            flags[arg] = True

    corpus = flags.get('corpus')
    if flags.get('scan'):
        if not corpus:
            raise Exception(f"Missing corpus on command-line")
    for task in ('scan', 'train', 'test'):
        if not flags.get(task):
            continue
        err.write(f"task {task} corpus {corpus}\n")
        main(CORPORA[corpus if corpus else task], task, flags)
        del flags['filter']

    sys.exit(0)
