#!/usr/bin/python

import gzip
import sys


vocabfname = sys.argv[1]

if vocabfname.endswith(".gz"):
    vocabf = gzip.GzipFile(vocabfname)
else:
    vocabf = open(vocabfname)

vocab = dict()
for line in vocabf:
    line = line.strip()
    parts = line.split()
    vocab[parts[0]] = parts[1]

for line in sys.stdin:
    line = line.strip()
    words = line.split()
    sent = []
    for word in words:
        if word == "<s>" or word == "</s>":
            continue
        elif word in vocab:
            sent.append(vocab[word])
        else:
            sent.append("1")

    print "<s> %s </s>" % (" ".join(sent))


