#!/usr/bin/python

import sys
import locale

# locale -a
locale.setlocale(locale.LC_ALL, 'en_US.UTF8')


vocab = set()
if (len(sys.argv) > 1):
    vocabf = open(sys.argv[1])
    for line in vocabf:
        line = line.strip()
        vocab.add(line)


read_words = set()
for line in sys.stdin:
    line = line.strip()
    words = line.split()
    for word in words:
        if len(vocab) and word not in vocab: continue
        read_words.add(word.capitalize())

for word in read_words:
    print word

