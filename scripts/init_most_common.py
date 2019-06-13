#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import gzip
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates an initialization with most common words in an own class')
    parser.add_argument('VOCABULARY', action="store", help='Vocabulary file')
    parser.add_argument('CORPUS', action="store", help='Corpus file')
    parser.add_argument('NUM_CLASSES', action="store", type=int, help='Number of classes')
    args = parser.parse_args()

    word_order = list()
    print("Reading vocabulary..", file=sys.stderr)
    if args.VOCABULARY.endswith(".gz"):
        vocabf = gzip.open(args.VOCABULARY, "rt")
    else:
        vocabf = open(args.VOCABULARY)
    vocab = dict()
    for line in vocabf:
        line = line.strip()
        if not len(line): continue
        tokens = line.split()
        vocab[tokens[0]] = 0
        word_order.append(tokens[0])
    print("number of words in vocabulary: %i" % len(vocab), file=sys.stderr)

    print("Reading corpus..", file=sys.stderr)
    if args.CORPUS.endswith(".gz"):
        corpusf = gzip.open(args.CORPUS, "rt")
    else:
        corpusf = open(args.CORPUS)
    for line in corpusf:
        line = line.strip()
        if not len(line): continue
        words = line.split()
        for word in words:
            if word in vocab:
                vocab[word] += 1
    print("number of word tokens: %i" % sum(vocab.values()), file=sys.stderr)

    print("Initializing classes..", file=sys.stderr)
    lvocab = list(vocab.items())
    lvocab.sort(key=lambda x: x[1], reverse=True)
    for i in range(0, args.NUM_CLASSES-1):
        vocab[lvocab[i][0]] = i
    for i in range(args.NUM_CLASSES-1, len(lvocab)):
        vocab[lvocab[i][0]] = args.NUM_CLASSES-1

    for word in word_order:
        print("%s\t%i" % (word, vocab[word]))

