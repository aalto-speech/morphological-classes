#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import gzip
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Creates a corpus for training a word-based model using the class model vocabulary.')
    parser.add_argument('class_memberships', action="store",
                        help='Class membership file written by exchange (.gz supported)')
    parser.add_argument('--cap_unk', action="store_true", default=False,
                        help='Unk symbol should be written in capitals i.e. <UNK>')
    parser.add_argument('--lc_unk', action="store_true", default=False,
                        help='Unk symbol should be written in lowercase i.e. <unk>')
    args = parser.parse_args()

    unk = None
    if args.cap_unk:
        unk = "<UNK>"
    elif args.lc_unk:
        unk = "<unk>"
    else:
        raise Exception("Define either cap_unk or lc_unk option")

    if args.class_memberships.endswith(".gz"):
        vocabf = gzip.open(args.class_memberships, "rt")
    else:
        vocabf = open(args.class_memberships)

    vocab = dict()
    for line in vocabf:
        line = line.strip()
        tokens = line.split()
        if len(tokens) != 3:
            print("Problem in line: %s" % line, file=sys.stderr)
            continue
        vocab[tokens[0]] = int(tokens[1])

    for line in sys.stdin:
        line = line.strip()
        words = line.split()
        sent = []
        for word in words:
            if word == "<s>": continue
            if word == "</s>": continue
            if word in vocab:
                sent.append(word)
            else:
                sent.append(unk)

        print("<s> %s </s>" % (" ".join(sent)))

