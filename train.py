#!/usr/bin/python

import os
import sys
import argparse
import subprocess


def write_vocab(initfname,
                vocabfname="vocab",
                uppercase_unk=False):

    vocabf = open(vocabfname, "w")
    print >>vocabf, "<s>"
    print >>vocabf, "</s>"
    if uppercase_unk:
        print >>vocabf, "<UNK>"
    else:
        print >>vocabf, "<unk>"

    initf = open(initfname)
    categories = set()
    for line in initf:
        line = line.strip()
        tokens = line.split()
        for token in tokens[1:]:
            categories.add(token)
    for cat in sorted_categories:
        print >>vocabf, cat
    vocabf.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training script for a category n-gram model .')
    parser.add_argument('word_init',
                        help='Initial categories for the words', )
    parser.add_argument('train_corpus',
                        help='Corpus for training the model', )
    parser.add_argument('--varikn', action='store_true',
                        help='Use VariKN for training the n-gram component')
    args = parser.parse_args()

    write_vocab(args.word_init, "vocab", args.varikn)

