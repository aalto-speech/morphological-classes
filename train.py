#!/usr/bin/python

import os
import sys
import subprocess


def write_vocab(initfname,
                vocabfname="vocab",
                varikn=False):

    vocabf = open(vocabfname, "w")
    print >>vocabf, "<s>"
    print >>vocabf, "</s>"
    if varikn:
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
    for cat in categories:
        print >>vocabf, cat
    vocabf.close()


if __name__ == "__main__":

    write_vocab("words.init", "vocab", False)
