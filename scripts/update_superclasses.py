#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import gzip
import locale
import argparse
import locale


def openFile(fname):
    if fname.endswith(".gz"):
        return gzip.open(fname, "rt")
    else:
        return open(fname)


def readClasses(fname):
    print("reading class membership file %s" % fname, file=sys.stderr)
    vocab = set()
    wordClasses = dict()
    numIgnoredLines = 0
    classFile = openFile(fname)
    for line in classFile:
        line = line.strip()
        if not len(line): continue
        tokens = line.split()
        if len(tokens) != 3:
            numIgnoredLines += 1
            continue
        classIdx = int(tokens[1])
        wordClasses[tokens[0]] = classIdx
        vocab.add(tokens[0])
    if numIgnoredLines:
        print("ignored %i lines in the class membership file" % numIgnoredLines, file=sys.stderr)
    return vocab, wordClasses


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Updates superclass definitions using a new class membership file')
    parser.add_argument('SUPERCLASSES', action="store", help='Original superclass definitions')
    parser.add_argument('ORIGINAL_CMEMPROBS', action="store", help='Original classes')
    parser.add_argument('NEW_CMEMPROBS', action="store", help='Updated classes')
    args = parser.parse_args()

    locale.setlocale(locale.LC_ALL, 'fi_FI.utf8')

    superClassFile = openFile(args.SUPERCLASSES)
    superClasses = []
    superClassLookup = dict()
    numClasses = 0
    for line in superClassFile:
        line = line.strip()
        if not len(line): continue
        classIdxs = list(map(lambda x: int(x), line.split(",")))
#        classIdxs = line.split(",")
        superClassIdx = len(superClasses)
        superClasses.append(classIdxs)
        for classIdx in classIdxs:
            superClassLookup[classIdx] = superClassIdx
        numClasses += len(classIdxs)
    print("read %i superclasses" % len(superClasses), file=sys.stderr)
    print("read %i classes" % numClasses, file=sys.stderr)

    originalVocab, originalWordClasses = readClasses(args.ORIGINAL_CMEMPROBS)
    newVocab, newWordClasses = readClasses(args.NEW_CMEMPROBS)

    if originalVocab != newVocab:
        raise Exception("The vocabularies do not match.")

    superClassWords = list()
    newSuperClasses = list()
    for i in range(len(superClasses)):
        superClassWords.append(set())
        newSuperClasses.append(set())
    for word, wordClass in originalWordClasses.items():
        superClassWords[superClassLookup[wordClass]].add(word)

    assertClasses = set()
    for i in range(len(superClassWords)):
        for word in superClassWords[i]:
            newSuperClasses[i].add(newWordClasses[word])
            assertClasses.add(newWordClasses[word])

    numSuperclassClasses = 0
    for i in range(len(newSuperClasses)):
        numSuperclassClasses += len(newSuperClasses[i])
    if len(assertClasses) != numSuperclassClasses:
        raise Exception("The updated class memberships do not match with the original superclasses")

    for i in range(len(newSuperClasses)):
        classes = map(lambda x: str(x), list(newSuperClasses[i]))
        print(",".join(classes))
