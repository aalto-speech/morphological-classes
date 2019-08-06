#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import gzip
import argparse
from operator import itemgetter


unkSymbol = "<unk>"
smallLP = -1000.0


def openFile(fname, mode="r"):
    if fname.endswith(".gz"):
        return gzip.open(fname, mode+"t")
    else:
        return open(fname, mode)


def add_log_domain_probs(a, b):
    delta = a - b
    if delta > 0.0:
        b = a
        delta = -delta
    return b + math.log1p(math.exp(delta))


def read_probs(prob_fnames, allow_unks=False):
    probfs = []
    for prob_fname in prob_fnames:
        probfs.append(openFile(prob_fname))

    lineIdx = 0
    numUnks = 0
    numSents = 0
    allProbs = []
    for prob_line in probfs[0]:
        lineIdx += 1
        probs = [prob_line.strip().split()]
        for probf in probfs[1:]:
            probs.append(probf.readline().strip().split())

        for i,_probs in enumerate(probs):
            if len(_probs) != len(probs[0]):
                raise Exception("Files do not match on line %i" % lineIdx)

        if len(probs[0]) == 0: continue

        for i in range(len(probs[0])):
            unkFound = False
            probFound = False
            for s in range(len(probs)):
                if probs[s][i] == unkSymbol:
                    unkFound = True
                elif probs[s][i].isnumeric():
                    probFound = True

            if not allow_unks and unkFound:
                numUnks += 1
            elif allow_unks and not probFound:
                numUnks += 1
            else:
                curr_probs = []
                for s in range(len(probs)):
                    curr_probs.append(float(probs[s][i]) if probs[s][i] != unkSymbol else smallLP)
                allProbs.append(tuple(curr_probs))
        numSents += 1

    return allProbs, numUnks, numSents


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interpolates log likelihoods in two probability files.')
    parser.add_argument('input_prob_file', nargs='+', help='Probability files written by a ppl executable')
    parser.add_argument('-w', '--weight', action="store", type=float,
                        help='Interpolation weight for the first model in the range [0.0,1.0]')
    parser.add_argument('-o', '--optimize_weights', action="store_true", default=False,
                        help='Finds the best interpolation weight in steps of 0.05')
    parser.add_argument('-n', '--num_words', action="store", type=int,
                        help='Number of words for computing word-normalized perplexity')
    parser.add_argument('-u', '--allow_unks', action="store_true", default=False,
                        help='Score words included only in the other model, probability assigned by the other model will be 0.0')
    parser.add_argument('-f', '--output_prob_file', action="store",
                        help='Write interpolated log likelihoods (ln) to a file')
    args = parser.parse_args()

    print(args.input_prob_file)
    allProbs, numUnks, numSents = read_probs(args.input_prob_file)
    print(len(allProbs))
    print(numUnks)
    print(numSents)
