#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import gzip
import argparse
from operator import itemgetter
from numpy import arange


def openFile(fname):
    if fname.endswith(".gz"):
        return gzip.open(fname, "rt")
    else:
        return open(fname, "r")


def add_log_domain_probs(a, b):
    delta = a - b
    if delta > 0.0:
        b = a
        delta = -delta
    return b + math.log1p(math.exp(delta))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interpolates log likelihoods in two probability files.')
    parser.add_argument('prob_file_1', action="store",
                        help='The first probability file written by a ppl executable')
    parser.add_argument('prob_file_2', action="store",
                        help='The second probability file written by a ppl executable')
    parser.add_argument('-w', '--weight', action="store", type=float,
                        help='Interpolation weight for the first model in the range [0.0,1.0]')
    parser.add_argument('-o', '--optimize_weights', action="store_true", default=False,
                        help='Finds the best interpolation weight in steps of 0.05')
    parser.add_argument('-n', '--num_words', action="store", type=int,
                        help='Number of words for computing word-normalized perplexity')
    args = parser.parse_args()

    totalLLs = dict()
    if args.weight:
        if args.weight < 0.0 or args.weight > 1.0:
            raise Exception("Invalid weight %f" % args.weight)
        first_log_iw = math.log(args.weight)
        second_log_iw = math.log(1.0 - args.weight)
        totalLLs[(first_log_iw, second_log_iw)] = 0.0
    elif args.optimize_weights:
        iws = arange(0.05, 1.00, 0.05)
        for iw in iws:
            first_log_iw = math.log(iw)
            second_log_iw = math.log(1.0 - iw)
            totalLLs[(first_log_iw, second_log_iw)] = 0.0
    else:
        raise Exception("define either --weight or --optimize_weights")

    probf1 = openFile(args.prob_file_1)
    probf2 = openFile(args.prob_file_2)

    lineIdx = 0
    numWords = 0
    numUnks = 0
    for probs1 in probf1:
        lineIdx += 1
        probs1 = probs1.strip().split()
        probs2 = probf2.readline().strip().split()

        if len(probs1) != len(probs2):
            raise Exception("Files do not match on line %i" % lineIdx)

        if not len(probs1): continue

        for i in range(len(probs1)):
            if probs1[i] == "<unk>":
                numUnks += 1
            elif probs2[i] == "<unk>":
                numUnks += 1
            else:
                numWords += 1
                for weights in totalLLs.keys():
                    interpolated_prob = add_log_domain_probs(
                        weights[0] + float(probs1[i]),
                        weights[1] + float(probs2[i]))
                    totalLLs[weights] += interpolated_prob
    best_weights, totalLL = max(totalLLs.items(), key=itemgetter(1))

    print("", file=sys.stderr)
    print("Number of sentences: %i" % lineIdx, file=sys.stderr)
    print("Number of in-vocabulary words exluding sentence ends: %i" % (numWords - lineIdx), file=sys.stderr)
    print("Number of in-vocabulary words including sentence ends: %i" % numWords, file=sys.stderr)
    print("Number of OOV words: %i" % numUnks, file=sys.stderr)
    print("Interpolation weights: %f, %f" % (math.exp(best_weights[0]), math.exp(best_weights[1])), file=sys.stderr)
    print("OOV rate: %f %%" % (100.0 * float(numUnks) / float(numUnks + numWords - lineIdx)), file=sys.stderr)
    print("Total log likelihood (ln): %s" % ('{:.5e}'.format(totalLL)), file=sys.stderr)
    print("Total log likelihood (log10): %s" % ('{:.5e}'.format(totalLL / 2.302585092994046)), file=sys.stderr)

    ppl = math.exp(-1.0 / float(numWords) * totalLL)
    print("Perplexity: %.2f" % ppl, file=sys.stderr)

    if args.num_words:
        wnppl = math.exp(-1.0 / float(args.num_words) * totalLL)
        print("Word-normalized perplexity: %f" % wnppl, file=sys.stderr)
