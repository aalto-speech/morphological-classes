#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import math
import gzip
import argparse


def openFile(fname):
    if fname.endswith(".gz"):
        return gzip.open(fname, "rt")
    else:
        return open(fname, "r")


def add_log_domain_probs(a, b):
    delta = a-b;
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
    parser.add_argument('weight', action="store", type=float,
                        help='Interpolation weight for the first model in the range [0.0,1.0]')
    parser.add_argument('--num_words', action="store", type=int,
                        help='Number of words for computing word-normalized perplexity')
    args = parser.parse_args()

    if args.weight < 0.0 or args.weight > 1.0:
        raise Exception("Invalid weight %f" % args.weight)

    first_log_iw = math.log(args.weight)
    second_log_iw = math.log(1.0-args.weight)

    probf1 = openFile(args.prob_file_1)
    probf2 = openFile(args.prob_file_2)

    lineIdx = 0
    numWords = 0
    numUnks = 0
    totalLL = 0.0
    for probs1 in probf1:
        lineIdx += 1
        probs1 = probs1.strip().split()
        probs2 = probf2.readline().strip().split()

        if len(probs1) != len(probs2):
            raise Exception("Files do not match on line %i" % lineIdx)

        if not len(probs1): continue

        sentLL = 0.0
        for i in range(len(probs1)):
            if probs1[i] == "<unk>":
                numUnks += 1
            elif probs2[i] == "<unk>":
                numUnks += 1
            else:
                numWords += 1
                interpolated_prob = add_log_domain_probs(\
                    first_log_iw + float(probs1[i]),
                    second_log_iw + float(probs2[i]))
                sentLL += interpolated_prob
        totalLL += sentLL

    print("", file=sys.stderr)
    print("Number of sentences: %i" % lineIdx, file=sys.stderr)
    print("Number of in-vocabulary words exluding sentence ends: %i" % (numWords-lineIdx), file=sys.stderr)
    print("Number of in-vocabulary words including sentence ends: %i" % numWords, file=sys.stderr)
    print("Number of OOV words: %i" % numUnks, file=sys.stderr)
    print("Total log likelihood (ln): %s" % ('{:.5e}'.format(totalLL)), file=sys.stderr)
    print("Total log likelihood (log10): %s" % ('{:.5e}'.format(totalLL/2.302585092994046)), file=sys.stderr)

    ppl = math.exp(-1.0/float(numWords) * totalLL)
    print("Perplexity: %.2f" % ppl, file=sys.stderr)

    if args.num_words:
        wnppl = math.exp(-1.0/float(args.num_words) * totalLL)
        print("Word-normalized perplexity: %f" % wnppl, file=sys.stderr)


