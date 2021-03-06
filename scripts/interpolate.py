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


def compute_likelihood(args):
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
            if probs1[i] == unkSymbol and probs2[i] == unkSymbol:
                numUnks += 1
            elif not args.allow_unks and probs1[i] == unkSymbol:
                numUnks += 1
            elif not args.allow_unks and probs2[i] == unkSymbol:
                numUnks += 1
            else:
                numWords += 1
                score1 = float(probs1[i]) if probs1[i] != unkSymbol else smallLP
                score2 = float(probs2[i]) if probs2[i] != unkSymbol else smallLP
                for weights in totalLLs.keys():
                    interpolated_prob = add_log_domain_probs(weights[0] + score1, weights[1] + score2)
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

    return best_weights


def write_probs(args, best_weights):
    probf1 = openFile(args.prob_file_1)
    probf2 = openFile(args.prob_file_2)
    outprobf = openFile(args.prob_file, "w")

    lineIdx = 0
    for probs1 in probf1:
        lineIdx += 1
        probs1 = probs1.strip().split()
        probs2 = probf2.readline().strip().split()

        if len(probs1) != len(probs2):
            raise Exception("Files do not match on line %i" % lineIdx)

        if not len(probs1): continue

        sentence = []
        for i in range(len(probs1)):
            if probs1[i] == unkSymbol and probs2[i] == unkSymbol:
                sentence.append(unkSymbol)
            elif not args.allow_unks and probs1[i] == unkSymbol:
                sentence.append(unkSymbol)
            elif not args.allow_unks and probs2[i] == unkSymbol:
                sentence.append(unkSymbol)
            else:
                score1 = float(probs1[i]) if probs1[i] != unkSymbol else smallLP
                score2 = float(probs2[i]) if probs2[i] != unkSymbol else smallLP
                interpolated_prob = add_log_domain_probs(best_weights[0] + score1, best_weights[1] + score2)
                sentence.append('{:.4e}'.format(interpolated_prob))
        print(" ".join(sentence), file=outprobf)
    outprobf.close()


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
    parser.add_argument('-u', '--allow_unks', action="store_true", default=False,
                        help='Score words included only in the other model, probability assigned by the other model will be 0.0')
    parser.add_argument('-f', '--prob_file', action="store",
                        help='Write interpolated log likelihoods (ln) to a file')
    args = parser.parse_args()

    totalLLs = dict()
    if args.weight:
        if args.weight < 0.0 or args.weight > 1.0:
            raise Exception("Invalid weight %f" % args.weight)
        first_log_iw = math.log(args.weight)
        second_log_iw = math.log(1.0 - args.weight)
        totalLLs[(first_log_iw, second_log_iw)] = 0.0
    elif args.optimize_weights:
        iws = [x * 0.05 for x in range(1,20)]
        for iw in iws:
            first_log_iw = math.log(iw)
            second_log_iw = math.log(1.0 - iw)
            totalLLs[(first_log_iw, second_log_iw)] = 0.0
    else:
        raise Exception("define either --weight or --optimize_weights")

    best_weights = compute_likelihood(args)
    if args.prob_file:
        write_probs(args, best_weights)
