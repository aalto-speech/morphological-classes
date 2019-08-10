#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import math
import gzip
import itertools
import argparse


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
                else:
                    try:
                        val = float(probs[s][i])
                    except:
                        raise Exception("Unexpected token: %s" % probs[s][i])
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


def _add_next_weights(combinations, remaining, accuracy=20):
    updated_combinations = []
    for c in combinations:
        new_range = list(map(lambda x: [x], range(1,accuracy-sum(c)-remaining)))
        prod = list(map(lambda x: x[0] + x[1], list(itertools.product([c], new_range))))
        updated_combinations.extend(prod)
    return updated_combinations


def get_weight_combinations(allProbs, accuracy=20):
    numWeights = len(allProbs[0])-1
    combinations = list(map(lambda x: [x], range(1,accuracy-numWeights+1)))
    for i in range(1,numWeights):
        combinations = _add_next_weights(combinations, numWeights-i-1, accuracy)

    combinations = list(map(lambda x: (x + [accuracy-sum(x)]), combinations))
    for combination in combinations:
        if sum(combination) != accuracy:
            raise Exception("Problem combination: %s" % (repr(combination)))
    float_weights = list(map(lambda x: [float(i) / float(accuracy) for i in x], combinations))

    return float_weights


def find_optimal_weights(allProbs):
    print("computing weight combinations..", file=sys.stderr, flush=True)
    weight_combinations = get_weight_combinations(allProbs)
    best_ll = None
    best_weights = None
    print("finding optimal weights..", file=sys.stderr, flush=True)
    for weight_combination in weight_combinations:
        weight_combination = list(map(lambda x: math.log(x), weight_combination))
        curr_ll = compute_likelihood(allProbs, weight_combination)
        if not best_ll or curr_ll > best_ll:
            best_ll = curr_ll
            best_weights = weight_combination

    return best_weights, best_ll


def compute_likelihood(allProbs, weights):
    totalLL = 0.0
    for probs in allProbs:
        currLL = weights[0] + probs[0]
        for i in range(1,len(weights)):
            currLL = add_log_domain_probs(currLL, weights[i] + probs[i])
        totalLL += currLL
    return totalLL


def print_result(numSents, numUnks, numWords, best_weights, totalLL):
    print("Number of sentences: %i" % numSents, file=sys.stderr)
    print("Number of in-vocabulary words exluding sentence ends: %i" % (numWords - numSents), file=sys.stderr)
    print("Number of in-vocabulary words including sentence ends: %i" % numWords, file=sys.stderr)
    print("Number of OOV words: %i" % numUnks, file=sys.stderr)
    print("OOV rate: %f %%" % (100.0 * float(numUnks) / float(numUnks + numWords - numSents)), file=sys.stderr)
    weight_str = ", ".join([("%f" % math.exp(w)) for w in best_weights])
    print("Interpolation weights: %s" % weight_str, file=sys.stderr)
    print("Total log likelihood (ln): %s" % ('{:.5e}'.format(totalLL)), file=sys.stderr)
    print("Total log likelihood (log10): %s" % ('{:.5e}'.format(totalLL / 2.302585092994046)), file=sys.stderr)
    ppl = math.exp(-1.0 / float(numWords) * totalLL)
    print("Perplexity: %.2f" % ppl, file=sys.stderr)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interpolates log likelihoods in two probability files.')
    parser.add_argument('input_prob_files', nargs='+', help='Probability files written by a ppl executable')
    parser.add_argument('-w', '--weights', action="store",
                        help='Interpolation weight for the models in the range, e.g. 0.6,0.4')
    parser.add_argument('-o', '--optimize_weights', action="store_true", default=False,
                        help='Finds the best interpolation weight in steps of 0.05')
    parser.add_argument('-n', '--num_words', action="store", type=int,
                        help='Number of words for computing word-normalized perplexity')
    parser.add_argument('-u', '--allow_unks', action="store_true", default=False,
                        help='Score words included only in the other model, probability assigned by the other model will be 0.0')
    parser.add_argument('-f', '--output_prob_file', action="store",
                        help='Write interpolated log likelihoods (ln) to a file')
    args = parser.parse_args()

    if args.weights:
        weights = list(map(lambda x: float(x), args.weights.split(",")))
        if len(weights) != len(args.input_prob_files):
            raise Exception("Wrong number of interpolation weights: %s" % args.weights)
        if abs(sum(weights) - 1.0) > 0.00001:
            raise Exception("Interpolation weights do not sum to 1, total was: %f" % sum(weights))
        weights = list(map(lambda x: math.log(x), weights))
        print("reading probabilities from files..", file=sys.stderr, flush=True)
        allProbs, numUnks, numSents = read_probs(args.input_prob_files, args.allow_unks)
        print("computing likelihood..", file=sys.stderr, flush=True)
        totalLL = compute_likelihood(allProbs, weights)
        print_result(numSents, numUnks, len(allProbs), weights, totalLL)
    elif args.optimize_weights:
        print("reading probabilities from files..", file=sys.stderr, flush=True)
        allProbs, numUnks, numSents = read_probs(args.input_prob_files, args.allow_unks)
        weights, totalLL = find_optimal_weights(allProbs)
        print_result(numSents, numUnks, len(allProbs), weights, totalLL)
    else:
        raise Exception("Define either --weights or --optimize_weights")
