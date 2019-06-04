#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import argparse
import configparser
import subprocess


def write_vocab(initfname,
                vocabfname="vocab",
                uppercase_unk=False):
    vocabf = open(vocabfname, "w")
    print("<s>", file=vocabf)
    print("</s>", file=vocabf)
    if uppercase_unk:
        print("<UNK>", file=vocabf)
    else:
        print("<unk>", file=vocabf)

    initf = open(initfname)
    categories = set()
    for line in initf:
        line = line.strip()
        tokens = line.split()
        for token in tokens[1:]:
            categories.add(token)
    for cat in categories:
        print(cat, file=vocabf)
    vocabf.close()


def init_model(config,
               word_init,
               vocabfname,
               corpus,
               init_id):
    catem_dir = config.get("common", "catem_dir")
    init_exe = os.path.join(catem_dir, "init")
    init_cmd = "%s %s %s %s" % (init_exe, word_init, corpus, init_id)
    subprocess.Popen(init_cmd, shell=True).wait()

    nc_exe = config.get("common", "srilm")
    nc_cmd = "%s -read %s.ccounts.gz -float-counts -unk -wbdiscount -order 1 -vocab %s -lm %s.arpa.gz" \
             % (nc_exe, init_id, vocabfname, init_id)
    subprocess.Popen(nc_cmd, shell=True).wait()


def catstats(prev_iter_id,
             curr_iter_id,
             corpus,
             max_order=None,
             update_catprobs=False,
             single_parse=False,
             num_threads=1,
             tag=0,
             max_num_categories=None):
    catem_dir = config.get("common", "catem_dir")
    catstats_exe = os.path.join(catem_dir, "catstats")

    stats_cmd = "%s %s.arpa.gz %s.cgenprobs.gz %s.cmemprobs.gz %s %s -t %i -g %i" \
                % (catstats_exe, prev_iter_id, prev_iter_id, prev_iter_id,
                   corpus, curr_iter_id, num_threads, tag)
    if max_order: stats_cmd = "%s -o %i" % (stats_cmd, max_order)
    if update_catprobs: stats_cmd = "%s -u" % stats_cmd
    if single_parse: stats_cmd = "%s -p 1" % stats_cmd
    if max_num_categories: stats_cmd = "%s -c %i" % (stats_cmd, max_num_categories)
    subprocess.Popen(stats_cmd, shell=True).wait()

    if num_threads > 1:
        catseqfnames = glob.glob("%s.thread*.catseq.gz" % curr_iter_id)
        merge_cmd = "cat %s >%s.catseq.gz" % (" ".join(catseqfnames), curr_iter_id)
        subprocess.Popen(merge_cmd, shell=True).wait()
        for catseqfname in catseqfnames:
            os.remove(catseqfname)


def ngram_training(iter_id,
                   smoothing,
                   vocab,
                   order):
    srilm_exe = config.get("common", "srilm")
    ngram_cmd = "%s -text %s.catseq.gz -unk -vocab %s -order %i -lm %s.arpa.gz" % (
    srilm_exe, iter_id, vocab, order, iter_id)
    if smoothing == "wb":
        ngram_cmd = "%s %s" % (ngram_cmd, "-wbdiscount -interpolate -text-has-weights -float-counts")
    elif smoothing == "kn":
        ngram_cmd = "%s %s" % (ngram_cmd, "-kndiscount -interpolate")
    else:
        print("Unknown smoothing option", file=sys.stderr)
        sys.exit(1)
    subprocess.Popen(ngram_cmd, shell=True).wait()


def evaluate(model_id,
             corpus,
             max_order=None,
             resfname=None,
             num_threads=1):
    catem_dir = config.get("common", "catem_dir")
    catstats_exe = os.path.join(catem_dir, "catstats")

    eval_cmd = "%s %s.arpa.gz %s.cgenprobs.gz %s.cmemprobs.gz %s -t %i" \
               % (catstats_exe, model_id, model_id, model_id, corpus, num_threads)
    if max_order: eval_cmd = "%s -o %i" % (eval_cmd, max_order)

    if resfname:
        resf = open(resfname, "a")
        print(model_id, file=resf)
        resf.flush()
        subprocess.Popen(eval_cmd, stdout=resf, shell=True).communicate()
        print("", file=resf)
        resf.close()
    else:
        subprocess.Popen(eval_cmd, shell=True).wait()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training script for a category n-gram model.')
    parser.add_argument('trainer_cfg',
                        help='Training configuration file')
    parser.add_argument('word_init',
                        help='Initial categories for the words')
    parser.add_argument('train_corpus',
                        help='Corpus for training the model')
    parser.add_argument('model_id',
                        help='Identifier for the model to be trained')
    parser.add_argument('--eval_corpus',
                        help='Corpus for evaluating the model')
    parser.add_argument('--num_threads', type=int, default=1,
                        help='Number of threads for collecting the category statistics')
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.trainer_cfg)

    vocab = "%s.vocab" % args.model_id
    vocab_capunk = "%s.vocab.capunk" % args.model_id
    write_vocab(args.word_init, vocab, False)
    write_vocab(args.word_init, vocab_capunk, True)

    max_order = 0
    iterations = config.items("training")
    for iteration in iterations:
        smoothing, order, update_catprobs, tag, max_num_categories = iteration[1].split(",")
        max_order = max(max_order, int(order))

    pplresfname = "%s.eval.ppl" % args.model_id
    if os.path.exists(pplresfname): os.remove(pplresfname)
    prev_iter_id = "%s.iter0" % args.model_id
    init_model(config, args.word_init, vocab, args.train_corpus, prev_iter_id)
    if args.eval_corpus:
        print("Computing evaluation corpus perplexity", file=sys.stderr)
        evaluate(prev_iter_id, args.eval_corpus, max_order,
                 pplresfname, args.num_threads)

    for iteration in iterations:
        iter_id = "%s.%s" % (args.model_id, iteration[0])
        print("", file=sys.stderr)
        print("Training %s" % iter_id, file=sys.stderr)
        smoothing, order, update_catprobs, tag, max_num_categories = iteration[1].split(",")
        order = int(order)
        tag = int(tag)
        max_num_categories = int(max_num_categories)
        update_catprobs = update_catprobs in ["true", "True", "1"]
        print("Smoothing: %s" % smoothing, file=sys.stderr)
        print("Model order: %i" % order, file=sys.stderr)
        print("Update category probabilities: %s" % update_catprobs, file=sys.stderr)
        print("Tag unks: %i" % tag, file=sys.stderr)
        print("Maximum number of categories: %i" % max_num_categories, file=sys.stderr)

        catstats(prev_iter_id, iter_id, args.train_corpus,
                 max_order, update_catprobs, smoothing == "kn",
                 args.num_threads, tag, max_num_categories)
        ngram_training(iter_id, smoothing, vocab, order)

        if args.eval_corpus:
            print("Computing evaluation corpus perplexity", file=sys.stderr)
            evaluate(iter_id, args.eval_corpus, max_order,
                     pplresfname, args.num_threads)

        prev_iter_id = iter_id
