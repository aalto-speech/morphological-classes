#!/usr/bin/python

import os
import sys
import argparse
import ConfigParser
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
    for cat in categories:
        print >>vocabf, cat
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
    nc_cmd = "%s -read %s.ccounts.gz -float-counts -unk -wbdiscount -order 1 -vocab %s -lm %s.arpa.gz"\
                % (nc_exe, init_id, vocabfname, init_id)
    subprocess.Popen(nc_cmd, shell=True).wait()


def catstats(prev_iter_id,
             curr_iter_id,
             corpus,
             single_parse=False):

    catem_dir = config.get("common", "catem_dir")
    catstats_exe = os.path.join(catem_dir, "catstats")

    stats_cmd = "%s %s.arpa.gz %s.cgenprobs.gz %s.cmemprobs.gz %s %s"\
                    % (catstats_exe, prev_iter_id, prev_iter_id, prev_iter_id, corpus, curr_iter_id)
    if single_parse: stats_cmd = "%s -p 1" % stats_cmd
    subprocess.Popen(stats_cmd, shell=True).wait()


def ngram_training(iter_id,
                   smoothing,
                   vocab,
                   order):

    srilm_exe = config.get("common", "srilm")
    ngram_cmd = "%s -text %s.catseq.gz -unk -vocab %s -order %i -lm %s.arpa.gz" % (srilm_exe, iter_id, vocab, order, iter_id)
    if smoothing == "wb":
        ngram_cmd = "%s %s" % (ngram_cmd, "-wbdiscount -interpolate -text-has-weights -float-counts")
    elif smoothing == "kn":
        ngram_cmd = "%s %s" % (ngram_cmd, "-kndiscount -interpolate")
    else:
        print >>sys.stderr, "Unknown smoothing option"
        sys.exit(1)
    subprocess.Popen(ngram_cmd, shell=True).wait()


def evaluate(model_id,
             corpus,
             resfname=None):

    catem_dir = config.get("common", "catem_dir")
    catstats_exe = os.path.join(catem_dir, "catstats")

    eval_cmd = "%s %s.arpa.gz %s.cgenprobs.gz %s.cmemprobs.gz %s"\
                    % (catstats_exe, model_id, model_id, model_id, corpus)

    if resfname:
        resf = open(resfname, "a")
        print >>resf, model_id
        resf.flush()
        subprocess.Popen(eval_cmd, stdout=resf, shell=True).communicate()
        print >>resf, ""
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
    args = parser.parse_args()

    config = ConfigParser.ConfigParser()
    config.read(args.trainer_cfg)

    vocab = "vocab"
    vocab_capunk = "vocab.capunk"
    write_vocab(args.word_init, vocab, False)
    write_vocab(args.word_init, vocab_capunk, True)

    prev_iter_id = "%s.iter0" % args.model_id
    init_model(config, args.word_init, vocab, args.train_corpus, prev_iter_id)
    if args.eval_corpus:
        print >>sys.stderr, "Computing evaluation corpus perplexity"
        evaluate(prev_iter_id, args.eval_corpus, "%s.eval.ppl" % args.model_id)

    iterations = config.items("training")
    for iteration in iterations:
        iter_id = "%s.%s" % (args.model_id, iteration[0])
        print >>sys.stderr, ""
        print >>sys.stderr, "Training %s" % iter_id
        smoothing, order, update_catprobs, tag = iteration[1].split(",")
        order = int(order)
        update_catprobs = update_catprobs in ["true", "True", "1"]
        print >>sys.stderr, "Smoothing: %s" % smoothing
        print >>sys.stderr, "Model order: %i" % order
        print >>sys.stderr, "Update category probabilities: %s" % update_catprobs
        print >>sys.stderr, "Tag unks: %s" % tag

        catstats(prev_iter_id, iter_id, args.train_corpus, smoothing=="kn")
        ngram_training(iter_id, smoothing, vocab, order)

        if args.eval_corpus:
            print >>sys.stderr, "Computing evaluation corpus perplexity"
            evaluate(iter_id, args.eval_corpus, "%s.eval.ppl" % args.model_id)

        prev_iter_id = iter_id
