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
               model_id):

    catem_dir = config.get("common", "catem_dir")
    init_exe = os.path.join(catem_dir, "init")
    iter0_id = "%s.iter0" % model_id
    init_cmd = "%s %s %s %s" % (init_exe, word_init, corpus, iter0_id)
    p = subprocess.Popen(init_cmd, shell=True)
    p.wait()

    nc_exe = config.get("common", "srilm")
    nc_cmd = "%s -read %s.ccounts.gz -float-counts -unk -wbdiscount -order 1 -vocab %s -lm %s.arpa.gz"\
                % (nc_exe, iter0_id, vocabfname, iter0_id)
    p = subprocess.Popen(nc_cmd, shell=True)
    p.wait()


def evaluate(model_id,
             corpus):

    catem_dir = config.get("common", "catem_dir")
    catstats_exe = os.path.join(catem_dir, "catstats")

    eval_cmd = "%s %s.arpa.gz %s.cgenprobs.gz %s.cmemprobs.gz %s"\
                    % (catstats_exe, model_id, model_id, model_id, corpus)
    print eval_cmd
    p = subprocess.Popen(eval_cmd, shell=True)
    p.wait()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training script for a category n-gram model .')
    parser.add_argument('trainer_cfg',
                        help='Training configuration file')
    parser.add_argument('word_init',
                        help='Initial categories for the words')
    parser.add_argument('train_corpus',
                        help='Corpus for training the model')
    parser.add_argument('model_id',
                        help='Identifier for the model to be trained')
    args = parser.parse_args()

    config = ConfigParser.ConfigParser()
    config.read(args.trainer_cfg)

    vocab = "vocab"
    vocab_capunk = "vocab.capunk"
    write_vocab(args.word_init, vocab, False)
    write_vocab(args.word_init, vocab_capunk, True)

    init_model(config, args.word_init, vocab, args.train_corpus, args.model_id)
    print >>sys.stderr, "Evaluating training corpus perplexity"
    evaluate("%s.iter0" % args.model_id, args.train_corpus)

    print config.sections()
    print config.items("training")

