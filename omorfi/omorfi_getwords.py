#!/usr/bin/python3
# -*- coding: utf-8 -*-

import argparse
import codecs
import sys
import locale

# locale -a
locale.setlocale(locale.LC_ALL, 'fi_FI.utf8')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prints a vocabulary of capitalized words for Omorfi analysis")
    parser.add_argument("-l", "--max_word_length", help="Maximum length for the words.", type=int, action="store")
    parser.add_argument("-v", "--vocabulary",
                        help="Utf-8 encoded vocabulary file. \
                              Only the words present in the vocabulary file and the corpus are printed")
    args = parser.parse_args()

    vocab = set()
    if args.vocabulary:
        vocabf = codecs.open(args.vocabulary, encoding="utf-8")
        for line in vocabf:
            line = line.strip()
            vocab.add(line.lower())

    read_words = set()
    for line in sys.stdin:
        line = line.strip()
        words = line.split()
        for word in words:
            if args.max_word_length and len(word) > args.max_word_length: continue
            if len(vocab) and word not in vocab: continue
            read_words.add(word.capitalize())

    for word in sorted(read_words):
        print(word)
