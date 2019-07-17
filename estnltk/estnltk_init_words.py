#!/usr/bin/env python3

import argparse
import gzip
import codecs
import estnltk
from collections import defaultdict
from pprint import pprint

TAG_SEPARATOR = ","

estnltk_options = {
    "disambiguate": False,
    "guess": False,
    "propername": False
}


def read_vocab(fname, max_word_length=0):
    if fname.endswith(".gz"):
        corpusf = gzip.open(fname, "rt", encoding="utf-8")
    else:
        corpusf = codecs.open(fname, "rt", encoding="utf-8")
    vocab = set()
    for line in corpusf:
        line = line.strip()
        if not len(line): continue
        tokens = line.split()
        for token in tokens:
            if max_word_length == 0 or len(token) <= max_word_length:
                vocab.add(token)
    return vocab


def get_analysis_string(analysis):
    if not "partofspeech" in analysis:
        raise Exception("partofspeech field not set: %s" % repr(analysis))
    analysis_tokens = []
    for entry in ["partofspeech", "form", "ending", "clitic"]:
        if entry in analysis and len(analysis[entry]):
            analysis_tokens.append("%s=%s" % (entry, analysis[entry].replace(" ", "_")))
    return TAG_SEPARATOR.join(analysis_tokens)


def analyse_words(vocab):
    analysis_types = set()
    problem_words = set()
    word_analyses = dict()

    for word in vocab:
        e_output = estnltk.Text(word, **estnltk_options)
        # mostly accents within a word, causing estnltk to detect multiple words
        if len(e_output.analysis) > 1:
            problem_words.add(word)
            continue
        elif len(e_output.analysis) == 0:
            raise Exception("no analysis returned from the analyzer for word: %s" % word)

        word_analyses[word] = list()

        for e_analysis in e_output.analysis[0]:
            analysis_string = get_analysis_string(e_analysis)
            analysis_types.add(analysis_string)
            if analysis_string not in word_analyses[word]:
                word_analyses[word].append(analysis_string)

        e_cap_output = estnltk.Text(word.capitalize(), **estnltk_options)
        if len(e_cap_output.analysis) != 1:
            raise Exception("problem in capital word analysis for word %s" % word)
        for e_analysis in e_cap_output.analysis[0]:
            analysis_string = get_analysis_string(e_analysis)
            analysis_types.add(analysis_string)
            if analysis_string not in word_analyses[word]:
                word_analyses[word].append(analysis_string)

    return word_analyses, analysis_types, problem_words


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize word categories with Estnltk analyses")
    parser.add_argument("CORPUS", help="Text corpus")
    parser.add_argument("WORD_INIT", help="Word initializations to be written")
    parser.add_argument("CLASS_DEFS", help="Class definitions to be written")
    parser.add_argument("--max_word_length", help="Maximum length for words to include in the analyses", type=int, default=0)
    args = parser.parse_args()

    vocab = read_vocab(args.CORPUS, args.max_word_length)
    word_analyses, analysis_types, problem_words = analyse_words(vocab)

    print("vocabulary size: %i" % len(vocab))
    print("number of problem words: %i" % len(problem_words))
    word_analysis_counts = list(map(lambda x: len(x), word_analyses.values()))
    num_words_wo_analyses = len(list(filter(lambda x: x == 0, word_analysis_counts)))
    num_words_with_analyses = len(list(filter(lambda x: x > 0, word_analysis_counts)))
    print("number of words without analyses: %i" % num_words_wo_analyses)
    print("number of words with analyses: %i" % len(word_analysis_counts))
    print("mean number of analyses for words with analyses: %f" % (float(sum(word_analysis_counts)) / float(num_words_with_analyses)))
    acc_counts = defaultdict(int)
    for count in word_analysis_counts:
        acc_counts[count] += 1
    for num_analyses, acc_count in acc_counts.items():
        print("  %i analyses: %i" % (num_analyses, acc_count))
    print("number of distinct analyses: %i" % len(analysis_types))

    clsmap = dict(map(reversed, enumerate(sorted(analysis_types))))
    rev_clsmap = dict(enumerate(sorted(analysis_types)))

    wordf = codecs.open(args.WORD_INIT, "w", encoding="utf-8")
    for word, word_analyses in sorted(word_analyses.items()):
        word_class_idxs = map(lambda x: clsmap[x], word_analyses)
        print("%s\t%s" % (word, " ".join(map(str, word_class_idxs))), file=wordf)
    wordf.close()

    classf = open(args.CLASS_DEFS, "w")
    for idx, clss in rev_clsmap.items():
        print("%i\t%s" % (idx, clss), file=classf)
    classf.close()

