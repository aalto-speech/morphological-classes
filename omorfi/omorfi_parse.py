#!/usr/bin/python3

import argparse
import codecs
import locale
import re
import sys


excluded_fields = ["BLACKLIST", "COMPOUND_WORD_ID"]

def parse_analyses(fname, encoding="utf8"):

    analysisFile = codecs.open(fname, encoding=encoding)
    analyses = dict()

    for line in analysisFile:
        line = line.strip()
        if not len(line): continue
        if line.startswith("Strings\tFound"): break

        tokens = line.split("\t")
        if len(tokens) != 2:
            raise Exception("Erroneous Omorfi analysis: %s" % line)

        word = tokens[0]
        if not word in analyses: analyses[word] = set()

        analysis_tokens = re.findall("\[(.*?)\]", tokens[1])
        if analysis_tokens:
            analysis_tokens = analysis_tokens[1:]
            analysis_tokens = list(filter(lambda x: x.split("=")[0] not in excluded_fields, analysis_tokens))
            word_analysis = "_".join(analysis_tokens)
            if len(word_analysis): analyses[word].add(word_analysis)

    return analyses


def analysis_counts(analyses):

    words_w_analyses = set()
    words_wo_analyses = set()
    words_w_uc_analysis = set()
    words_w_lc_analysis = set()

    for word, word_analyses in analyses.items():
        if len(word_analyses):
            words_w_analyses.add(word)

            uc_analysis_found = False
            lc_analysis_found = False
            for word_analysis in word_analyses:
                if "UPOS=PROPN" in word_analysis:
                    uc_analysis_found = True
                else:
                    lc_analysis_found = True
            if uc_analysis_found:
                words_w_uc_analysis.add(word)
            if lc_analysis_found:
                words_w_lc_analysis.add(word)
        else:
            words_wo_analyses.add(word)

    num_uc = len(words_w_uc_analysis)
    num_lc = len(words_w_lc_analysis)
    num_both = len(words_w_uc_analysis & words_w_lc_analysis)

    return len(words_w_analyses), num_uc, num_lc, num_both


def get_num_analyses_per_word(analyses):
    analysis_counts = dict()
    for word, word_analyses in analyses.items():
        if not word_analyses: continue
        word_analysis_count = len(word_analyses)
        if not word_analysis_count in analysis_counts:
            analysis_counts[word_analysis_count] = 0
        analysis_counts[word_analysis_count] += 1

    return analysis_counts


def get_analysis_classes(analyses):
    analysis_types = dict()
    for word, word_analyses in analyses.items():
        if not word_analyses: continue
        for analysis in word_analyses:
            if not analysis in analysis_types:
                analysis_types[analysis] = 0
            analysis_types[analysis] += 1

    return analysis_types


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prints statistics about the Omorfi analyses")
    parser.add_argument("ANALYSES", help="Output from the Omorfi analyzer.")
    args = parser.parse_args()

    locale.setlocale(locale.LC_ALL, 'en_US.utf8')
    analyses = parse_analyses(args.ANALYSES)

    num_w_analyses, num_uc, num_lc, num_both = analysis_counts(analyses)

    print("Number of words: %i" % len(analyses), file=sys.stderr)
    print("Words with analysis: %i" % num_w_analyses, file=sys.stderr)
    print("Words with uppercase analysis: %i" % num_uc, file=sys.stderr)
    print("Words with lowercase analysis: %i" % num_lc, file=sys.stderr)
    print("Words with both upper and lowercase analysis: %i" % num_both, file=sys.stderr)

    print("", file=sys.stderr)
    counts = get_num_analyses_per_word(analyses)
    for analysis_count, word_count in counts.items():
        print("Words with %i analyses: %i" % (analysis_count, word_count), file=sys.stderr)

    print("", file=sys.stderr)
    analysis_types = get_analysis_classes(analyses)
    print("Distinct morphological classes: %i" % len(analysis_types), file=sys.stderr)

#    for analysis_type, count in analysis_types.items():
#        print("%s\t%i" % (analysis_type, count))
