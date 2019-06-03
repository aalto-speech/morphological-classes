#!/usr/bin/python3

import argparse
import codecs
import locale
import re
import sys
from collections import defaultdict

# excluded_fields = ["BLACKLIST", "COMPOUND_WORD_ID"]
excluded_fields = ["BLACKLIST"]

MAX_NUM_ANALYSES_PER_WORD = 15


def read_analyses_file(analysisFname, analyses, encoding="utf8"):
    analysisFile = codecs.open(analysisFname, encoding=encoding)
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


def merge_large_coverage_analyses(analyses, large_coverage_analyses):
    extra_words = 0
    extra_analyses = 0
    for word, word_analyses in analyses.items():
        large_coverage_word_analyses = large_coverage_analyses[word]
        if (len(large_coverage_word_analyses)) <= MAX_NUM_ANALYSES_PER_WORD:
            if not len(word_analyses) and len(large_coverage_word_analyses):
                extra_words += 1
                analyses[word] = large_coverage_word_analyses
            elif len(large_coverage_word_analyses) > len(word_analyses):
                extra_analyses += 1
    return extra_words, extra_analyses


def get_num_analyses_per_word(analyses):
    analysis_counts = defaultdict(int)
    word_analysis_counts = [len(x) for x in analyses.values() if x]
    for count in word_analysis_counts:
        analysis_counts[count] += 1
    return analysis_counts


def get_analysis_classes(analyses):
    analysis_types = defaultdict(int)
    for word_analyses in analyses.values():
        if not word_analyses: continue
        for word_analysis in word_analyses:
            analysis_types[word_analysis] += 1
    return analysis_types


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Prints statistics about the Omorfi analyses")
    parser.add_argument("ANALYSES", help="Output from the Omorfi analyzer without the -X switch.")
    parser.add_argument("LARGE_COVERAGE_ANALYSES", help="Output from the Omorfi analyzer with the -X switch.")
    args = parser.parse_args()

    locale.setlocale(locale.LC_ALL, 'en_US.utf8')

    analyses = dict()
    print("Reading default vocabulary Omorfi analyses..", file=sys.stderr)
    read_analyses_file(args.ANALYSES, analyses)

    large_coverage_analyses = dict()
    print("Reading extended vocabulary Omorfi analyses..", file=sys.stderr)
    read_analyses_file(args.LARGE_COVERAGE_ANALYSES, large_coverage_analyses)

    if len(analyses) != len(large_coverage_analyses):
        raise Exception("number of analyses don't match, (%i and %i)" % (len(analyses), len(large_coverage_analyses)))

    print("Merging analysis outputs..", file=sys.stderr)
    extra_words, extra_analyses = merge_large_coverage_analyses(analyses, large_coverage_analyses)
    print("Number of new words with analyses in the extended vocabulary: %i" % extra_words, file=sys.stderr)
    print("Number of words with more analyses in the extended vocabulary: %i" % extra_analyses, file=sys.stderr)
    print("")

    num_w_analyses = len([x for x in analyses.values() if len(x)])
    case_info = dict()
    for word, word_analyses in analyses.items():
        case_info[word] = list(map(lambda x: "UPOS=PROPN" in x, list(word_analyses)))
    num_uc = len([x for x in case_info.values() if x.count(True) > 0])
    num_lc = len([x for x in case_info.values() if x.count(False) > 0])
    num_both = len([x for x in case_info.values() if x.count(False) > 0 and x.count(True) > 0])

    print("Number of words: %i" % len(analyses), file=sys.stderr)
    print("Words with analysis: %i" % num_w_analyses, file=sys.stderr)
    print("Words with uppercase analysis: %i" % num_uc, file=sys.stderr)
    print("Words with lowercase analysis: %i" % num_lc, file=sys.stderr)
    print("Words with both upper and lowercase analysis: %i" % num_both, file=sys.stderr)

    print("", file=sys.stderr)
    counts = get_num_analyses_per_word(analyses)
    for analysis_count in sorted(counts):
        print("Words with %i analyses: %i" % (analysis_count, counts[analysis_count]), file=sys.stderr)

    print("", file=sys.stderr)
    analysis_types = get_analysis_classes(analyses)
    print("Distinct morphological classes: %i" % len(analysis_types), file=sys.stderr)

#    for analysis_type, count in analysis_types.items():
#        print("%s\t%i" % (analysis_type, count))
