#!/usr/bin/python

import re
import sys
import locale
import codecs


def parse_analyses(fname, encoding="utf8"):

    analf = codecs.open(fname, encoding=encoding)
    analyses = dict()

    for line in analf:
        line = line.strip()
        if not len(line): continue
        if line.startswith("Strings\tFound"): break

        tokens = line.split("\t")
        wrd = tokens[0]
        analysis_tokens = tokens[1].split()

        tmpi = 0
        for i in range(len(analysis_tokens)):
            if r"#" in analysis_tokens[i]: tmpi = i
        if tmpi > 0: analysis_tokens = [analysis_tokens[0]] + analysis_tokens[tmpi+1:]

        if len(analysis_tokens) > 2:
            toks = "_".join(set(analysis_tokens[2:]))
            wrd_analysis = "%s_%s" % (analysis_tokens[1], toks)
        elif len(analysis_tokens) > 1:
            wrd_analysis = "%s" % analysis_tokens[1]
        else:
            wrd_analysis = ""

        lowercase = analysis_tokens[0].lower() == analysis_tokens[0]
        if lowercase: wrd = wrd.lower()

        if not wrd in analyses: analyses[wrd] = set()
        if len(wrd_analysis): analyses[wrd].add(wrd_analysis)

    return analyses


def analysis_counts(analyses):

    words_w_analyses = set()
    words_wo_analyses = set()
    words_w_uc_analysis = set()
    words_w_lc_analysis = set()

    for word, word_analyses in analyses.items():

        lc_word = word.lower()
        lowercase = lc_word == word

        if len(word_analyses):
            words_w_analyses.add(lc_word)
            if lowercase: words_w_lc_analysis.add(lc_word)
            else: words_w_uc_analysis.add(lc_word)
        else:
            words_wo_analyses.add(lc_word)

    num_uc = len(words_w_uc_analysis)
    num_lc = len(words_w_lc_analysis)
    num_both = len(words_w_uc_analysis & words_w_lc_analysis)

    return len(words_w_analyses), num_uc, num_lc, num_both


def merge_uc_lc(analyses):

    uc_words = set()
    for word, word_analyses in analyses.items():
        lc_word = word.lower()
        if word == lc_word: continue
        if not lc_word in analyses: analyses[lc_word] = set()
        analyses[lc_word] |= analyses[word]
        uc_words.add(word)
    for uc_word in uc_words:
        del analyses[uc_word]

    return analyses


def get_num_analyses_per_word(analyses):
    anal_counts = dict()
    for word in analyses.keys():
        if not analyses[word]: continue
        anal_count = len(analyses[word])
        if not anal_count in anal_counts:
            anal_counts[anal_count] = 0
        anal_counts[anal_count] += 1

    return anal_counts


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

    if len(sys.argv) < 2:
        print >>sys.stderr, "USAGE: ANALYSIS"
        sys.exit(1)

    locale.setlocale(locale.LC_ALL, 'en_US.utf8')
    analyses = parse_analyses(sys.argv[1])

    num_w_analyses, num_uc, num_lc, num_both = analysis_counts(analyses)
    analyses = merge_uc_lc(analyses)

    print >>sys.stderr, "Number of words: %i" % len(analyses)
    print >>sys.stderr, "Words with analysis: %i" % num_w_analyses
    print >>sys.stderr, "Words with uppercase analysis: %i" % num_uc
    print >>sys.stderr, "Words with lowercase analysis: %i" % num_lc
    print >>sys.stderr, "Words with both upper and lowercase analysis: %i" % num_both

    print >>sys.stderr, ""
    counts = get_num_analyses_per_word(analyses)
    for analc, c in counts.items():
        print >>sys.stderr, "Words with %i analyses: %i" % (analc, c)

    print >>sys.stderr, ""
    anal_types = get_analysis_classes(analyses)
    print >>sys.stderr, "Distinct morphological classes: %i" % len(anal_types)

#    for anal_type, count in anal_types.items():
#        print "%s\t%i" % (anal_type, count)

