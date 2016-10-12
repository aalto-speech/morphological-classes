#!/usr/bin/python

import re
import sys


def parse_line(line, analyses):
    m = re.search(r".\((.*?)\):(.):(.*)", line)
    word = m.group(1)
    anali = int(m.group(2))
    anal = m.group(3)

    if not word in analyses:
        analyses[word] = dict()
    if anal.startswith("STRUCTURE"): return
    if "MALAGA" in anal: return
    if not anali in analyses[word]:
        analyses[word][anali] = anal
    else:
        analyses[word][anali] += "_%s" % anal


def parse_analyses(fname):

    analf = open(fname)
    analyses = dict()
    for line in analf:
        line = line.strip()
        if line.startswith("C: "):
            word = line.replace("C: ", "")
        elif line.startswith("W: "):
            word = line.replace("W: ", "")
            analyses[word] = set()
        elif line.startswith("A("):
            parse_line(line, analyses)

    # There are some duplicate analyses with only different structure
    for word, word_analyses in analyses.items():
        if not word_analyses: continue
        analyses[word] = set(word_analyses.values())

    return analyses


def analysis_counts(analyses):

    words_w_analyses = 0
    words_wo_analyses = 0
    words_w_only_uc = 0

    for word, word_analyses in analyses.items():
        lc_word = word.decode("utf-8").lower().encode("utf-8")
        if word == lc_word: continue

        if len(word_analyses) and len(analyses[lc_word]):
            words_w_analyses += 1
        elif len(word_analyses):
            words_w_analyses += 1
            words_w_only_uc += 1
        elif len(analyses[lc_word]):
            words_w_analyses += 1
        else: words_wo_analyses += 1

    return words_w_analyses, words_w_only_uc, words_wo_analyses


def merge_uc_lc(analyses):

    uc_words = set()
    for word, word_analyses in analyses.items():
        lc_word = word.decode("utf-8").lower().encode("utf-8")
        if word == lc_word: continue
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

    analyses = parse_analyses(sys.argv[1])

    words_w_analyses, words_w_only_uc, words_wo_analyses = analysis_counts(analyses)
    analyses = merge_uc_lc(analyses)

    print >>sys.stderr, "Number of words: %i" % len(analyses)
    print >>sys.stderr, "Words with analysis: %i" % words_w_analyses
    print >>sys.stderr, "Words with only uppercase analysis: %i" % words_w_only_uc
    print >>sys.stderr, "Words without analysis: %i" % words_wo_analyses

    print >>sys.stderr, ""
    counts = get_num_analyses_per_word(analyses)
    for analc, c in counts.items():
        print >>sys.stderr, "Words with %i analyses: %i" % (analc, c)

    print >>sys.stderr, ""
    anal_types = get_analysis_classes(analyses)
    print >>sys.stderr, "Distinct morphological classes: %i" % len(anal_types)

    #for anal_type, count in anal_types.items():
    #    print "%s\t%i" % (anal_type, count)
