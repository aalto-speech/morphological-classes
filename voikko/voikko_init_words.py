#!/usr/bin/python

import sys
import locale
from voikko_parse import parse_analyses
from voikko_parse import get_analysis_classes
from voikko_parse import merge_uc_lc


def get_class_idx_map(classes):
    idx_map = dict()
    idx = 2
    for clss in classes:
        idx_map[clss] = idx
        idx += 1
    return idx_map


def state_with_analyses(word_analyses, clsmap):
    classes = set()
    for analysis in word_analyses:
        classes.add(clsmap[analysis])
    return classes


def initialize(analyses, clsmap):
    word_states = dict()
    for word, word_analyses in analyses.items():
        if word_analyses:
            word_states[word] = state_with_analyses(word_analyses, clsmap)
        # in-vocabulary word with no analyses
        else:
            word_states[word] = set()
    word_states["<s>"] = {0}
    word_states["<unk>"] = {1}
    #word_states["</s>"] = {0}

    return word_states


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print >>sys.stderr, "USAGE: INIT_WORDS ANALYSIS"
        sys.exit(1)

    locale.setlocale(locale.LC_ALL, 'en_US.utf8')
    analyses = parse_analyses(sys.argv[1])
    analyses = merge_uc_lc(analyses)
    classes = get_analysis_classes(analyses)
    clsmap = get_class_idx_map(classes)
    word_state_vectors = initialize(analyses, clsmap)

    for word, vector in word_state_vectors.items():
        print "%s\t%s" % (word, " ".join(map(str, vector)))
