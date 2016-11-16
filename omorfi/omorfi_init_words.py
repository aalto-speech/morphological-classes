#!/usr/bin/python

import sys
import locale
import argparse
from omorfi_parse import parse_analyses
from omorfi_parse import get_analysis_classes
from omorfi_parse import merge_uc_lc


def get_class_idx_map(classes):
    idx = 0
    idx_map = dict()
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

    return word_states


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Initialize word categories with Omorfi analyses')
    parser.add_argument('analyses', help='Omorfi analyses')
    parser.add_argument('word_init', help='Word initializations to be written')
    parser.add_argument('class_defs', help='Class information to be written')
    parser.add_argument("--encoding", action="store", type=str, default="utf-8")
    args = parser.parse_args()

    analyses = parse_analyses(args.analyses, encoding=args.encoding)
    analyses = merge_uc_lc(analyses)
    classes = get_analysis_classes(analyses)
    clsmap = get_class_idx_map(classes)
    word_state_vectors = initialize(analyses, clsmap)

    wordf = open(args.word_init, "w")
    for word, vector in word_state_vectors.items():
        print >>wordf, "%s\t%s" % (word.encode(args.encoding), " ".join(map(str, vector)))
    wordf.close()

    classf = open(args.class_defs, "w")
    rev_clsmap = dict()
    for clss, idx in clsmap.items():
        rev_clsmap[idx] = clss
    rev_clsmap[0] = "<s>"
    rev_clsmap[1] = "<unk>"
    for idx, clss in rev_clsmap.items():
        print >>classf, "%i\t%s" % (idx, clss)
    classf.close()
