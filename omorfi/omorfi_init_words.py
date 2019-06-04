#!/usr/bin/python3
# -*- coding: utf-8 -*-

import codecs
import locale
import argparse
from omorfi_parse import read_analyses
from omorfi_parse import get_analysis_classes


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
    parser.add_argument("ANALYSES", help="Output from the Omorfi analyzer without the -X switch.")
    parser.add_argument("LARGE_COVERAGE_ANALYSES", help="Output from the Omorfi analyzer with the -X switch.")
    parser.add_argument('WORD_INIT', help='Word initializations to be written')
    parser.add_argument('CLASS_DEFS', help='Class information to be written')
    parser.add_argument("--encoding", action="store", type=str, default="utf-8")
    args = parser.parse_args()

    locale.setlocale(locale.LC_ALL, 'en_US.utf8')

    analyses = read_analyses(args.ANALYSES, args.LARGE_COVERAGE_ANALYSES, True)

    classes = get_analysis_classes(analyses)
    clsmap = dict(map(reversed, enumerate(sorted(classes))))

    word_state_vectors = initialize(analyses, clsmap)

    wordf = codecs.open(args.WORD_INIT, "w", encoding=args.encoding)
    for word, vector in word_state_vectors.items():
        print("%s\t%s" % (word, " ".join(map(str, vector))), file=wordf)
    wordf.close()

    classf = open(args.CLASS_DEFS, "w")
    rev_clsmap = dict()
    for clss, idx in clsmap.items():
        rev_clsmap[idx] = clss
    for idx, clss in rev_clsmap.items():
        print("%i\t%s" % (idx, clss), file=classf)
    classf.close()
