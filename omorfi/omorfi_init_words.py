#!/usr/bin/python3
# -*- coding: utf-8 -*-

import codecs
import locale
import argparse
from omorfi_parse import read_analyses
from omorfi_parse import get_analysis_classes


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Initialize word categories with Omorfi analyses')
    parser.add_argument("ANALYSES", help="Output from the Omorfi analyzer without the -X switch.")
    parser.add_argument("LARGE_COVERAGE_ANALYSES", help="Output from the Omorfi analyzer with the -X switch.")
    parser.add_argument('WORD_INIT', help='Word initializations to be written')
    parser.add_argument('CLASS_DEFS', help='Class definitions to be written')
    parser.add_argument("--encoding", action="store", type=str, default="utf-8")
    args = parser.parse_args()

    locale.setlocale(locale.LC_ALL, 'en_US.utf8')

    analyses = read_analyses(args.ANALYSES, args.LARGE_COVERAGE_ANALYSES, True)

    classes = get_analysis_classes(analyses)

    rev_clsmap = dict(enumerate(sorted(classes.keys())))
    clsmap = dict(map(reversed, enumerate(sorted(classes.keys()))))

    wordf = codecs.open(args.WORD_INIT, "w", encoding=args.encoding)
    for word, word_analyses in sorted(analyses.items()):
        word_class_idxs = map(lambda x: clsmap[x], word_analyses)
        print("%s\t%s" % (word, " ".join(map(str, word_class_idxs))), file=wordf)
    wordf.close()

    classf = open(args.CLASS_DEFS, "w")
    for idx, clss in rev_clsmap.items():
        print("%i\t%s" % (idx, clss), file=classf)
    classf.close()
