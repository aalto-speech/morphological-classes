#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import argparse
from collections import defaultdict

from estnltk_init_words import TAG_SEPARATOR


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Construct super class definitions for Estnltk classes')
    parser.add_argument('classes', help='Estnltk classes with indices')
    parser.add_argument('super_classes', help='Resulting super classes of the Estnltk classes')
    args = parser.parse_args()

    classes = dict()
    super_classes = defaultdict(set)

    expected_idx = 0
    classf = open(args.classes)
    for line in classf:
        line = line.strip()
        idx, estnltk_class = line.split()
        idx = int(idx)
        if expected_idx != idx:
            raise Exception("Problem in class indexing: %s" % line)
        expected_idx += 1

        estnltk_tags = estnltk_class.split(TAG_SEPARATOR)
        super_classes[estnltk_tags[0]].add(idx)

    sclassf = open(args.super_classes, "w")
    for class_label, subclasses in super_classes.items():
        sorted_subclasses = map(lambda x: str(x), sorted(subclasses))
        print(",".join(sorted_subclasses), file=sclassf)
        print("%s: %i" % (class_label, len(subclasses)), file=sys.stderr)
    sclassf.close()

