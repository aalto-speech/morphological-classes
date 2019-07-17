#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import argparse
from collections import defaultdict

from omorfi_parse import TAG_SEPARATOR


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Construct super class definitions for Omorfi classes')
    parser.add_argument('classes', help='Omorfi classes with indices')
    parser.add_argument('super_classes', help='Resulting super classes of the Omorfi classes')
    args = parser.parse_args()

    classes = dict()
    super_classes = defaultdict(set)

    expected_idx = 0
    classf = open(args.classes)
    for line in classf:
        line = line.strip()
        idx, omorfi_class = line.split()
        idx = int(idx)
        if expected_idx != idx:
            raise Exception("Problem in class indexing: %s" % line)
        expected_idx += 1

        omorfi_tags = omorfi_class.split(TAG_SEPARATOR)
        super_classes[omorfi_tags[0]].add(str(idx))

    sclassf = open(args.super_classes, "w")
    for class_label, subclasses in super_classes.items():
        print(",".join(subclasses), file=sclassf)
        print("%s: %i" % (class_label, len(subclasses)), file=sys.stderr)
    sclassf.close()
