# -*- coding: utf-8 -*-
#!/usr/bin/env python

""" Utility code. """

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import glob


def split_path(path):
    """
    'a/b/c.wav' => ('a/b', 'c', 'wav')
    :param path: filepath = 'a/b/c.wav'
    :return: basename, filename, and extension = ('a/b', 'c', 'wav')
    """
    basepath, filename = os.path.split(path)
    filename, extension = os.path.splitext(filename)
    return basepath, filename, extension


def remove_all_files(path):
    files = glob.glob('{}/*'.format(path))
    for f in files:
        os.remove(f)
