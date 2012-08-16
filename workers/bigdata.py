from music21 import corpus
from music21.stream import Opus
import sys
import os
import random
import argparse
import math

sys.path.insert(0, os.path.abspath('..'))
from keydetection import *

def print_keys(n, t):

    paths = sorted(corpus.getCorePaths())
    paths = partition(paths, n, t)
    print paths[0]

    for path in paths:
        try:
            piece = corpus.parse(path)
        except Exception:
            sys.stderr.write('Failed to parse %s\n' % path)
            continue
        if isinstance(piece, Opus):
            for score in piece.scores:
                print_keys_for_score(score, path)
        else:
            print_keys_for_score(piece, path)


def print_keys_for_score(score, path):

    keys = score.parts[0].getKeySignatures()
    if len(keys) == 0:
        sys.stderr.write('No keys for %s\n' % path)
    if len(keys) > 1:
        sys.stderr.write('Too many keys for %s: %d\n' % (path, len(keys)))
        return

    key = keys[0]
    if not key.mode:
        sys.stderr.write('No mode for %s\n' % path)
        return

    root_pitch, mode = key.pitchAndMode
    root = root_pitch.pitchClass
    if mode == 'major':
        key = MajorKey(root)
    elif mode == 'minor':
        key = MinorKey(root)
    else:
        sys.stderr.write('Unknown mode for %s: %s\n' % (path, mode))
        return

    print '%s: %s' % (path, key)


def partition(array, n, t):
    part = len(array) / float(t)
    start = int(math.floor(part * n))
    end = int(math.floor(part * (n + 1)))
    return array[start:end]
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'bigdata')
    parser.add_argument('-n', '--nth-worker', type = int)
    parser.add_argument('-t', '--total-workers', type = int)
    parser.add_argument('command', choices = ['printkeys'])
    args = parser.parse_args()

    if args.command == 'printkeys':
        print_keys(args.nth_worker - 1, args.total_workers)
