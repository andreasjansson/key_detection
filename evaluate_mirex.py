import re
import argparse

from keydetection import *

logging.basicConfig(level = logging.DEBUG)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'evaluate mirex output')
    parser.add_argument('mirex')
    parser.add_argument('truth')
    args = parser.parse_args()

    scoreboard = Scoreboard()

    truth = {}
    with open(args.truth, 'r') as f:
        for line in f.readlines():
            match = re.search(r'lilymp3/(.+)::key:\<(Major|Minor)Key: ([A-G]\#?)', line)
            if match:
                basename, mode, root = match.groups()
                if mode == 'Major':
                    key = MajorKey(note_names.index(root))
                else:
                    key = MinorKey(note_names.index(root))
                truth[basename] = key

    with open(args.mirex, 'r') as f:
        for line in f.readlines():
            match = re.search(r'lilymp3/(.+)::([A-G]\#?)\t(minor|major)', line)
            if match:
                basename, root, mode = match.groups()

                if basename not in truth:
                    continue

                if mode == 'major':
                    key = MajorKey(note_names.index(root))
                else:
                    key = MinorKey(note_names.index(root))

                actual_key = truth[basename]
                diff = actual_key.compare(key)

                logging.info('%s: Predicted: %s; Actual: %s; Diff: %s' % (basename, key, actual_key, diff.name()))
                scoreboard.add(diff)

    scoreboard.print_scores()

