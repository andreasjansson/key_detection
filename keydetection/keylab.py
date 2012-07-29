import csv
import os.path
import re
from key import *

simple_keymap = {'C': MajorKey(0), 'C#': MajorKey(1), 'Db': MajorKey(1),
                 'D': MajorKey(2), 'D#': MajorKey(3), 'Eb': MajorKey(3),
                 'E': MajorKey(4), 'F': MajorKey(5), 'F#': MajorKey(6),
                 'Gb': MajorKey(6), 'G': MajorKey(7), 'G#': MajorKey(8),
                 'Ab': MajorKey(8), 'A': MajorKey(9), 'A#': MajorKey(10),
                 'Bb': MajorKey(10), 'B': MajorKey(11),
                 'C:minor': MinorKey(3), 'C#:minor': MinorKey(4), 'Db:minor': MinorKey(4),
                 'D:minor': MinorKey(5), 'D#:minor': MinorKey(6), 'Eb:minor': MinorKey(6),
                 'E:minor': MinorKey(7), 'F:minor': MinorKey(8), 'F#:minor': MinorKey(9),
                 'Gb:minor': MinorKey(9), 'G:minor': MinorKey(10), 'G#:minor': MinorKey(11),
                 'Ab:minor': MinorKey(11), 'A:minor': MinorKey(0), 'A#:minor': MinorKey(1),
                 'Bb:minor': MinorKey(1), 'B:minor': MinorKey(2), 'Silence': None}

class LabParser(object):

    def parse_keys(self, filename, keymap = simple_keymap):
        handle = open(filename, 'r')
        reader = csv.reader(handle, dialect = "excel-tab")
        keys = []
        for row in reader:
            if len(row) == 4:
                key_name = row[3]
                if key_name in keymap:
                    key = keymap[key_name]
                else:
                    key = None
            else:
                key = None
            start_time = float(row[0])
            end_time = float(row[1])
            keys.append((key, start_time, end_time))
        handle.close()
        return keys

    def parse_beats(self, filename):
        beats = []
        with open(filename, 'r') as f:
            reader = csv.reader(f, dialect = "excel-tab")
            for row in reader:
                time = float(row[0])
                beat = int(row[1])
                beats.append(Beat(beat, time))
        return beats

class KeyLab(object):

    def __init__(self, lab_file):
        self.keys = LabParser().parse_keys(lab_file)

    def is_valid(self):
        return self.keys is not None

    def key_at(self, time):
        # brute force for now
        for k, t, _ in reversed(self.keys):
            if t <= time:
                return k
        return None

    def majority_key(self):
        time_per_key = {}
        for key, start_time, end_time in self.keys:
            if key not in time_per_key:
                time_per_key[key] = end_time - start_time
            else:
                time_per_key[key] += end_time - start_time
        return max(time_per_key, key = time_per_key.get)

    def real_keys(self):
        keys = []
        for key in self.keys:
            if key[0] is not None:
                keys.append(key[0])
        return keys

    def key_count(self):
        return len(self.real_keys())

class LilyKeyLab(KeyLab):

    def __init__(self, filename):
        with open(filename, 'r') as f:
            self.key = self.grep_key(f.read())

    def is_valid(self):
        return self.key is not None

    def key_at(self, time):
        return self.key

    def majority_key(self):
        return self.key

    def real_keys(self):
        return [self.key]

    def grep_key(self, text):

        notemap = {
            'cb':   11,
            'cf':   11,
            'ces':  11,
            'c':    0,
            'cis':  1,
            'cs':   1,
            'cd':   1,
            'db':   1,
            'df':   1,
            'des':  1,
            'd':    2,
            'dis':  3,
            'ds':   3,
            'dd':   3,
            'es':   3,
            'eb':   3,
            'ef':   3,
            'ees':  3,
            'e':    4,
            'eis':  5,
            'ed':   5,
            'fb':   4,
            'ff':   4,
            'fes':  4,
            'f':    5,
            'fis':  6,
            'fs':   6,
            'fd':   6,
            'gb':   6,
            'gf':   6,
            'ges':  6,
            'g':    7,
            'gis':  8,
            'gs':   8,
            'gd':   8,
            'ab':   8,
            'af':   8,
            'aes':  8,
            'a':    9,
            'ais':  10,
            'as':   10,
            'ad':   10,
            'bb':   10,
            'bf':   10,
            'bes':  10,
            'h':    11,
            'bis':  0,
            'bs':   0,
            'bd':   0,
            'dob':  11,
            'do':   0,
            'dod':  1,
            'reb':  1,
            're':   2,
            'red':  3,
            'mib':  3,
            'mi':   4,
            'mid':  5,
            'fab':  4,
            'fa':   5,
            'fad':  6,
            'solb': 6,
            'sol':  7,
            'sold': 8,
            'lab':  8,
            'la':   9,
            'lad':  10,
            'sib':  10,
            'si':   11,
            'sid':  0,
            }

        matches = re.finditer('\\\key +([a-z]+) *\\\(major|minor)+', text)
        key = prev_key = None

        for match in matches:
            keyname, mode = match.groups()
            if not keyname in notemap:
                return None
            if mode == 'major':
                key = MajorKey(notemap[keyname])
            else:
                key = MinorKey(notemap[keyname])
            if prev_key is not None and key != prev_key:
                return None
            prev_key = key

        return key

class Beat(object):
    def __init__(self, beat, time):
        self.beat = beat
        self.time = time

    def __str__(self):
        return "{0}: {1}".format(self.time, self.beat)


def get_key_lab(filename):
    name, extension = os.path.splitext(filename)
    if extension == '.lab':
        return KeyLab(filename)
    elif extension == '.ly':
        return LilyKeyLab(filename)
    raise Exception("Don't have a KeyLab to deal with %s files" % extension)

