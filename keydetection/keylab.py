import csv
import os.path
import re

from key import *
from util import *

simple_keymap = {'C': MajorKey(0), 'C#': MajorKey(1), 'Db': MajorKey(1),
                 'D': MajorKey(2), 'D#': MajorKey(3), 'Eb': MajorKey(3),
                 'E': MajorKey(4), 'F': MajorKey(5), 'F#': MajorKey(6),
                 'Gb': MajorKey(6), 'G': MajorKey(7), 'G#': MajorKey(8),
                 'Ab': MajorKey(8), 'A': MajorKey(9), 'A#': MajorKey(10),
                 'Bb': MajorKey(10), 'B': MajorKey(11),
                 'C:minor': MinorKey(0), 'C#:minor': MinorKey(1), 'Db:minor': MinorKey(1),
                 'D:minor': MinorKey(2), 'D#:minor': MinorKey(3), 'Eb:minor': MinorKey(3),
                 'E:minor': MinorKey(4), 'F:minor': MinorKey(5), 'F#:minor': MinorKey(6),
                 'Gb:minor': MinorKey(6), 'G:minor': MinorKey(7), 'G#:minor': MinorKey(8),
                 'Ab:minor': MinorKey(8), 'A:minor': MinorKey(9), 'A#:minor': MinorKey(10),
                 'Bb:minor': MinorKey(10), 'B:minor': MinorKey(11),
                 'Silence': None}

class LabParser(object):
    '''
    Parse annotated lab files.
    '''

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
    '''
    Python representation of contents of lab file.
    '''

    def __init__(self, lab_filename):
        lab_filename = make_local(lab_filename, '/tmp/lab.lab')
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
    '''
    Extend KeyLab to accept a lilypond file in place of
    a lab file. Assumes only one key.
    '''

    def __init__(self, lab_filename):
        lab_filename = make_local(lab_filename, '/tmp/ly.ly')
        self.key = lilyparser.get_key(lab_filename)

    def is_valid(self):
        return self.key is not None

    def key_at(self, time):
        return self.key

    def majority_key(self):
        return self.key

    def real_keys(self):
        return [self.key]

class SingleKeyLab(KeyLab):
    '''
    Extend KeyLab to create an artificial KeyLab instances
    with only one, pre-defined key.
    '''

    def __init__(self, key_name):
        self.key = Key.from_repr(key_name)

    def is_valid(self):
        return self.key is not None

    def key_at(self, time):
        return self.key

    def majority_key(self):
        return self.key

    def real_keys(self):
        return [self.key]

class Beat(object):
    def __init__(self, beat, time):
        self.beat = beat
        self.time = time

    def __str__(self):
        return "{0}: {1}".format(self.time, self.beat)


def get_key_lab(filename):
    '''
    Get a KeyLab instance from a string of text. If the string looks
    something like "key:<MajorKey G#>", a SingleKeyLab instance in G#
    major is returned. If the string is a filename ending in .lab a
    real KeyLab instance is returned. If it's a .ly GNU LilyPond file,
    the file is parsed using the functions in lilyparser.py, and a
    LilyKeyLab instance is returned.
    '''

    if filename.find('key:') == 0:
        return SingleKeyLab(filename[4:])

    name, extension = os.path.splitext(filename)
    if extension == '.lab':
        return KeyLab(filename)
    elif extension == '.ly':
        return LilyKeyLab(filename)
    raise Exception("Don't have a KeyLab to deal with %s files" % extension)

