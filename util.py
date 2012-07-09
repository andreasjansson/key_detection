# TODO: test with different training sets;
#       test with "handcoded" model
#       why is yellow submarine tested as C# major!!!?
#       normalise matrices to 1, otherwise we'll never match minor keys
#       find collection of annotated classical music (where the
#         annotation format supports key, e.g. abc), synthesise with
#         timidity and use these as training data. compare to
#         beatles data.

import operator
import tempfile
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import numpy as np
import math
import os
import csv
from scipy import fft
from scipy.spatial.distance import cosine
#import sqlite3 as sqlite
import sys
import string
from copy import copy
#import matplotlib.pyplot as plt
import copy
import os.path
from glob import glob
import hashlib
import pickle
import urllib
import boto.s3.key
import re
import pdb

note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
CACHING = False

class Key(object):

    def __init__(self, root):
        self.root = root

    def __hash__(self):
        return self.root

    def __eq__(self, other):
        return type(self) == type(other) and hash(self) == hash(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def compare(self, other):
        if self == other:
            return KeySame()

        if type(self) == type(other) and \
                ((self.root - other.root) % 12 == 7 or \
                     (other.root - self.root) % 12 == 7):
            return KeyFifth()

        if type(self) != type(other) and \
                isinstance(other, Key) and \
                self.root == other.root:
            return KeyParallel()

        return KeyOther()

class MajorKey(Key):

    def __repr__(self):
        return '<MajorKey: %s>' % note_names[self.root]

    def compare(self, other):
        if isinstance(other, MinorKey) and \
                (self.root - 3) % 12 == other.root:
            return KeyRelative()
        return Key.compare(self, other)
    
class MinorKey(Key):

    def __repr__(self):
        return '<MinorKey: %s>' % note_names[self.root]

    def compare(self, other):
        if isinstance(other, MajorKey) and \
                (self.root + 3) % 12 == other.root:
            return KeyRelative()
        return Key.compare(self, other)


# Based on http://www.music-ir.org/mirex/wiki/2012:Audio_Key_Detection#Evaluation_Procedures

class KeyDiff(object):
    def __hash__(self):
        return 0
    def __eq__(self, other):
        return type(self) == type(other) and hash(self) == hash(other)

class KeySame(KeyDiff):
    def score(self):
        return 1.0
    def name(self):
        return 'Same'

class KeyFifth(KeyDiff):
    def score(self):
        return 0.5
    def name(self):
        return 'Perfect Fifth'

class KeyRelative(KeyDiff):
    def score(self):
        return 0.3
    def name(self):
        return 'Relative Major/Minor'

class KeyParallel(KeyDiff):
    def score(self):
        return 0.2
    def name(self):
        return 'Parallel Major/Minor'

class KeyOther(KeyDiff):
    def score(self):
        return 0.0
    def name(self):
        return 'Other'


class Scoreboard(object):

    def __init__(self):
        self.diffs = {
            KeySame(): 0,
            KeyFifth(): 0,
            KeyRelative(): 0,
            KeyParallel(): 0,
            KeyOther(): 0
            }

    def add(self, diff):
        self.diffs[diff] += 1

    def get_score(self):
        score = 0
        for diff, count in self.diffs.iteritems():
            score += diff.score() * count
        return score

    def max_score(self):
        return sum(self.diffs.values())

    def print_scores(self):
        print '\nTotal score: %.2f / %.2f (%.2f%%)\n' % (self.get_score(), self.max_score(), self.get_score() / self.max_score() * 100)
        for diff, count in self.diffs.iteritems():
            print '%s: %d' % (diff.name(), count)

#class Key(object):
#    def __init__(self, key, time):
#        self.key = key
#        self.time = time
#
#    def __str__(self):
#        return "{0}: {1}".format(self.time, self.key)

class Beat(object):
    def __init__(self, beat, time):
        self.beat = beat
        self.time = time

    def __str__(self):
        return "{0}: {1}".format(self.time, self.beat)


class AudioReader(object):

    def _process(self, samp_rate, stereo, length, downsample_factor):

        if len(stereo.shape) == 2:
            mono = stereo[:,0]
        else:
            mono = stereo

        if length and len(mono) / samp_rate > length:
            mono = mono[0:int(length * samp_rate)]

        # pad with zeroes before downsampling
        padding = np.array([0] * (downsample_factor - (len(mono) % downsample_factor)), dtype = mono.dtype)
        mono = np.concatenate((mono, padding))
        # downsample
        if downsample_factor > 1:
            mono = downsample(mono, downsample_factor)
        return (samp_rate / downsample_factor, mono)

class WavReader(AudioReader):

    def read(self, wav_filename, length = None, downsample_factor = 4):
        samp_rate, stereo = wavfile.read(wav_filename)
        return self._process(samp_rate, stereo, length, downsample_factor)

class Mp3Reader(AudioReader):

    def read(self, mp3_filename, length = None, downsample_factor = 4):
        """
        Returns (sampling_rate, data), where the sampling rate is the
        original sampling rate, downsampled by a factor of 4, and
        the data signal is a downsampled, mono (left channel) version
        of the original signal.
        """

        wav_filename = tempfile.NamedTemporaryFile(suffix = '.wav', delete = False).name
        self._mp3_to_wav(mp3_filename, wav_filename)
        samp_rate, stereo = wavfile.read(wav_filename)
        os.unlink(wav_filename)
        return self._process(samp_rate, stereo, length, downsample_factor)
        
    def _mp3_to_wav(self, mp3_filename, wav_filename):
        if mp3_filename.find('http') == 0:
            mp3_filename = download(mp3_filename, '.mp3')

        if not os.path.exists(mp3_filename):
            raise IOError('File not found')
        os.system("mpg123 -w \"" + wav_filename.replace('"', '\\"') + "\" \"" + mp3_filename.replace('"', '\\"') + "\" &> /dev/null")
        if not os.path.exists(wav_filename):
            raise IOError('Failed to create wav file')


class Template(object):
    def match(self, chromagram):
        max_score = float("-inf")
        max_i = -1
        for i in range(12):
            profile = np.roll(self.profile, i)
            score = sum(profile * chromagram.values)
            if score > max_score:
                max_score = score
                max_i = i
        return max_i
        

class BasicTemplate(Template):
    def __init__(self):
        self.profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])


class Chromagram(object):
    """
    This an n-bin narrow-band chromagram tuned to 440Hz.
    """

    def __init__(self, values = None, chroma_bins = None):

        if values is None:
            self.values = np.zeros(chroma_bins)
            self.chroma_bins = chroma_bins
        elif len(values) < 2:
            raise Exception('At least two values are required for a chromagram')
        elif values is not None and chroma_bins is not None:
            raise Exception('Please specify values or chroma_bins, not both.')
        else:
            self.values = values
            self.chroma_bins = len(values)

    @staticmethod
    def from_spectrum(spectrum, samp_rate,
                      chroma_bins = 12, band_fqs = None):
        """
        spectrum is only left half of the spectrum, so its length
        is signal_length / 2.
        """

        chromagram = Chromagram(chroma_bins = chroma_bins)
        window_size = len(spectrum) * 2
        samp_rate = float(samp_rate)

        if band_fqs is not None:
            band = map(lambda b: len(spectrum) * b / samp_rate, band_fqs)
            subspectrum = spectrum[int(band[0]):int(band[1])]
            freqs = np.arange(band[0], band[1]) * 2 * samp_rate / window_size
        else:
            subspectrum = spectrum
            freqs = np.arange(0, len(spectrum)) * samp_rate / window_size

        c0 = 16.3516
        for i, val in enumerate(subspectrum):
            freq = freqs[i]
            if freq > 0: # disregard dc offset
                bin = int(round(chroma_bins * math.log(freq / c0, 2))) % chroma_bins
                chromagram.values[bin] += math.sqrt(val)

        return chromagram

    def normalise(self):
        if self.values.max() == 0:
            self.values = np.zeros(chroma_bins)
        else:
            self.values = self.values / self.values.max()

    def plot(self):
        plot_chroma(self.values, self.chroma_bins)

    def get_zweiklang(self, threshold = .1, silent = 100):
        # first, determine if it's a nullklang, einklang or zweiklang
        sorted_values = np.sort(self.values)[::-1]

        if sorted_values[0] < silent:
            return Nullklang()

        if sorted_values[1] < silent:
            return Einklang(np.where(self.values == sorted_values[0])[0][0])
        
        # zweiklang
        if sorted_values[0] == sorted_values[1]:
            first = np.where(self.values == sorted_values[0])[0][0]
            second = np.where(self.values == sorted_values[0])[0][1]
        else:
            first = np.where(self.values == sorted_values[0])[0][0]
            second = np.where(self.values == sorted_values[1])[0][0]

        # likely to be noise if adjacent
        if abs(second - first) == 1 or abs(second - first) == 11:
            return Einklang(first)

        return Zweiklang(first, second)
        

        # old way:

        # nullklang (ambiguous)
        if sorted_values[0] * threshold < sorted_values[1] and \
                sorted_values[0] * threshold < sorted_values[2]:
            return Nullklang()

        # einklang
        if sorted_values[0] * threshold > sorted_values[1] and \
                sorted_values[0] * threshold > sorted_values[2]:
            return Einklang(np.where(self.values == sorted_values[0])[0][0])

        # zweiklang
        if sorted_values[0] == sorted_values[1]:
            first = np.where(self.values == sorted_values[0])[0][0]
            second = np.where(self.values == sorted_values[0])[0][1]
        else:
            first = np.where(self.values == sorted_values[0])[0][0]
            second = np.where(self.values == sorted_values[1])[0][0]
        return Zweiklang(first, second)

    def plot(self, show = True, yticks = True):
        ind = np.arange(len(self.values))
        plt.bar(ind, self.values)
        xticks = reduce(lambda x, y: x + ([y] * (self.chroma_bins / 12)), note_names, [])
        plt.xticks(ind + .4, xticks)
        if not yticks:
            plt.gca().axes.get_yaxis().set_visible(False)
        if show:
            plt.show()

class Nklang(object):

    def get_number(self):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

class Nullklang(Nklang):

    def __init__(self):
        pass

    def get_name(self):
        return '-'

    def get_number(self):
        return -1

    def transpose(self, _):
        return Nullklang();

    def __repr__(self):
        return '<Nullklang>'

class Einklang(Nklang):

    def __init__(self, n):
        self.n = n
    
    def get_name(self):
        return note_names[self.n]

    # effectively a zweiklang, with n1 == n2
    def get_number(self):
        return self.n + 12 * self.n

    def transpose(self, delta):
        return Einklang((self.n + delta) % 12)

    def __repr__(self):
        return '<Einklang: %s>' % note_names[self.n]

class Zweiklang(Nklang):

    def __init__(self, first, second):
        if first is not None and second is not None and first > second:
            self.first = second
            self.second = first
        elif first is None and second is not None:
            raise Exception('Only second is not allowed')
        else:
            self.first = first
            self.second = second
        
    def get_name(self):
        return note_names[self.first] + ', ' + note_names[self.second]

    def get_number(self):
        return self.first + 12 * self.second
        
    def transpose(self, delta):
        return Zweiklang((self.first + delta) % 12, (self.second + delta) % 12)

    def __repr__(self):
        return '<Zweiklang: %s, %s>' % (note_names[self.first], note_names[self.second])


def plot_chromas(chromas, chroma_bins = 12):
    root = math.sqrt(len(chromas))
    cols = int(math.floor(root))
    rows = int(math.ceil(len(chromas) / float(cols)))
    for i, chroma in enumerate(chromas):
        row = int(math.floor(i / float(cols)))
        col = i % cols
        plt.subplot2grid((rows, cols), (row, col))
        chroma.plot(show = False, yticks = False)
    plt.show()

simple_keymap = {'C': MajorKey(0), 'C#': MajorKey(1), 'Db': MajorKey(1), 'D': MajorKey(2), 'D#': MajorKey(3), 'Eb': MajorKey(3),
                 'E': MajorKey(4), 'F': MajorKey(5), 'F#': MajorKey(6), 'Gb': MajorKey(6), 'G': MajorKey(7), 'G#': MajorKey(8),
                 'Ab': MajorKey(8), 'A': MajorKey(9), 'A#': MajorKey(10), 'Bb': MajorKey(10), 'B': MajorKey(11),
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

def get_key_lab(filename):
    name, extension = os.path.splitext(filename)
    if extension == '.lab':
        return KeyLab(filename)
    elif extension == '.ly':
        return LilyKeyLab(filename)
    raise Exception("Don't have a KeyLab to deal with %s files" % extension)

class Table(object):

    def __init__(self, database, table, columns, auto_id = True):
        self.database = database
        self.table = table
        self.auto_id = auto_id
        if auto_id:
            columns = [('id', 'INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT')] + columns
        self.columns = columns
        self.con = sqlite.connect(self.database)
        self.cur = self.con.cursor()

    def create_table(self):
        columns_string = ', '.join(map(' '.join, self.columns))
        sql = 'CREATE TABLE %s (%s)' % (self.table, columns_string)
        self.cur.execute(sql)
        self.con.commit()

    def truncate_table(self):
        sql = 'DELETE FROM %s' % (self.table)
        self.cur.execute(sql)
        self.con.commit()

    def insert(self, row):
        expected_columns = len(self.columns) - 1 if self.auto_id else len(self.columns)
        if len(row) != expected_columns:
            message = "Column length %d doesn't match expected column length %d" % \
                (len(row), expected_columns)
            raise Exception(message)

        row = "'" + "', '".join(str(x).replace("'", "\\'") for x in row) + "'"
        if self.auto_id:
            sql = "INSERT INTO %s VALUES (NULL, %s)" % (self.table, row)
        else:
            sql = "INSERT INTO %s VALUES (%s)" % (row)
        self.cur.execute(sql)
        self.con.commit()

    def select(self, columns, where = None, order = None, as_dict = True):
        table_columns = map(lambda x: x[0], self.columns)
        columns = map(string.strip, columns)
        columns = reduce(lambda x, y: x + (table_columns if y == '*' else [y]), columns, [])

        if len(set(columns).difference(table_columns)) != 0:
            raise Exception('Columns don\'t match')
        if where:
            where_clause = 'WHERE ' + where
        else:
            where_clause = ''
        if order:
            order_clause = 'ORDER BY ' + order
        else:
            order_clause = ''
        sql = 'SELECT %s FROM %s %s %s' % \
            ((', ').join(columns), self.table, where_clause, order_clause)

        result = self.con.execute(sql)
        if as_dict:
            return [dict(zip(columns, row)) for row in result.fetchall()]
        else:
            return result.fetchall()

class ChromaTable(Table):

    def __init__(self, database, chroma_bins, bands_count, table_name):
        self.chroma_bins = chroma_bins
        self.bands_count = bands_count
        columns = [('track_id', 'INTEGER NOT NULL'),
                   ('i', 'INTEGER NOT NULL'),
                   ('key', 'INTEGER NOT NULL')] + self.get_chroma_columns()
        Table.__init__(self, database, table_name, columns)

    def get_chroma_columns(self):
        return [('chroma_%d_%d' % (i, j),  'FLOAT NOT NULL')
                for i in range(self.bands_count)
                for j in range(self.chroma_bins)]

    def select_chroma(self, where = None, order = None, as_dict = True):
        return self.select(dict(self.get_chroma_columns()).keys(), where, order, as_dict)
        

class RawTable(ChromaTable):

    def __init__(self, database, chroma_bins = 3 * 12, bands_count = 3):
        ChromaTable.__init__(self, database, chroma_bins, bands_count, 'raw')

class TunedTable(ChromaTable):

    def __init__(self, database, chroma_bins = 12, bands_count = 3):
        ChromaTable.__init__(self, database, chroma_bins, bands_count, 'raw')

# TODO: higher order
class HMM(object):
    """
    Basic 1st order HMM, can only compute Viterbi path.
    Takes a np matrix of trained profiles as emission inputs.
    Originally based on the Wikipedia implementation.
    """

    def __init__(self, profiles, trans_probs, start_probs):

        assert len(profiles) == len(start_probs) == \
            len(trans_probs) == len(trans_probs[0])

        self.profiles = profiles
        self.trans_probs = trans_probs
        self.start_probs = start_probs

    # TODO: how to incorporate std dev? in get_emission_probability?
    def viterbi(self, emissions):

        assert len(emissions[0]) == len(self.profiles[0])
        nstates = len(self.profiles)

        v = [{}]
        path = {}

        for state in range(nstates):
            v[0][state] = self.start_probs[state] * self.get_emission_probability(emissions[0], state)
            path[state] = [state]
            
        for t in range(1, len(emissions)):
            v.append({})
            new_path = {}
            emission = emissions[t]

            for state in range(nstates):
                emission_probability = self.get_emission_probability(emission, state)
                (prob, prev_state) = max([(v[t - 1][prev] * self.trans_probs[prev][state] * emission_probability, prev) \
                                              for prev in range(nstates)])
                v[t][state] = prob
                new_path[state] = path[prev_state] + [state]

            path = new_path

        (prob, state) = max([(v[len(emissions) - 1][s], s) for s in range(nstates)])
        return path[state]

    def get_emission_probability(self, emission, state):
        return dot_product(emission, self.profiles[state])

class Tuner(object):

    def __init__(self, bins_per_pitch, bands, pitches = 12):
        self.bins_per_pitch = bins_per_pitch
        self.bands = bands
        self.pitches = pitches

    def tune(self, chromas):
        """
        Tune multiple bands of chromagrams.
        """
        tuned_chromas = []
        max_bins = [0] * self.bins_per_pitch
        for chroma in chromas:
            max_bins[self.get_max_bin(chroma)] += 1
        # TODO: proper argmax
        max_bin = max_bins.index(max(max_bins))
        for chroma in chromas:
            tuned_chroma = self.tune_chroma(chroma, max_bin)
            tuned_chromas.append(tuned_chroma)
        return tuned_chromas

    def get_max_bin(self, chroma):
        bins = [0] * self.bins_per_pitch
        for i, value in enumerate(chroma.values):
            bins[i % self.bins_per_pitch] += value
        return bins.index(max(bins))

    def tune_chroma(self, chroma, max_bin):
        tuned = []
        for i in range(self.bands):
            tuned += self.tune_band(
                chroma.values[(i * self.pitches * self.bins_per_pitch) :
                        ((i + 1) * self.pitches * self.bins_per_pitch)], max_bin)
        return Chromagram(tuned)

    def tune_band(self, values, max_bin):
        values = self.roll_values(values, max_bin)
        tuned_values = [0] * self.pitches
        for i, value in enumerate(values):
            tuned_values[int(math.floor(i / self.bins_per_pitch))] += value
        return tuned_values

    def roll_values(self, values, max_bin):
        mid = math.floor(self.bins_per_pitch / 2)
        if max_bin <= mid:
            shift = mid - max_bin
        else:
            shift = max_bin
        values = np.roll(values, int(shift)).tolist()
        return values

class SpectrumQuantileFilter(object):

    def __init__(self, quantile = 97, window_width = 200):
        self.quantile = quantile
        self.window_width = 200

    def filter(self, spectrum):
        filtered = []
        for i in range(0, len(spectrum), self.window_width):
            subspec = list(spectrum[i:(i + self.window_width)])
            sortspec = [(i, v) for i, v in enumerate(subspec)]
            sortspec.sort(key = operator.itemgetter(1))
            q = int(len(sortspec) * (self.quantile / 100.0))
            for i in range(q):
                subspec[sortspec[i][0]] = 0
            filtered += subspec
        return filtered

class SpectrumPeakFilter(object):

    def __init__(self, audio, window_size = 8192,
                 samp_rate = 11025, max_peaks = 20):
        self.audio = audio
        self.window_size = window_size
        self.samp_rate = samp_rate
        self.max_peaks = max_peaks

    def filter(self, spectrum, frame):
        import simpl
        spectrum = [0] * len(spectrum)
        audio = self.audio[(self.window_size * frame):
                               (self.window_size * (frame + 1))]
        pd = simpl.SndObjPeakDetection()
        pd.set_sampling_rate(self.samp_rate)
        pd.max_peaks = self.max_peaks
        frames = pd.find_peaks(audio)
        for frame in frames:
            for peak in frame.peaks:
                freq = peak.frequency / 4
                bin = int(freq * self.window_size / self.samp_rate)
                spectrum[bin] += peak.amplitude * 32768.0
            return spectrum

# class EmissionMatrix:

#     def __init__(self):
#         self.matrix = np.matrix(12, 12)

#     def add(self, fr0m, to):
#         self.matrix[fr0m, to] += 1

#     def get(self, fr0m, to):
#         return self.matrix[fr0m, to]

class MarkovMatrix:

    def __init__(self, width = None):
        if width is not None:
            self.m = np.zeros(shape = (width, width))

    @staticmethod
    def from_matrix(m):
        markov = MarkovMatrix()
        markov.m = m
        return markov

    def increment(self, klang1, klang2):
        x = klang1.get_number()
        y = klang2.get_number()
        self.m[x][y] += 1

    # "transpose" in the musical sense, not matrix transposition
    def transpose_key(self, delta):
        width = np.shape(self.m)[0]
        m = np.roll(self.m, delta % width, 0)
        m = np.roll(m, delta % width, 1)
        return MarkovMatrix.from_matrix(m)

    def print_summary(self, max_lines = 20):
        # fucking numpy mutability. side effects everywhere
        # so need to copy all along.
        m = copy.copy(self.m)
        seq = m.reshape(-1)
        seq.sort()
        seq = np.unique(seq)[::-1]
        i = 0
        lines = 0
        while(seq[i] > 0 and i < len(seq)):
            where = np.where(seq[i] == self.m)
            for fr0m, to in zip(where[0], where[1]):
                print '%6s => %-6s: %.3f' % (klang_number_to_name(fr0m), klang_number_to_name(to), seq[i])
                lines += 1
                if lines > max_lines:
                    return
            i += 1

    # mutable for performance
    def add(self, other):
        if np.shape(self.m) != np.shape(other.m):
            raise Error('Cannot add markov matrices of different shapes')
        width = np.shape(self.m)[0]
        for i in range(width):
            for j in range(width):
                self.m[i][j] += other.m[i][j]

    def similarity(self, other):
        # better without sqrt it appears
        # return np.dot(np.sqrt(self.m).ravel(), np.sqrt(other.m).ravel())
        return np.dot(self.m.ravel(), other.m.ravel())

    def normalise(self):
        max = np.max(self.m)
        if max > 0:
            self.m /= max

    def __repr__(self):
        s = np.shape(self.m)
        return '<Matrix %dx%d sum %d>' % (s[0], s[1], np.sum(self.m))


class Cache(object):

    def __init__(self, prefix, key):
        self.name = 'cache_%s_%s.pkl' % (prefix, hashlib.md5(key).hexdigest())
        self._data = None

    def exists(self):
        if not CACHING:
            False

        return self.get() is not None

    # handles broken pickle
    def get(self):
        if not CACHING:
            return None

        if self._data is not None:
            return self._data
        try:
            with open(self.name, 'r') as f:
                self._data = pickle.load(f)
                return self._data
        except Exception:
            return None

    def set(self, data):
        if not CACHING:
            return

        with open(self.name, 'wb') as f:
            pickle.dump(data, f)


def filenames_from_file(filename):
    '''
    Reads from a file with colon-separated (mp3_filename.mp3:lab_filename.lab)
    lines.
    '''
    # TODO
    pass

def filenames_from_twin_directories(mp3_root, lab_root, limit = None):
    '''
    Requires the mp3_root and lab_root to have the exact same structure, with
    identical filenames, except for the file extension
    '''
    mp3_folders = set(os.listdir(mp3_root))
    lab_folders = set(os.listdir(lab_root))
    shared_folders = mp3_folders.intersection(lab_folders)

    i = 0
    for folder in shared_folders:

        mp3_folder = mp3_root + "/" + folder
        lab_folder = lab_root + "/" + folder
        mp3_files = set(map(lambda s: os.path.basename(s).replace(".mp3", ""),
                            glob(mp3_folder + "/*.mp3")))
        lab_files = set(map(lambda s: os.path.basename(s).replace(".lab", ""),
                            glob(lab_folder + "/*.lab")))

        shared_files = mp3_files.intersection(lab_files)

        for f in shared_files:

            if limit is not None and i >= limit:
                raise StopIteration

            mp3_file = mp3_folder + "/" + f + ".mp3"
            lab_file = lab_folder + "/" + f + ".lab"
            i += 1
            yield (mp3_file, lab_file)


def get_klangs(mp3 = None, audio = None):
    fs = 11025
    winlength = 8192

    if mp3:
        _, audio = Mp3Reader().read(mp3)

    s = [spectrum for (t, spectrum) in generate_spectrogram(audio, winlength)]

    filt = SpectrumQuantileFilter(98)
    sf = map(filt.filter, s)

    bins = 3
    cs = [Chromagram.from_spectrum(ss, fs / 4, 12 * bins, (50, 500)) for ss in sf]

    tuner = Tuner(bins, 1)
    ts = tuner.tune(cs)

    return [(i * winlength / fs, t.get_zweiklang()) for i, t in enumerate(ts)]

def get_training_matrices(mp3, keylab_file):
    cache = Cache('training', '%s:%s' % (mp3, keylab_file))
    if cache.exists():
        matrices = cache.get()
    else:
        keylab = get_key_lab(keylab_file)

        # no point doing lots of work if it won't
        # give any results
        if not keylab.is_valid():
            return None

        klangs = get_klangs(mp3)
        matrices = get_markov_matrices(keylab, klangs)
        cache.set(matrices)

    return matrices

def aggregate_matrices(matrices_list):
    aggregate_matrices = [MarkovMatrix(12 * 12) for i in range(12 * 2)]
    for matrices in matrices_list:
        if matrices is not None:
            for i in range(24):
                aggregate_matrices[i].add(matrices[i])

    for matrix in aggregate_matrices:
        matrix.normalise()

    return aggregate_matrices

def get_aggregate_markov_matrices(filenames):
    aggregate_matrices = [MarkovMatrix(12 * 12) for i in range(12 * 2)]
    n = 1
    matrices_list = []
    for mp3, keylab_file in filenames:
        print('Analysing %d (%s)' % (n, mp3))
        matrices = get_training_matrices()
        matrices_list.append(matrices)

    aggregate_matrices = aggregate_matrices(matrices_list)

    return aggregate_matrices

def get_markov_matrices(keylab, klangs):
    '''
    Return one or two matrices in a dict
    keyed by mode.
    '''
    mwidth = 12 * 12

    # 12 major and 12 minor matrices
    matrices = [MarkovMatrix(mwidth) for i in range(12 * 2)]
    prev_klang = None
    prev_key = None
    for t, klang in klangs:
        key = keylab.key_at(t)
        #print key, klang # do this if verbose
        if key is not None and \
                prev_klang is not None and \
                prev_key == key and \
                not isinstance(klang, Nullklang) and \
                not isinstance(prev_klang, Nullklang):

            for i in range(12):

                def t(klang):
                    root = key.root
                    if isinstance(key, MinorKey):
                        root = (root - 3) % 12
                    return klang.transpose(-root + i)

                if isinstance(key, MajorKey):
                    matrices[i].increment(t(prev_klang), t(klang))
                elif isinstance(key, MinorKey):
                    matrices[i + 12].increment(t(prev_klang), t(klang))

        prev_klang = klang
        prev_key = key

    return matrices

def get_test_matrix(mp3):

    cache = Cache('test', mp3)
    if cache.exists():
        return cache.get()

    klangs = get_klangs(mp3)
    matrix = MarkovMatrix(12 * 12)
    prev_klang = None
    for t, klang in klangs:
        if prev_klang is not None and \
                not isinstance(klang, Nullklang) and \
                not isinstance(prev_klang, Nullklang):
            matrix.increment(prev_klang, klang)
        prev_klang = klang

    cache.set(matrix)

    return matrix

def get_key(training_matrices, test_matrix):
    argmax = -1
    maxsim = 0
    for i, matrix in enumerate(training_matrices):
        sim = matrix.similarity(test_matrix)
        if sim > maxsim:
            maxsim = sim
            argmax = i
    argmax = argmax % 24
    if argmax < 12:
        return MajorKey(argmax)
    else:
        return MinorKey(argmax - 12)

def dot_product(a, b):
    return sum(map(operator.mul, a, b))

def generate_spectrogram(audio, window_size):
    for t in xrange(0, len(audio), window_size):
        # windowed spectrogram
        actual_window_size = min(window_size, len(audio) - t)
        spectrum = abs(fft(audio[t:(t + window_size)] * np.hanning(actual_window_size)))
        spectrum = spectrum[0:len(spectrum) / 2]
        yield (t, spectrum)

def normalise_spectra(spectra):
    spectra = copy(spectra)
    for i, spectrum in enumerate(spectra):
        m = max(spectrum)
        if m > 0:
            spectrum = (np.array(spectrum) / max(spectrum)).tolist()
        spectra[i] = spectrum
    return spectra

def downsample(sig, factor):
    fir = signal.firwin(61, 1.0 / factor)
    sig2 = np.convolve(sig, fir, mode="valid")
    sig2 = np.array([int(x) for i, x in enumerate(sig2) if i % factor == 0], dtype = sig.dtype)
    return sig2

def klang_number_to_name(number):
    if number == -1:
        return 'Silence'
    else:
        first = number % 12
        second = int(math.floor(number / 12))
        if first == second:
            return note_names[first]
        else:
            return note_names[first] + ', ' + note_names[second]

def get_key_base(key, keymap):

    # for now, no distinction between major and minor.
    return (0, key)

    if key == -1:
        return (-1, -1)

    swap_keymap = dict((v, k) for k, v in keymap.iteritems())
    name = swap_keymap[key]
    if name.find(':minor'):
        base = 12
    else:
        base = 0
    offset = key - base
    return (base, offset)

def roll_bands(values, offset, bands, bins = 12):
    rolled_values = [0] * (bands * bins)
    for b in range(bands):
        for i in range(bins):
            rolled_values[b * bins + ((i - offset) % bins)] = \
                values[b * bins + i]
    return rolled_values

def set_implicit_keys(totals, keymap):
    all_keys = sorted(set(keymap.values()))
    for key in all_keys:
        if key is not None:
            base, _ = get_key_base(key, keymap)
            totals[key] = np.roll(totals[base * 12], - key % 12).tolist()
    return totals


def download(filename, suffix = ''):
    tmp = tempfile.NamedTemporaryFile(suffix = suffix, delete = False).name
    response = urllib.urlretrieve(filename, tmp)
    return tmp

def s3_download(bucket, s3_filename):
    local_file = tempfile.NamedTemporaryFile(suffix = os.path.splitext(s3_filename)[1], delete = False)
    k = boto.s3.key.Key(bucket)
    k.key = s3_filename
    k.get_contents_to_file(local_file)
    local_file.close
    return local_file.name
