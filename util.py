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
import sqlite3 as sqlite
import sys
import string

class Key:
    def __init__(self, key, time):
        self.key = key
        self.time = time

    def __str__(self):
        return "{0}: {1}".format(self.time, self.key)

class Beat:
    def __init__(self, beat, time):
        self.beat = beat
        self.time = time

    def __str__(self):
        return "{0}: {1}".format(self.time, self.beat)


class AudioReader:

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
        if not os.path.exists(mp3_filename):
            raise IOError('File not found')
        os.system("mpg123 -w \"" + wav_filename.replace('"', '\\"') + "\" \"" + mp3_filename.replace('"', '\\"') + "\" &> /dev/null")
        if not os.path.exists(wav_filename):
            raise IOError('Failed to create wav file')


class Template:
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


class Chromagram:
    """
    This an n-bin multi-band chromagram tuned to 440Hz.
    """    
    def __init__(self, spectrum, samp_rate,
                 chroma_bins = 12, band_fqs = None):
        """
        spectrum is only left half of the spectrum, so its length
        is signal_length / 2.
        """
        window_size = len(spectrum) * 2

        if band_fqs is not None:
            band = map(lambda b: len(spectrum) * b / samp_rate, band_fqs)
            subspectrum = spectrum[band[0]:band[1]]
            freqs = np.arange(band[0], band[1]) * samp_rate / window_size
        else:
            subspectrum = spectrum
            freqs = np.arange(0, len(spectrum)) * samp_rate / window_size

        self.values = np.zeros(chroma_bins)
        c0 = 16.3516
        for i, val in enumerate(subspectrum):
            freq = freqs[i]
            if freq > 0: # disregard dc offset
                bin = round(chroma_bins * math.log(freq / c0, 2)) % chroma_bins
                self.values[bin] += val

        if self.values.max() == 0:
            self.values = np.zeros(chroma_bins)
        else:
            self.values = self.values / self.values.max()


simple_keymap = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
                 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
                 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
                 'C:minor': 3, 'C#:minor': 4, 'Db:minor': 4, 'D:minor': 5, 'D#:minor': 6, 'Eb:minor': 6,
                 'E:minor': 7, 'F:minor': 8, 'F#:minor': 9, 'Gb:minor': 9, 'G:minor': 10, 'G#:minor': 11,
                 'Ab:minor': 11, 'A:minor': 0, 'A#:minor': 1, 'Bb:minor': 1, 'B:minor': 2, 'Silence': None}

class LabParser:

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
            time = float(row[0])
            keys.append(Key(key, time))
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

class KeyLab:

    def __init__(self, lab_file):
        self.keys = LabParser().parse_keys(lab_file)

    def key_at(self, time):
        # brute force for now
        for k in reversed(self.keys):
            if k.time <= time:
                return k.key
        return None

class Db:

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

    def select(self, columns, where = None, order = None):
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
        return [dict(zip(columns, row)) for row in result.fetchall()]

class RawDb(Db):

    def __init__(self, database, chroma_bins = 3 * 12, bands_count = 3):
        chroma_bins = chroma_bins
        bands_count = bands_count
        columns = [('chroma_%d_%d' % (i, j),  'FLOAT NOT NULL')
                   for i in range(bands_count)
                   for j in range(chroma_bins)]
        columns = [('track_id', 'INTEGER NOT NULL'),
                   ('i', 'INTEGER NOT NULL'),
                   ('key', 'INTEGER NOT NULL')] + columns
        Db.__init__(self, database, 'raw', columns)

# TODO: higher order
class HMM:
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

class Tuner:

    def tune_chromas(self, chromas):
        pass

def dot_product(a, b):
    return sum(map(operator.mul, a, b))


def generate_spectrogram(audio, window_size):
    for t in xrange(0, len(audio), window_size):
        spectrum = abs(fft(audio[t:(t + window_size)]))
        spectrum = spectrum[0:len(spectrum) / 2]
        yield (t, spectrum)

def downsample(sig, factor):
    fir = signal.firwin(61, 1.0 / factor)
    sig2 = np.convolve(sig, fir, mode="valid")
    sig2 = np.array([int(x) for i, x in enumerate(sig2) if i % factor == 0], dtype = sig.dtype)
    return sig2
