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
from copy import copy
import matplotlib.pyplot as plt
import simpl

note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

class Key(object):
    def __init__(self, key, time):
        self.key = key
        self.time = time

    def __str__(self):
        return "{0}: {1}".format(self.time, self.key)

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
    def __init__(self, spectrum, samp_rate,
                 chroma_bins = 12, band_fqs = None):
        """
        spectrum is only left half of the spectrum, so its length
        is signal_length / 2.
        """
        self.chroma_bins = chroma_bins
        window_size = len(spectrum) * 2
        samp_rate = float(samp_rate)

        if band_fqs is not None:
            band = map(lambda b: len(spectrum) * b / samp_rate, band_fqs)
            subspectrum = spectrum[int(band[0]):int(band[1])]
            freqs = np.arange(band[0], band[1]) * 2 * samp_rate / window_size
        else:
            subspectrum = spectrum
            freqs = np.arange(0, len(spectrum)) * samp_rate / window_size

        self.values = np.zeros(chroma_bins)
        c0 = 16.3516
        for i, val in enumerate(subspectrum):
            freq = freqs[i]
            if freq > 0: # disregard dc offset
                bin = int(round(chroma_bins * math.log(freq / c0, 2)) % chroma_bins)
                self.values[bin] += val

#        if self.values.max() == 0:
#            self.values = np.zeros(chroma_bins)
#        else:
#            self.values = self.values / self.values.max()
            
    def plot(self):
        plot_chroma(self.values, self.chroma_bins)

def get_zweiklang(values):
    # first, determine if it's a nullklang, einklang or zweiklang
    sorted_values = np.sort(values)[::-1]
    threshold = 0.2

    # nullklang
    if sorted_values[0] * threshold < sorted_values[1] and \
            sorted_values[0] * threshold < sorted_values[2]:
        return -1

    # einklang
    if sorted_values[0] * threshold > sorted_values[1] and \
            sorted_values[0] * threshold > sorted_values[2]:
        return np.where(values == sorted_values[0])[0][0]

    # zweiklang
    return 12 + 12 * np.where(values == sorted_values[0])[0][0] + \
            np.where(values == sorted_values[1])[0][0]

def zweiklang2name(zweiklang):
    if zweiklang == -1:
        return 'Nullklang'
    if zweiklang < 12:
        return note_names[zweiklang]
    zweiklang -= 12
    return note_names[int(zweiklang / 12)] + ', ' + \
        note_names[zweiklang % 12]


def plot_chroma(values, chroma_bins = 12, show = True, yticks = True):
    ind = np.arange(len(values))
    plt.bar(ind, values)
    xticks = reduce(lambda x, y: x + ([y] * (chroma_bins / 12)), note_names, [])
    plt.xticks(ind + .4, xticks)
    if not yticks:
        plt.gca().axes.get_yaxis().set_visible(False)
    if show:
        plt.show()

def plot_chromas(matrix, chroma_bins = 12):
    root = math.sqrt(len(matrix))
    cols = int(math.floor(root))
    rows = int(math.ceil(len(matrix) / float(cols)))
    for i, values in enumerate(matrix):
        if isinstance(values, Chromagram):
            print 'here'
            values = values.values
        row = int(math.floor(i / float(cols)))
        col = i % cols
        plt.subplot2grid((rows, cols), (row, col))
        plot_chroma(values, show = False, yticks = False)
    plt.show()

simple_keymap = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
                 'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
                 'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11,
                 'C:minor': 3, 'C#:minor': 4, 'Db:minor': 4,
                 'D:minor': 5, 'D#:minor': 6, 'nEb:minor': 6,
                 'E:minor': 7, 'F:minor': 8, 'F#:minor': 9,
                 'Gb:minor': 9, 'G:minor': 10, 'G#:minor': 11,
                 'Ab:minor': 11, 'A:minor': 0, 'A#:minor': 1,
                 'Bb:minor': 1, 'B:minor': 2, 'Silence': None}


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

class KeyLab(object):

    def __init__(self, lab_file):
        self.keys = LabParser().parse_keys(lab_file)

    def key_at(self, time):
        # brute force for now
        for k in reversed(self.keys):
            if k.time <= time:
                return k.key
        return None

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

    def tune(self, rows):
        """
        Tune multiple bands of chromagrams.
        """
        tuned_rows = []
        max_bins = [0] * self.bins_per_pitch
        for row in rows:
            max_bins[self.get_max_bin(row)] += 1
        # TODO: proper argmax
        max_bin = max_bins.index(max(max_bins))
        for row in rows:
            tuned_row = self.tune_row(row, max_bin)
            tuned_rows.append(tuned_row)
        return tuned_rows

    def get_max_bin(self, row):
        bins = [0] * self.bins_per_pitch
        for i, value in enumerate(row):
            bins[i % self.bins_per_pitch] += value
        return bins.index(max(bins))

    def tune_row(self, row, max_bin):
        tuned_row = []
        for i in range(self.bands):
            tuned_row += self.tune_band(
                row[(i * self.pitches * self.bins_per_pitch) :
                        ((i + 1) * self.pitches * self.bins_per_pitch)], max_bin)
        return tuned_row

    def tune_band(self, subchroma, max_bin):
        subchroma = self.roll_chroma(subchroma, max_bin)
        tuned_subchroma = [0] * self.pitches
        for i, value in enumerate(subchroma):
            tuned_subchroma[int(math.floor(i / self.bins_per_pitch))] += value
        return tuned_subchroma

    def roll_chroma(self, chroma, max_bin):
        mid = math.floor(self.bins_per_pitch / 2)
        if max_bin <= mid:
            shift = mid - max_bin
        else:
            shift = max_bin
        chroma = np.roll(chroma, int(shift)).tolist()
        return chroma

class SpectrumQuantileFilter(object):

    def __init__(self, quantile = 99, window_width = 200):
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

def dot_product(a, b):
    return sum(map(operator.mul, a, b))

def generate_spectrogram(audio, window_size):
    for t in xrange(0, len(audio), window_size):
        spectrum = abs(fft(audio[t:(t + window_size)]))
        spectrum = spectrum[0:len(spectrum) / 2]
        yield (t, spectrum)

def normalise_spectra(spectra):
    spectra = copy(spectra)
    for i, spectrum in enumerate(spectra):
        m = max(spectrum)
        if(m > 0):
            spectrum = (np.array(spectrum) / max(spectrum)).tolist()
        spectra[i] = spectrum
    return spectra

def downsample(sig, factor):
    fir = signal.firwin(61, 1.0 / factor)
    sig2 = np.convolve(sig, fir, mode="valid")
    sig2 = np.array([int(x) for i, x in enumerate(sig2) if i % factor == 0], dtype = sig.dtype)
    return sig2

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
