import operator
import tempfile
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import numpy as np
import math
import os
import csv
from scipy.spatial.distance import cosine

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


class Mp3Reader:

    def read(self, mp3_filename, length = None):
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

        if len(stereo.shape) == 2:
            mono = stereo[:,0]
        else:
            mono = stereo

        if length and len(mono) / samp_rate > length:
            mono = mono[0:int(length * samp_rate)]

        # pad with zeroes before downsampling
        padding = np.array([0] * (4 - (len(mono) % 4)), dtype = mono.dtype)
        mono = np.concatenate((mono, padding))
        # downsample
        downsample_factor = 4
        if downsample_factor > 1:
            mono = downsample(mono, downsample_factor)
        return (samp_rate / downsample_factor, mono)
        
    def _mp3_to_wav(self, mp3_filename, wav_filename):
        if not os.path.exists(mp3_filename):
            raise IOError('File not found')
        os.system("mpg123 -w " + wav_filename + " " + mp3_filename + " &> /dev/null")
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
    This is a simple 12-bin chromagram (1 bin per semitone),
    tuned to 440.
    """    
    def __init__(self, spectrum, samp_rate):
        """
        spectrum is only left half of the spectrum, so its length
        is signal_length / 2.
        """
        self.values = np.zeros(12)
        freqs = np.arange(len(spectrum)) * samp_rate / (len(spectrum) * 2)
        c0 = 16.3516
        for i, val in enumerate(spectrum):
            freq = freqs[i]
            if freq > 0: # disregard dc offset
                bin = round(12 * math.log(freq / c0, 2)) % 12
                self.values[bin] += val

        if self.values.max() == 0:
            self.values = np.zeros(12)
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
                key = keymap[row[3]]
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

def dot_product(a, b):
    return sum(map(operator.mul, a, b))


def generate_spectrogram(audio, window_size):
    for t in xrange(len(audio), step = window_size):
        spectrum = abs(fft(audio[t:(t + window_size)]))
        spectrum = spectrum[0:len(spectrum) / 2]
        yield (t, spectrum)

def downsample(sig, factor):
    fir = signal.firwin(61, 1.0 / factor)
    sig2 = np.convolve(sig, fir, mode="valid")
    sig2 = np.array([int(x) for i, x in enumerate(sig2) if i % factor == 0], dtype = sig.dtype)
    return sig2
