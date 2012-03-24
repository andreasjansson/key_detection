from util import *
from scipy import fft
import matplotlib.pyplot as plt
import math
import itertools
import numpy as np
from pprint import pprint
import ghmm
from copy import copy
import logging

class Algorithm:

    def __init__(self, mp3_file, length):
        self.samp_rate, self.audio = Mp3Reader().read(mp3_file, length)
        self.keys = []

    def execute(self):
        raise NotImplementedError()

    def filter_repeated_keys(self):
        new_keys = []
        for i, key in enumerate(self.keys):
            if i == 0 or key.key != self.keys[i - 1].key:
                new_keys.append(key)
        self.keys = new_keys
        return new_keys


class Windowed(Algorithm):

    def __init__(self, mp3_file, windows, length = None):
        Algorithm.__init__(self, mp3_file, length)
        self.windows = windows
        self.template = BasicTemplate()

    def execute(self):
        for i, offset in enumerate(self.windows):

            if i + 1 < len(self.windows):
                next_offset = self.windows[i + 1] * self.samp_rate
            else:
                next_offset = len(self.audio)

            if next_offset > len(self.audio):
                break

            spectrum = self._get_spectrum(offset * self.samp_rate, next_offset)
            chromagram = Chromagram(spectrum, self.samp_rate)
            key = self.get_next_key(chromagram)
            self.keys.append(Key(key, offset))
            print("analysed %.1f/%.1f" % (offset, len(self.audio) / self.samp_rate))

        return self.keys

    def get_next_key(self, chromagram):
        return self.template.match(chromagram)

    def _get_spectrum(self, start, end):
        """
        start and end both in samples.
        """
        spectrum = abs(fft(self.audio[start:end]))
        spectrum = spectrum[0:len(spectrum) / 2]
        return spectrum


class FixedWindow(Windowed):
    """
    Naive template based key detection algorithm with fixed window size.
    """
    def __init__(self, mp3_file, options = None, length = None):
        Windowed.__init__(self, mp3_file, None, length)
        if options is not None and 'window_size' in options:
            window_size = options.window_size
        else:
            window_size = 8192 # 2 ^ 13 = 8192, almost one second (assuming Fs = 11025)
        self.windows = np.arange(0, float(len(self.audio)) / self.samp_rate, float(window_size) / self.samp_rate)


class BeatWindowsSimple(Windowed):
    """
    Template based key detection algorithm with beat-based window sizes.
    """
    def __init__(self, mp3_file, options, length = None):
        beats = LabParser().parse_beats(options['beat_file'])
        windows = [beat.time for beat in beats]
        Windowed.__init__(self, mp3_file, windows, length)


class BasicHMM(FixedWindow):

    def execute(self):
        raw_keys = Windowed.execute(self)
        symbols = ghmm.IntegerRange(0, 12)
        stay_prob = .78
        trans_prob = .02
        trans_matrix = (np.diag([stay_prob - trans_prob] * 12) + trans_prob).tolist()
        same_prob = .78
        different_prob = .02
        emission_matrix = (np.diag([stay_prob - trans_prob] * 12) + trans_prob).tolist()
        initial = [1.0 / 12] * 12
        hmm = ghmm.HMMFromMatrices(symbols, ghmm.DiscreteDistribution(symbols), trans_matrix, emission_matrix, initial)
        emissions = ghmm.EmissionSequence(symbols, [key.key for key in raw_keys])
        actual_keys = hmm.viterbi(emissions)[0]
        keys = copy(raw_keys)
        for i in range(len(keys)):
            keys[i].key = actual_keys[i]

        return keys
        

class GaussianMixtureHMM(FixedWindow):

    def execute(self):

        raw_keys = Windowed.execute(self)
        symbols = ghmm.IntegerRange(0, 12)
        stay_prob = .78
        trans_prob = .02
        trans_matrix = (np.diag([stay_prob - trans_prob] * 12) + trans_prob).tolist()
        initial = [1.0 / 12] * 12

        mixture_matrix = None
        profile = np.array([100, 0, 100, 0, 100, 100, 0, 100, 0, 100, 0, 100])
        for i in range(0, 12):
            row = np.array([np.roll(profile, i), np.array([10] * 12), np.array([1.0/12] * 12)])
            if mixture_matrix is None:
                mixture_matrix = np.array([row])
            else:
                mixture_matrix = np.vstack((mixture_matrix, np.array([row])))

        #logging.getLogger("GHMM").setLevel(logging.DEBUG)
        f = ghmm.Float()
        hmm = ghmm.HMMFromMatrices(f, ghmm.GaussianMixtureDistribution(f), trans_matrix,
                                   mixture_matrix.tolist(), initial)
        emissions = ghmm.EmissionSequence(f, [key.key for key in raw_keys])

        test = hmm.sampleSingle(12)
        pprint("here")
        pprint(str(test))

        #actual_keys = hmm.viterbi(emissions)[0]
        actual_keys = hmm.viterbi(test)[0]

        pprint(actual_keys)

        keys = copy(raw_keys)
        for i in range(len(keys)):
            keys[i].key = actual_keys[i]

        return keys
