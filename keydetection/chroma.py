import numpy as np
import math
import matplotlib.pyplot as plt
from copy import copy
import operator

from nklang import *
from util import *

class Chromagram(object):
    '''
    This an n-bin narrow-band chromagram tuned to 440Hz.
    '''

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
    def from_spectrum(spectrum, samp_rate, chroma_bins = 12, band_fqs = None):
        '''
        Create a new chromagram from a spectrum. If band_fqs is specified,
        it must be a tuple (low, high), that define the lower and higher
        bounds of the spectrum.
        '''
        chromagram = Chromagram(chroma_bins = chroma_bins)
        window_size = len(spectrum)
        samp_rate = float(samp_rate)
        nyquist = samp_rate / 2

        if band_fqs is not None:
            low, high = map(lambda b: int(window_size * b / nyquist), band_fqs)
            subspectrum = spectrum[low:high]
            freqs = np.arange(low, high) * nyquist / window_size
        else:
            subspectrum = spectrum
            freqs = np.arange(0, len(spectrum)) * nyquist / window_size

        c0 = 16.3516
        for i, val in enumerate(subspectrum):
            freq = freqs[i]
            if freq > 0: # disregard dc offset
                bin = int(round(chroma_bins * math.log(freq / c0, 2))) % chroma_bins
                # Since the FIR filter we use before downsampling isn't very
                # steep, we take the sqrt of the spectrum to even it out a bit.
                chromagram.values[bin] += math.sqrt(val)

        return chromagram

    def get_nklang(self, threshold = .1, silent = 100, n = 2, filter_adjacent = True):
        '''
        Compute the nklang for the chromagram by sorting the amplitudes of the chromagram,
        and returning the am nklang made from the bin indices of the n highest amplitudes.
        '''
        sorted_values = np.sort(self.values)[::-1]

        amps = []
        i = 0
        while sorted_values[i] > silent and i < n:
            amps.append(sorted_values[i])
            i += 1

        if len(amps) == 0:
            return Nullklang()

        # copy values so that we can zero out values when we use them
        # if we don't do this, two equal values will return the same index
        # in both where calls
        values = copy(self.values)
        note_amps = []
        for amp in amps:
            note = np.where(values == amp)[0][0]
            note_amps.append((note, amp))
            values[note] = 0

        # if two high amplitude chroma bins are right next to each other, something
        # fishy might be going on. it's quite likely that one of them is the result
        # of spectral side lobes.
        if filter_adjacent:
            note_amps.sort(key = operator.itemgetter(1))
            all_amps = [0] * 12
            for note, a in note_amps:
                all_amps[note] = a

            notes = []
            for note, a in note_amps:
                if all_amps[(note - 1) % 12] < a and all_amps[(note + 1) % 12] < a:
                    notes.append(note)

        else:
            notes = map(operator.itemgetter(0), note_amps)

        return Anyklang(notes, n)

    def plot(self, show = True, yticks = True):
        ind = np.arange(len(self.values))
        plt.bar(ind, self.values)
        xticks = reduce(lambda x, y: x + ([y] * (self.chroma_bins / 12)), note_names, [])
        plt.xticks(ind + .4, xticks)
        if not yticks:
            plt.gca().axes.get_yaxis().set_visible(False)
        if show:
            plt.show()

    @staticmethod
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


class Tuner(object):
    '''
    Tune an n*x bin chromagram to an n bin chromagram.
    '''

    def __init__(self, bins_per_pitch, pitches = 12, global_tuning = True):
        self.bins_per_pitch = bins_per_pitch
        self.pitches = pitches
        self.global_tuning = global_tuning

    def tune(self, chromas):
        tuned_chromas = []

        if self.global_tuning:
            max_bins = [0] * self.bins_per_pitch
            for chroma in chromas:
                max_bins[self.get_max_bin(chroma)] += 1
            max_bin = max_bins.index(max(max_bins))

        for chroma in chromas:
            if not self.global_tuning:
                max_bin = self.get_max_bin(chroma)

            tuned_chroma = self.tune_chroma(chroma, max_bin)
            tuned_chromas.append(tuned_chroma)

        return tuned_chromas

    def get_max_bin(self, chroma):
        bins = [0] * self.bins_per_pitch
        for i, value in enumerate(chroma.values):
            bins[i % self.bins_per_pitch] += value
        return bins.index(max(bins))

    def tune_chroma(self, chroma, max_bin):
        values = self.roll_values(chroma.values, max_bin)
        tuned_values = [0] * self.pitches
        for i, value in enumerate(values):
            tuned_values[int(math.floor(i / self.bins_per_pitch))] += value
        return Chromagram(tuned_values)

    def roll_values(self, values, max_bin):
        mid = math.floor(self.bins_per_pitch / 2)
        if max_bin <= mid:
            shift = mid - max_bin
        else:
            shift = max_bin
        values = np.roll(values, int(shift)).tolist()
        return values
