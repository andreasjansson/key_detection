import numpy as np
import math

from nklang import *

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

    # TODO: differentiate between <C, E> and <E, C>
    def get_nklang(self, threshold = .1, silent = 100):
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
