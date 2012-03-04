from util import *
from scipy import fft
import matplotlib.pyplot as plt
import math

class Naive(Algorithm):
    """
    Naive template based key detection algorithm with fixed window size.
    """

    # 2 ^ 13 = 8192, almost one second (assuming Fs = 11025)
    def __init__(self, mp3_file, window_size = 8192, length = None):
        self.window_size = window_size
        self.template = BasicTemplate()
        Algorithm.__init__(self, mp3_file, length)

    def execute(self):
        windows = int(math.ceil(len(self.audio) / self.window_size))
        for i in range(windows):
            spectrum = self._get_spectrum(i)
            chromagram = Chromagram(spectrum, self.samp_rate)
            time = float(i) * self.window_size / self.samp_rate
            key = self.template.match(chromagram)
            self.keys.append(Key(key, time))
        return self.keys

    def _get_spectrum(self, index):
        start = index * self.window_size
        end = min((index + 1) * self.window_size, len(self.audio))
        spectrum = abs(fft(self.audio[start:end]))
        spectrum = spectrum[0:len(spectrum) / 2]
        return spectrum

