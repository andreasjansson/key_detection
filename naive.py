from util import *
from numpy.fft import fft

class Naive(Algorithm):
    """
    Naive template based key detection algorithm with fixed window size.
    """

    # 2 ^ 15 = 32768, almost one second (assuming Fs = 44100)
    def __init__(self, mp3_file, window_size = 32768):
        self.window_size = window_size
        self.template = BasicTemplate()

    def execute(self):
        windows = ceil(len(self.audio) / window_size)
        for i in range(len(spectra)):
            spectrum = self._get_spectrum(i)
            time = i * self.window_size
            key = self.template.match(spectrum)
            this.keys.append(Key(key, time))
        return this.keys

    def _get_spectrum(self, index):
        start = index * self.window_size
        end = min((index + 1) * self.window_size, len(self.audio))
        spectrum = fft(self.audio[start:end])
        return spectrum

