from util import *
from evaluate import Evaluator
from scipy import fft
import matplotlib.pylab as pylab
import unittest

class TestMp3Reader(unittest.TestCase):

    def test_read(self):
        reader = Mp3Reader()
        samp_rate, signal = reader.read("sine_440.mp3")
        signal = signal[0:samp_rate / 2] # 500ms
        spectrum = abs(fft(signal))[0:len(signal) / 2]
        peak = np.where(spectrum == max(spectrum))[0]
        peak_freq = peak * samp_rate / len(signal) 
        self.assertGreater(442, peak_freq)
        self.assertLess(438, peak_freq)

if __name__ == '__main__':
    unittest.main()
