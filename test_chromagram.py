import unittest
from util import *
from scipy import fft
import matplotlib.pylab as pylab
import os
import sys

class TestChromagram(unittest.TestCase):

    def test_multiband_profile(self):
        #                                                        65     247    294    1397   2349   4699
        os.system("sox -n -c1 -r 44100 -b16 tmp1.wav synth 4 sin C1 sin B2 sin D4 sin F6 sin C7 sin D8")
        reader = WavReader()
        samp_rate, signal = reader.read("tmp1.wav", downsample_factor = 4)
        window_size = 8192
        spectrum = abs(fft(signal[0:window_size]))

        #pylab.plot(spectrum)
        #pylab.show()

        divs = [33, 262, 2093, 5000]
        bands = zip(divs[:-1], divs[1:])

        maxes = [[0, 11], [2, 5], [0, 2]]
        for i, band in enumerate(bands):
            m = maxes[i]
            chroma = Chromagram(spectrum, samp_rate, 12 * b, band).values

            pylab.bar(range(12 * b), chroma)
            pylab.show()

            order = np.argsort(chroma)
            top2 = np.sort(order[-2:])
            self.assertTrue(np.all(top2 == m))

        
    def test_profile(self):
        os.system("sox -n -c1 -r44100 -b16 tmp.wav synth 1 sin C4 sin E4 sin G4")
        os.system("lame tmp.wav")
        reader = Mp3Reader()
        samp_rate, signal = reader.read("tmp.mp3")
        spectrum = abs(fft(signal))[0:len(signal) / 2]

        chromagram = Chromagram(spectrum, samp_rate)

        #pylab.bar(range(12), chromagram._values)
        #pylab.show()

        self.assertTrue(np.all(chromagram.values >
                               np.array([.5, 0, 0, 0, .5, 0, 0, .5, 0, 0, 0, 0])))
        self.assertTrue(np.all(chromagram.values <=
                               np.array([1, .5, .5, .5, 1, .5, .5, 1, .5, .5, .5, .5])))

if __name__ == '__main__':
    unittest.main()
