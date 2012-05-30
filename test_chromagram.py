import unittest
from util import *
from scipy import fft
import matplotlib.pylab as pylab
import os
import sys
import numpy as np

class TestChromagram(unittest.TestCase):

    def test_multiband_profile(self):
        #                                                        65     247    294    1397   2349   4699
        os.system("sox -n -c1 -r 44100 -b16 tmp1.wav synth 4 sin C1 sin B2 sin D4 sin F6 sin C7 sin D8")
        reader = WavReader()
        samp_rate, signal = reader.read("tmp1.wav", downsample_factor = 4)
        window_size = 8192
        spectrum = abs(fft(signal[0:window_size] * np.hanning(window_size)))

        #pylab.plot(spectrum)
        #pylab.show()

        divs = [33, 262, 2093, 5000]
        bands = zip(divs[:-1], divs[1:])

        maxes = [[0, 11], [2, 5], [0, 2]]
        b = 1
        for i, band in enumerate(bands):
            m = maxes[i]
            chroma = Chromagram.from_spectrum(spectrum, samp_rate, 12 * b, band).values

            #pylab.bar(range(12 * b), chroma)
            #pylab.show()

            order = np.argsort(chroma)
            top2 = np.sort(order[-2:])
            self.assertTrue(np.all(top2 == m))

        
    def test_profile(self):
        os.system("sox -n -c1 -r44100 -b16 tmp.wav synth 1 sin C4 sin E4 sin G4")
        os.system("lame tmp.wav")
        reader = Mp3Reader()
        samp_rate, signal = reader.read("tmp.mp3")
        spectrum = abs(fft(signal * np.hanning(len(signal))))[0:len(signal) / 2]

        chromagram = Chromagram.from_spectrum(spectrum, samp_rate)
        
        #pylab.bar(range(12), chromagram.values)
        #pylab.show()

        chromagram.normalise()
        print map(lambda x: round(x, 3), chromagram.values)

        self.assertTrue(np.all(chromagram.values >
                               np.array([.5, 0, 0, 0, .5, 0, 0, .5, 0, 0, 0, 0])))
        self.assertTrue(np.all(chromagram.values <=
                               np.array([1, .5, .5, .5, 1, .5, .5, 1, .5, .5, .5, .5])))

    def test_zweiklang(self):
        c = Chromagram(np.array([0, .8, .9, 1]))
        self.assertEquals(-1, c.get_zweiklang().get_number())

        c = Chromagram(np.array([0, 1, 0]))
        self.assertEquals(1, c.get_zweiklang().get_number())

        c = Chromagram(np.array([0, 1, 0, 1]))
        self.assertEquals(1 + 3 * 4, c.get_zweiklang().get_number())

if __name__ == '__main__':
    unittest.main()
