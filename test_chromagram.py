import unittest
from util import *
from scipy import fft
import matplotlib.pylab as pylab
import os

class TestChromagram(unittest.TestCase):

    def test_profile(self):
        os.system("sox -n -c1 -r44100 -b16 tmp.wav synth 1 sin C4 sin E4 sin G4")
        os.system("lame tmp.wav")
        reader = Mp3Reader()
        samp_rate, signal = reader.read("tmp.mp3")
        spectrum = abs(fft(signal))[0:len(signal) / 2]

        chromagram = Chromagram(spectrum, samp_rate)

#        pylab.bar(range(12), chromagram._values)
#        pylab.show()

        self.assertTrue(np.all(chromagram.values >
                               np.array([.5, 0, 0, 0, .5, 0, 0, .5, 0, 0, 0, 0])))
        self.assertTrue(np.all(chromagram.values <=
                               np.array([1, .5, .5, .5, 1, .5, .5, 1, .5, .5, .5, .5])))

if __name__ == '__main__':
    unittest.main()
