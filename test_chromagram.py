import unittest
from util import *
from scipy import fft
import matplotlib.pylab as pylab
import os

class TestChromagram(unittest.TestCase):

    def test_multiband_profile(self):
        os.system("sox -n -c1 -r44100 -b16 tmp1.wav synth 1 sin C1 sin C2 sin C3 sin C4 sin C5 sin E6 sin E7 sin F8")
        os.system("lame tmp1.wav")
        reader = Mp3Reader()
        samp_rate, signal = reader.read("tmp1.mp3")
        window_size = 8192;
        spectrum = abs(fft(signal[0:window_size]))[0:(window_size / 2)]

        pylab.plot(spectrum)
        pylab.show()

        p1 = 0
        p2 = int((window_size / 2) / 3.0)
        p3 = int((window_size / 2) * (2.0/3.0))
        p4 = window_size / 2

        subspec1 = spectrum[p1:p2]
        subspec2 = spectrum[p2:p3]
        subspec3 = spectrum[p3:p4]
        pylab.plot(subspec1)
        pylab.show()
        pylab.plot(subspec2)
        pylab.show()
        pylab.plot(subspec3)
        pylab.show()
        chrg1 = Chromagram(subspec1, samp_rate, 12, p1, window_size)
        chrg2 = Chromagram(subspec2, samp_rate, 12, p2, window_size)
        chrg3 = Chromagram(subspec3, samp_rate, 12, p3, window_size)

        pylab.bar(range(12), chrg3.values)
        pylab.show()

        """
        self.assertTrue(np.all(chrg1.values >
                               np.array([.5, 0, 0, 0, .5, 0, 0, .5, 0, 0, 0, 0])))
        self.assertTrue(np.all(chromagram.values <=
                               np.array([1, .5, .5, .5, 1, .5, .5, 1, .5, .5, .5, .5])))
                               """
        
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
