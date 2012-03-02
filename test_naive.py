import unittest
from util import *
from naive import Naive
from scipy import fft
import matplotlib.pylab as pylab
import os

class TestTemplate(unittest.TestCase):

    def test_profile(self):
        os.system("sox -n -c1 -r44100 -b16 tmp.wav synth 3 sin E4 sin F4 sin G4 sin A4 sin Bb4")
        os.system("lame tmp.wav")
        naive = Naive("tmp.mp3")
        keys = naive.execute()
        self.assertEquals(4, len(keys))
        keys_only = [k.key for k in keys]
        self.assertEquals([5] * 4, keys_only)

if __name__ == '__main__':
    unittest.main()
