import unittest
from util import *
from scipy import fft
import matplotlib.pylab as plb
import scipy.io.wavfile
import os

class TestTemplate(unittest.TestCase):

    def test_profile(self):
        os.system("sox -n -c1 -r44100 -b16 tmp.wav synth 1 sin C4 sin D4 sin E4 sin F4 sin G4 sin A4 sin B4")
        # os.system("sox -n -c1 -r44100 -b16 tmp.wav synth 1 sin C4 sin E4 sin G4")
        os.system("lame tmp.wav")
        reader = Mp3Reader()
        samp_rate, signal = reader.read("tmp.mp3")
        spectrum = abs(fft(signal))[0:len(signal) / 2]
        chromagram = Chromagram(spectrum, samp_rate)
        template = BasicTemplate()

#        wavfile.write("tmp2.wav", samp_rate, signal)
        plb.bar(range(12), chromagram.values)
        plb.show()
        

        self.assertEquals(0, template.match(chromagram))

        os.system("sox -n -c1 -r44100 -b16 tmp.wav synth 1 sin C4 sin D4 sin E4 sin F4 sin G4 sin A4 sin Bb4")
        os.system("lame tmp.wav")
        reader = Mp3Reader()
        samp_rate, signal = reader.read("tmp.mp3")
        spectrum = abs(fft(signal))[0:len(signal) / 2]
        chromagram = Chromagram(spectrum, samp_rate)
        template = BasicTemplate()
        self.assertEquals(5, template.match(chromagram))

if __name__ == '__main__':
    unittest.main()
