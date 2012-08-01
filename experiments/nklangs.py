# Path hack.
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import matplotlib.pyplot as plt

from keydetection import *

plt.ion()

mp3 = '/home/andreas/music/The Beatles/Revolver/02_Eleanor_Rigby.mp3'
_, audio = Mp3Reader().read(mp3)

s = [spectrum for (t, spectrum) in generate_spectrogram(audio, 8192)]
filt = SpectrumQuantileFilter(99)
sf = map(filt.filter, s)

bins = 3
cs = [Chromagram.from_spectrum(ss, 44100 / 4, 12 * bins, (50, 500)) for ss in sf]
tuner = Tuner(bins, 1)
ts = tuner.tune(cs)

einklangs = chroma.get_nklang(n = 1)
zweiklangs = chroma.get_nklang(n = 2)
dreiklangs = chroma.get_nklang(n = 3)

for chroma in ts:
    
for einklang, zweiklang, dreiklang in zip(einklangs, zweiklangs, dreiklangs):
    print '%s & %s & %s \\\\' % (einklang.get_name(), zweiklang.get_name(), dreiklang.get_name())


