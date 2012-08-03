# Path hack.
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import matplotlib.pyplot as plt

from keydetection import *

plt.ion()

mp3 = '/home/andreas/music/The Beatles/Revolver/13_Got_To_Get_You_Into_My_Life.mp3'
_, audio = Mp3Reader().read(mp3)

bins = 3

s = [spectrum for (t, spectrum) in generate_spectrogram(audio, 8192)]

filt99 = SpectrumQuantileFilter(99)
sf99 = map(filt99.filter, s)
cs99 = [Chromagram.from_spectrum(ss, 44100 / 4, 12 * bins, (50, 500)) for ss in sf99]
tuner99 = Tuner(bins, 1)
ts99 = tuner99.tune(cs99)

filt95 = SpectrumQuantileFilter(95)
sf95 = map(filt95.filter, s)
cs95 = [Chromagram.from_spectrum(ss, 44100 / 4, 12 * bins, (50, 500)) for ss in sf95]
tuner95 = Tuner(bins, 1)
ts95 = tuner95.tune(cs95)

f = open('klangs.txt', 'w')

for i, (chroma95, chroma99) in enumerate(zip(ts95, ts99)):
    einklang95 = chroma95.get_nklang(n = 1)
    zweiklang95 = chroma95.get_nklang(n = 2)
    dreiklang95 = chroma95.get_nklang(n = 3)
    einklang99 = chroma99.get_nklang(n = 1)
    zweiklang99 = chroma99.get_nklang(n = 2)
    dreiklang99 = chroma99.get_nklang(n = 3)
#    f.write('%d & %s & %s & %s & %s & %s & %s \\\\\n' % (
#            i * 8192, einklang95.get_name(), zweiklang95.get_name(), dreiklang95.get_name(),
#            einklang99.get_name(), zweiklang99.get_name(), dreiklang99.get_name()))

    print(zweiklang99.get_name())

f.close()

