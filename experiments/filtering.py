# Path hack.
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import matplotlib.pyplot as plt

from keydetection import *

plt.ion()

mp3 = '/home/andreas/music/The Beatles/Revolver/02_Eleanor_Rigby.mp3'
_, audio = Mp3Reader().read(mp3)

s = [spectrum for (t, spectrum) in generate_spectrogram(audio, 8192)]
filters = [SpectrumQuantileFilter(50), SpectrumQuantileFilter(90),
           SpectrumQuantileFilter(95), SpectrumQuantileFilter(99)]
sf = [map(filt.filter, s) for filt in filters]

bins = 3
cs = [Chromagram.from_spectrum(ss, 44100 / 4, 12 * bins, (50, 500)) for ss in sf[3]]
tuner = Tuner(bins, 1)
ts = tuner.tune(cs)
for i in range(0, len(ts), 16):
    Chromagram.plot_chromas(ts[i:(i + 16)])
    break

spectrum = sf[3][13]

indices = np.where(np.array(spectrum) > 0)[0]
freqs = indices * 11025 / (len(spectrum) * 2)
c0 = 16.3516
degrees = np.round(12 * np.log2(freqs / c0)) % 12
notes = [note_names[int(i)] for i in degrees]
scores = np.array(spectrum)[indices] / max(spectrum)
round_scores = map(lambda x: '%.2f' % x, scores)

np.vstack((freqs, notes, round_scores)).transpose()[0:20, :]

chroma = Chromagram.from_spectrum(spectrum, 44100 / 4, 12, (50, 500))
plt.clf(); chroma.plot()

plot_spectrum(s[13], zoom=[0,1000,0,1])
plot_spectrum(sf[3][13], clear = False, line = 'ro-', zoom=[0,1000,0,1])
