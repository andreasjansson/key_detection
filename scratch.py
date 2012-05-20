import util
import matplotlib.pyplot as plt
import numpy as np

mp3 = '/home/andreas/music/The Beatles/Magical_Mystery_Tour/09_Penny_Lane.mp3'
_, audio = util.Mp3Reader().read(mp3)
s = [spectrum for (t, spectrum) in util.generate_spectrogram(audio, 8192)]
#filt = util.SpectrumPeakFilter(audio)
#sf = [filt.filter(x, i) for i, x in enumerate(s)]

# using windowed quantile filter made a massive difference.
filt = util.SpectrumQuantileFilter(98)
sf = map(filt.filter, s)

# show the effect of filtering
plt.plot(s[50], 'r-', sf[50], 'b-')
plt.show()

cs = [util.Chromagram(ss, 44100 / 4, 36, (50, 500)) for ss in sf]
tuner = util.Tuner(3, 1)
csv = [c.values for c in cs]
t = tuner.tune(csv)
for i in range(0, len(t), 16):
    util.plot_chromas(t[i:(i + 16)])
    break

# this isn't very good:
# (think this proves that the traditional template-based approach
# is broken, no matter which template you use)
tt = np.array(t).sum(0)
util.plot_chroma(tt)

# looking at the plot before, it was much more obvious which key
# we were in. let's see if normalising to max == 1 helps make a
# more distinct chroma.
sn = util.normalise_spectra(t)
stt = np.array(sn[0:32]).sum(0)
util.plot_chroma(stt)

# again, the raw template approach proves not that great.
# there must be a way to take advantage of the sequential nature


# Two types of clustering to discretise:
# 1. K-means
# 2. Pre-defined clusters of einklang und zweiklang


# Write up what we discussed. Explain what each point means to me (as well as I understand).
#  - Discretise with clustering algorithm above
#  - Multiple "chord" candidates plus key context
#  - Mixing higher and lower order Markov models
#  - Hierarchical models working on different time scales
