import util
import matplotlib.pyplot as plt
import np

mp3 = '/home/andreas/music/The Beatles/Abbey_Road/11_Mean_Mr_Mustard.mp3'
_, audio = util.Mp3Reader().read(mp3)
s = [spectrum for (t, spectrum) in util.generate_spectrogram(audio, 8192)]
#filt = util.SpectrumPeakFilter(audio)
#sf = [filt.filter(x, i) for i, x in enumerate(s)]
filt = util.SpectrumQuantileFilter(99)
sf = map(filt.filter, s)

# show the effect of filtering
plt.plot(s[50], 'r-', sf[50], 'b-')
plt.show()

cs = [util.Chromagram(ss, 44100 / 4, 36, (50, 500)) for ss in sf]
tuner = util.Tuner(3, 1)
csv = [c.values for c in cs]
t = tuner.tune(csv)
util.plot_chromas(t[0:16])

# this isn't very good:
# (think this proves that the traditional template-based approach
# is broken, no matter which template you use)
tt = np.array(t).sum(0)
util.plot_chromas(tt)
