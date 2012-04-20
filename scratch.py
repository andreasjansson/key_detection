import util
import matplotlib.pyplot as plt

mp3 = '/home/andreas/music/The Beatles/Abbey_Road/11_Mean_Mr_Mustard.mp3'
_, audio = util.Mp3Reader().read(mp3)
s = [spectrum for (t, spectrum) in util.generate_spectrogram(audio, 8192)]
filt = util.SpectrumPeakFilter(audio)
sf = [filt.filter(x, i) for i, x in enumerate(s)]

# show the effect of filtering
plt.plot(s[50], 'r-', sf[50], 'b-')
plt.show()

cs = [util.Chromagram(ss, 44100 / 4, 12, (200, 1000)) for ss in sf]
tuner = util.Tuner(3, 1)
csv = [c.values for c in cs]
t = tuner.tune(csv)
