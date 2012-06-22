import util
import matplotlib.pyplot as plt
import numpy as np

mp3 = '/home/andreas/music/The Beatles/Sgt._Pepper\'s_Lonely_Hearts_Club_Band/09_When_I\'m_Sixty-Four.mp3'
klangs = util.get_klangs(mp3)
keylab = util.KeyLab("/home/andreas/data/beatles_annotations/keylab/the_beatles/Sgt._Pepper's_Lonely_Hearts_Club_Band/09_When_I'm_Sixty-Four.lab")
matrices = util.get_markov_matrices(keylab, klangs)


# The goal is to create a number of emissions that make sense to a human. I.e. a human can deduce
# the key just by looking at the emissions. That way, we can use introspection to create an algorithm
# that does the same thing.

#mp3 = '/home/andreas/music/The Beatles/Rubber_Soul/04_Nowhere_Man.mp3'
#mp3 = '/home/andreas/music/The Beatles/Rubber_Soul/03_You_Won\'t_See_Me.mp3'
#mp3 = '/home/andreas/music/The Beatles/Rubber_Soul/03_You_Won\'t_See_Me.mp3'
#mp3 = '/home/andreas/music/The Beatles/Sgt._Pepper\'s_Lonely_Hearts_Club_Band/10_Lovely_Rita.mp3'
#mp3 = '/home/andreas/music/The Beatles/Sgt._Pepper\'s_Lonely_Hearts_Club_Band/01_Sgt._Pepper\'s_Lonely_Hearts_Club_Band.mp3'
#mp3 = '/home/andreas/music/The Beatles/Sgt._Pepper\'s_Lonely_Hearts_Club_Band/08_Within_You_Without_You.mp3'
#mp3 = '/home/andreas/music/The Beatles/Sgt._Pepper\'s_Lonely_Hearts_Club_Band/13_A_Day_In_The_Life.mp3'
#mp3 = '/home/andreas/music/The Beatles/Sgt._Pepper\'s_Lonely_Hearts_Club_Band/12_Sgt._Pepper\'s_Lonely_Hearts_Club_Band_(Reprise).mp3'
#mp3 = '/home/andreas/music/The Beatles/Sgt._Pepper\'s_Lonely_Hearts_Club_Band/11_Good_Morning_Good_Morning.mp3'
#mp3 = '/home/andreas/music/The Beatles/Sgt._Pepper\'s_Lonely_Hearts_Club_Band/04_Getting_Better.mp3'
#mp3 = '/home/andreas/music/The Beatles/Sgt._Pepper\'s_Lonely_Hearts_Club_Band/05_Fixing_A_Hole.mp3'
#mp3 = '/home/andreas/music/The Beatles/Sgt._Pepper\'s_Lonely_Hearts_Club_Band/03_Lucy_In_The_Sky_With_Diamonds.mp3'
#mp3 = '/home/andreas/music/The Beatles/Sgt._Pepper\'s_Lonely_Hearts_Club_Band/07_Being_For_The_Benefit_Of_Mr._Kite!.mp3'
mp3 = '/home/andreas/music/The Beatles/Sgt._Pepper\'s_Lonely_Hearts_Club_Band/09_When_I\'m_Sixty-Four.mp3'
#mp3 = '/home/andreas/music/The Beatles/Sgt._Pepper\'s_Lonely_Hearts_Club_Band/06_She\'s_Leaving_Home.mp3'
#mp3 = '/home/andreas/music/The Beatles/Sgt._Pepper\'s_Lonely_Hearts_Club_Band/02_With_A_Little_Help_From_My_Friends.mp3'


_, audio = util.Mp3Reader().read(mp3)
s = [spectrum for (t, spectrum) in util.generate_spectrogram(audio, 8192)]
#filt = util.SpectrumPeakFilter(audio)
#sf = [filt.filter(x, i) for i, x in enumerate(s)]

# using windowed quantile filter made a massive difference.
filt = util.SpectrumQuantileFilter(98)
sf = map(filt.filter, s)

# show the effect of filtering
#plt.plot(s[50], 'r-', sf[50], 'b-')
#plt.show()

bins = 3
cs = [util.Chromagram.from_spectrum(ss, 44100 / 4, 12 * bins, (50, 500)) for ss in sf]
tuner = util.Tuner(bins, 1)
ts = tuner.tune(cs)
#for i in range(0, len(ts), 16):
#    util.plot_chromas(ts[i:(i + 16)])
#    break

# this isn't very good:
# (think this proves that the traditional template-based approach
# is broken, no matter which template you use)
#tt = np.array(t).sum(0)
#tt.plot()

# looking at the plot before, it was much more obvious which key
# we were in. let's see if normalising to max == 1 helps make a
# more distinct chroma.
#sn = util.normalise_spectra(t)
#stt = np.array(sn[0:32]).sum(0)
#util.plot_chroma(stt)

# again, the raw template approach proves not that great.
# there must be a way to take advantage of the sequential nature


for t in ts:
    print "%-10s%.0f" % (t.get_zweiklang().get_name(), sum(t.values))





# Two types of clustering to discretise:
# 1. K-means
# 2. Pre-defined clusters of einklang und zweiklang


# Write up what we discussed. Explain what each point means to me (as well as I understand).
#  - Discretise with clustering algorithm above
#  - Multiple "chord" candidates plus key context
#  - Mixing higher and lower order Markov models
#  - Hierarchical models working on different time scales
