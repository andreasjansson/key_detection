import operator
import tempfile
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import numpy as np
import scipy
import os
import sys
import os.path
from StringIO import StringIO
import matplotlib.pyplot as plt
import logging

from util import *
from chroma import *

class AudioReader(object):
    '''
    Base class for mp3 and wav readers.
    '''

    def process(self, samp_rate, stereo, length, downsample_factor):
        '''
        Pre-process audio by making it mono and downsampling.
        '''

        logging.debug('Making mono')

        if len(stereo.shape) == 2:
            mono = stereo[:,0]
        else:
            mono = stereo

        if length and len(mono) / samp_rate > length:
            mono = mono[0:int(length * samp_rate)]

        logging.debug('Padding')
        # pad with zeroes before downsampling
        padding = np.array([0] * (downsample_factor - (len(mono) % downsample_factor)), dtype = mono.dtype)
        logging.debug('Making mono')
        mono = np.concatenate((mono, padding))
        # downsample
        if downsample_factor > 1:
            mono = downsample(mono, downsample_factor)

        logging.debug('Finished processing audio')

        return (samp_rate / downsample_factor, mono)

    @staticmethod
    def for_filename(filename):
        extension = os.path.splitext(filename)[1]
        if extension == '.wav':
            return WavReader()
        if extension == '.mp3':
            return Mp3Reader()
        raise Exception('Unknown audio file extension: %s' % extension)

class WavReader(AudioReader):

    def read(self, wav_filename, length = None, downsample_factor = 4):
        logging.debug('About to read wavfile')
        samp_rate, stereo = WavReader.read_silent(wav_filename)
        return self.process(samp_rate, stereo, length, downsample_factor)

    @staticmethod
    def read_silent(wav_filename):
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        samp_rate, stereo = wavfile.read(wav_filename)
        sys.stdout = old_stdout
        return (samp_rate, stereo)



class Mp3Reader(AudioReader):

    def read(self, mp3_filename, length = None, downsample_factor = 4):
        '''
        Returns (sampling_rate, data), where the sampling rate is the
        original sampling rate, downsampled by a factor of 4, and
        the data signal is a downsampled, mono (left channel) version
        of the original signal.
        '''

        mp3_filename = make_local(mp3_filename, '/tmp/mp3.mp3')

        # first we must convert to wav
        wav_file = tempfile.NamedTemporaryFile(suffix = '.wav', delete = False)
        wav_filename = wav_file.name
        wav_file.close()
        self.mp3_to_wav(mp3_filename, wav_filename)
        
        samp_rate, stereo = WavReader.read_silent(wav_filename)
        os.unlink(wav_filename)
        return self.process(samp_rate, stereo, length, downsample_factor)
        
    def mp3_to_wav(self, mp3_filename, wav_filename):
        if mp3_filename.find('http') == 0:
            mp3_filename = download(mp3_filename, '.mp3')

        if not os.path.exists(mp3_filename):
            raise IOError('File not found')
        os.system("mpg123 -q -w \"" + wav_filename.replace('"', '\\"') + "\" \"" + mp3_filename.replace('"', '\\"') + "\"")
        logging.debug('Finished decoding mp3')
        if not os.path.exists(wav_filename):
            raise IOError('Failed to create wav file')

class SpectrumQuantileFilter(object):
    '''
    A really crude spectral peak detector / filter that works by splitting the spectrum
    into fixed size windows and setting all spectral bins that have an amplitude
    below a certain amplitude quantile for that window to 0.
    '''

    def __init__(self, quantile = 99, window_width = 200, upper_bound = None):
        self.quantile = quantile
        self.window_width = window_width
        self.upper_bound = upper_bound

    def filter(self, spectrum):
        if self.upper_bound:
            spectrum = spectrum[:self.upper_bound]

        filtered = []
        for i in range(0, len(spectrum), self.window_width):
            subspec = list(spectrum[i:(i + self.window_width)])
            sortspec = [(i, v) for i, v in enumerate(subspec)]
            sortspec.sort(key = operator.itemgetter(1))
            q = int(len(sortspec) * (self.quantile / 100.0))
            for i in range(q):
                subspec[sortspec[i][0]] = 0
            filtered += subspec
        return filtered

class SpectrumGrainFilter(object):
    '''
    Imagine a line plot of a spectrum, but upside down. Now place tiny grains of
    sand along the x-axis, at the spectral bin points. Drop the grains and let
    them fall down on the spectrum. They slide on the gradients on the spectrum
    and end up in little groups at the bottom of the spectrum where they can't slide
    or fall any further. When all grains have stopped moving, set all spectral bins
    that have no grains in them to 0. Flip it back around the x-axis. Filtered spectrum.
    '''

    def __init__(self, upper_bound = None):
        self.upper_bound = upper_bound

    def filter(self, spectrum):
        if self.upper_bound:
            spectrum = spectrum[:self.upper_bound]

        moving_grains = range(len(spectrum))
        stable_grains = []

        while len(moving_grains) > 0:
            for (i, x) in reversed(list(enumerate(moving_grains))):

                def stable():
                    stable_grains.append(x)
                    del moving_grains[i]

                if x > 0 and x < len(spectrum) - 1:
                    if spectrum[x] >= spectrum[x - 1] and spectrum[x] >= spectrum[x + 1]:
                        stable()
                    elif spectrum[x] < spectrum[x - 1]:
                        moving_grains[i] -= 1
                    else:
                        moving_grains[i] += 1

                elif x == 0:
                    if spectrum[x] >= spectrum[x + 1]:
                        stable()
                    else:
                        moving_grains[i] += 1

                else:
                    if spectrum[x] >= spectrum[x - 1]:
                        stable()
                    else:
                        moving_grains[i] -= 1

        filtspec = [0] * len(spectrum)
        for x in stable_grains:
            filtspec[x] = spectrum[x]

        return filtspec


def get_klangs(audio_filename = None, audio = None, time_limit = None, n = 2):
    '''
    Helper function that reads and pre-processes an mp3/wav, computes the spectrogram,
    filters each spectrum in the spectrogram, computes the chromagram for each spectrum,
    and for each chromagram, computes the nklang.    
    '''
    fs = 11025
    winlength = 4096

    max_fq = 1000

    if audio_filename:
        logging.debug('Reading audio file')
        _, audio = AudioReader.for_filename(audio_filename).read(audio_filename)

    if time_limit:
        audio = audio[: fs * time_limit] # first [time_limit] seconds

    logging.debug('Generating spectrum')
    s = [spectrum for (t, spectrum) in generate_spectrogram(audio, winlength)]

    logging.debug('Filtering spectrum')

    upper_bound = int(math.ceil(winlength * max_fq / (fs / 2)))

    filt = SpectrumQuantileFilter(98, 100, upper_bound = upper_bound)
    sf = map(filt.filter, s)

    filt = SpectrumGrainFilter(upper_bound)
    sf = map(filt.filter, sf)

    # add the missing zeroes at the end to get the length right
    sf = map(lambda spectrum: spectrum + [0] * (winlength - upper_bound), sf)

    bins = 1
    logging.debug('Getting chromagram')
    cs = [Chromagram.from_spectrum(ss, fs, 12 * bins, (20, max_fq)) for ss in sf]

    if bins > 1:
        logging.debug('Tuning')
        tuner = Tuner(bins, global_tuning = True)
        cs = tuner.tune(cs)

    logging.debug('Returning klangs')
    klangs = [(i * winlength / float(fs), t.get_nklang(n = n)) for i, t in enumerate(cs)]
    return klangs

def generate_spectrogram(audio, window_size):
    '''
    Hanning-windowed spectrogram
    '''
    for t in xrange(0, len(audio), window_size):
        actual_window_size = min(window_size, len(audio) - t)
        windowed_signal = audio[t:(t + window_size)] * np.hanning(actual_window_size)
        spectrum = abs(scipy.fft(windowed_signal))
        spectrum = spectrum[0:len(spectrum) / 2]
        yield (t, spectrum)

def normalise_spectra(spectra):
    '''
    Normalise to 1.
    '''
    spectra = copy(spectra)
    for i, spectrum in enumerate(spectra):
        m = max(spectrum)
        if m > 0:
            spectrum = (np.array(spectrum) / max(spectrum)).tolist()
        spectra[i] = spectrum
    return spectra

def downsample(sig, factor):
    '''
    Low-pass filter using simple FIR, then pick every n sample, where n is
    the downsampling factor.
    '''
    logging.debug('Creating filter')
    fir = signal.firwin(61, 1.0 / factor)
    logging.debug('Convolving')
    sig2 = np.convolve(sig, fir, mode="valid")
    logging.debug('Downsampling')
    sig2 = [int(x) for i, x in enumerate(sig2) if i % factor == 0]
    logging.debug('Done downsampling')
    return sig2

def plot_spectrum(spectrum, fs = 11025, zoom = None, clear = True,
                  line = 'b-'):
    '''
    A graph says more than a thousand words. Or less.
    '''
    if clear:
        plt.clf()
    plt.plot(spectrum, line)
    fn = fs / 2

    nticks = 10
    if zoom is not None:
        nticks = 10 / ((zoom[1] - zoom[0]) / float(fn))

    plt.xticks(range(0, len(spectrum), int(math.ceil(len(spectrum) / nticks))), range(0, fn, int(math.ceil(fn / nticks))))
    plt.yticks([])
    if zoom is not None:
        zoom[0] = zoom[0] / float(fn) * len(spectrum)
        zoom[1] = zoom[1] / float(fn) * len(spectrum)
        zoom[2] = zoom[2] * max(spectrum)
        zoom[3] = zoom[3] * max(spectrum)
        plt.axis(zoom)
