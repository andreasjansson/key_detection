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

from chroma import *

class AudioReader(object):

    def _process(self, samp_rate, stereo, length, downsample_factor):

        if len(stereo.shape) == 2:
            mono = stereo[:,0]
        else:
            mono = stereo

        if length and len(mono) / samp_rate > length:
            mono = mono[0:int(length * samp_rate)]

        # pad with zeroes before downsampling
        padding = np.array([0] * (downsample_factor - (len(mono) % downsample_factor)), dtype = mono.dtype)
        mono = np.concatenate((mono, padding))
        # downsample
        if downsample_factor > 1:
            mono = downsample(mono, downsample_factor)
        return (samp_rate / downsample_factor, mono)

class WavReader(AudioReader):

    def read(self, wav_filename, length = None, downsample_factor = 4):
        samp_rate, stereo = wavfile_read_silent(wav_filename)
        return self._process(samp_rate, stereo, length, downsample_factor)

class Mp3Reader(AudioReader):

    def read(self, mp3_filename, length = None, downsample_factor = 4):
        """
        Returns (sampling_rate, data), where the sampling rate is the
        original sampling rate, downsampled by a factor of 4, and
        the data signal is a downsampled, mono (left channel) version
        of the original signal.
        """

        wav_filename = tempfile.NamedTemporaryFile(suffix = '.wav', delete = False).name
        self._mp3_to_wav(mp3_filename, wav_filename)
        samp_rate, stereo = wavfile_read_silent(wav_filename)
        os.unlink(wav_filename)
        return self._process(samp_rate, stereo, length, downsample_factor)
        
    def _mp3_to_wav(self, mp3_filename, wav_filename):
        if mp3_filename.find('http') == 0:
            mp3_filename = download(mp3_filename, '.mp3')

        if not os.path.exists(mp3_filename):
            raise IOError('File not found')
        os.system("mpg123 -w \"" + wav_filename.replace('"', '\\"') + "\" \"" + mp3_filename.replace('"', '\\"') + "\" &> /dev/null")
        if not os.path.exists(wav_filename):
            raise IOError('Failed to create wav file')

class SpectrumQuantileFilter(object):

    def __init__(self, quantile = 97, window_width = 200):
        self.quantile = quantile
        self.window_width = 200

    def filter(self, spectrum):
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

class SpectrumPeakFilter(object):

    def __init__(self, audio, window_size = 8192,
                 samp_rate = 11025, max_peaks = 20):
        self.audio = audio
        self.window_size = window_size
        self.samp_rate = samp_rate
        self.max_peaks = max_peaks

    def filter(self, spectrum, frame):
        import simpl
        spectrum = [0] * len(spectrum)
        audio = self.audio[(self.window_size * frame):
                               (self.window_size * (frame + 1))]
        pd = simpl.SndObjPeakDetection()
        pd.set_sampling_rate(self.samp_rate)
        pd.max_peaks = self.max_peaks
        frames = pd.find_peaks(audio)
        for frame in frames:
            for peak in frame.peaks:
                freq = peak.frequency / 4
                bin = int(freq * self.window_size / self.samp_rate)
                spectrum[bin] += peak.amplitude * 32768.0
            return spectrum

def get_klangs(mp3 = None, audio = None):
    fs = 11025
    winlength = 8192

    if mp3:
        _, audio = Mp3Reader().read(mp3)

    s = [spectrum for (t, spectrum) in generate_spectrogram(audio, winlength)]

    filt = SpectrumQuantileFilter(98)
    sf = map(filt.filter, s)

    bins = 3
    cs = [Chromagram.from_spectrum(ss, fs / 4, 12 * bins, (50, 500)) for ss in sf]

    tuner = Tuner(bins, 1)
    ts = tuner.tune(cs)

    return [(i * winlength / fs, t.get_nklang()) for i, t in enumerate(ts)]

def wavfile_read_silent(wav_filename):
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    samp_rate, stereo = wavfile.read(wav_filename)
    sys.stdout = old_stdout
    return (samp_rate, stereo)

def generate_spectrogram(audio, window_size):
    for t in xrange(0, len(audio), window_size):
        # windowed spectrogram
        actual_window_size = min(window_size, len(audio) - t)
        windowed_signal = audio[t:(t + window_size)] * np.hanning(actual_window_size)
        spectrum = abs(scipy.fft(windowed_signal))
        spectrum = spectrum[0:len(spectrum) / 2]
        yield (t, spectrum)

def normalise_spectra(spectra):
    spectra = copy(spectra)
    for i, spectrum in enumerate(spectra):
        m = max(spectrum)
        if m > 0:
            spectrum = (np.array(spectrum) / max(spectrum)).tolist()
        spectra[i] = spectrum
    return spectra

def downsample(sig, factor):
    fir = signal.firwin(61, 1.0 / factor)
    sig2 = np.convolve(sig, fir, mode="valid")
    sig2 = np.array([int(x) for i, x in enumerate(sig2) if i % factor == 0], dtype = sig.dtype)
    return sig2
