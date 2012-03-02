import tempfile
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import numpy as np
import math
import os

class Key:
    def __init__(self, key, time):
        self.key = key
        self.time = time


class Algorithm:

    def __init__(self, mp3_file):
        self.samp_rate, self.audio = Mp3Reader().read_mono(mp3_file)
        self.keys = []

    def execute(self):
        raise NotImplementedError()


class Mp3Reader:

    def read(self, mp3_filename):
        """
        Returns (sampling_rate, data), where the sampling rate is the
        original sampling rate, downsampled by a factor of 4, and
        the data signal is a downsampled, mono (left channel) version
        of the original signal.
        """
        wav_filename = tempfile.NamedTemporaryFile(suffix = '.wav', delete = False).name
        self._mp3_to_wav(mp3_filename, wav_filename)
        samp_rate, stereo = wavfile.read(wav_filename)
        os.unlink(wav_filename)
        if(len(stereo.shape) == 2):
            mono = stereo[:,0]
        else:
            mono = stereo
        # pad with zeroes before downsampling
        mono = np.concatenate((mono, [0] * (4 - (len(mono) % 4))))
        # downsample
        downsample_factor = 4
        mono = signal.resample(mono, len(mono) / downsample_factor)
        return (samp_rate / downsample_factor, mono)
        
    def _mp3_to_wav(self, mp3_filename, wav_filename):
        if not os.path.exists(mp3_filename):
            raise IOError('File not found')
        os.system("mpg123 -w " + wav_filename + " " + mp3_filename + " &> /dev/null")
        if not os.path.exists(wav_filename):
            raise IOError('Failed to create wav file')


class Template:
    def match(self, chromagram):
        max_score = -1
        max_i = -1
        for i in range(12):
            profile = np.roll(self.profile, i)
            score = sum(profile * chromagram.values)
            if(score > max_score):
                max_score = score
                max_i = i
        return max_i
        

class BasicTemplate(Template):
    def __init__(self):
        self.profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])


# TODO: test!!!
class Chromagram:
    """
    This is a simple 12-bin chromagram (1 bin per semitone),
    tuned to 440.
    """    
    def __init__(self, spectrum, samp_rate):
        """
        spectrum is only left half of the spectrum, so its length
        is signal_length / 2.
        """
        self.values = np.zeros(12)
        freqs = np.arange(len(spectrum)) * samp_rate / (len(spectrum) * 2)
        for i, val in enumerate(spectrum):
            freq = freqs[i]
            if freq > 0: # disregard dc offset
                bin = self._bin_for_freq(freq)
                self.values[bin] += val
        self.values = self.values / self.values.max()

    def _bin_for_freq(self, freq):
        c0 = 16.3516
        return round(12 * math.log(freq / c0, 2)) % 12
