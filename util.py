import tempfile
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import numpy as np
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
        mono = stereo[:,0]
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
