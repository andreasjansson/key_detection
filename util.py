from subprocess import call
import tempfile
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import os
from itertools import repeat

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
    
    def read_mono(self, mp3_filename):
        wav_filename = tempfile.NamedTemporaryFile('w+b', -1, '.wav', 'tmp', None, False)
        self._mp3_to_wav(mp3_filename, wav_filename)
        samp_rate, stereo = wavfile.read(wav_filename)
        os.unlink(wav_filename)
        mono = stereo[:,0] + stereo[:,1]
        # pad with zeroes before downsampling
        mono = mono + repeat(0, 4 - (len(mono) % 4))
        # downsample
        mono = signal.resample(mono, len(mono) / 4)
        return (samp_rate / 4, mono)
        
    def _mp3_to_wav(self, mp3_filename, wav_filename):
        # TODO: UPNEXT
        pass
