#!/usr/bin/env python

# for d in ~/data/bach10/0*; do echo $(basename $d); python getkey.py -t -i $d/$(basename $d).wav -o ~/scratch/key_transitions/$(basename $d).csv -t -m model.pkl; echo; done

from StringIO import StringIO
from copy import copy
import argparse
import logging
import math
import numpy as np
import operator
import os
import os.path
import pickle
import scipy
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import sys
import tempfile

note_names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']

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
    max_fq = 800

    if audio_filename:
        logging.debug('Reading audio file')
        _, audio = AudioReader.for_filename(audio_filename).read(audio_filename)

    if time_limit:
        audio = audio[: fs * time_limit] # first [time_limit] seconds

    logging.debug('Generating spectrum')
    s = [spectrum for (t, spectrum) in generate_spectrogram(audio, winlength)]

    logging.debug('Filtering spectrum')

    upper_bound = int(math.ceil(winlength * max_fq / (fs / 2)))

    filt = SpectrumGrainFilter(upper_bound)
    s = map(filt.filter, s)

    # add the missing zeroes at the end to get the length right
    s = map(lambda spectrum: spectrum + [0] * (winlength - upper_bound), s)

    logging.debug('Getting chromagram')
    cs = [Chromagram.from_spectrum(ss, fs, 12, (20, max_fq)) for ss in s]

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

class Chromagram(object):
    '''
    This an n-bin narrow-band chromagram tuned to 440Hz.
    '''

    def __init__(self, values = None, chroma_bins = None):

        if values is None:
            self.values = np.zeros(chroma_bins)
            self.chroma_bins = chroma_bins
        elif len(values) < 2:
            raise Exception('At least two values are required for a chromagram')
        elif values is not None and chroma_bins is not None:
            raise Exception('Please specify values or chroma_bins, not both.')
        else:
            self.values = values
            self.chroma_bins = len(values)

    @staticmethod
    def from_spectrum(spectrum, samp_rate, chroma_bins = 12, band_fqs = None):
        '''
        Create a new chromagram from a spectrum. If band_fqs is specified,
        it must be a tuple (low, high), that define the lower and higher
        bounds of the spectrum.
        '''
        chromagram = Chromagram(chroma_bins = chroma_bins)
        window_size = len(spectrum)
        samp_rate = float(samp_rate)
        nyquist = samp_rate / 2

        if band_fqs is not None:
            low, high = map(lambda b: int(window_size * b / nyquist), band_fqs)
            subspectrum = spectrum[low:high]
            freqs = np.arange(low, high) * nyquist / window_size
        else:
            subspectrum = spectrum
            freqs = np.arange(0, len(spectrum)) * nyquist / window_size

        c0 = 16.3516
        for i, val in enumerate(subspectrum):
            freq = freqs[i]
            if freq > 0: # disregard dc offset
                bin = int(round(chroma_bins * math.log(freq / c0, 2))) % chroma_bins
                # Since the FIR filter we use before downsampling isn't very
                # steep, we take the sqrt of the spectrum to even it out a bit.
                chromagram.values[bin] += math.sqrt(val)

        return chromagram

    def get_nklang(self, threshold = .1, silent = 100, n = 2, filter_adjacent = True):
        '''
        Compute the nklang for the chromagram by sorting the amplitudes of the chromagram,
        and returning the am nklang made from the bin indices of the n highest amplitudes.
        '''
        sorted_values = np.sort(self.values)[::-1]

        amps = []
        i = 0
        while sorted_values[i] > silent and i < n:
            amps.append(sorted_values[i])
            i += 1

        if len(amps) == 0:
            return Nullklang()

        # copy values so that we can zero out values when we use them
        # if we don't do this, two equal values will return the same index
        # in both where calls
        values = copy(self.values)
        note_amps = []
        for amp in amps:
            note = np.where(values == amp)[0][0]
            note_amps.append((note, amp))
            values[note] = 0

        # if two high amplitude chroma bins are right next to each other, something
        # fishy might be going on. it's quite likely that one of them is the result
        # of spectral side lobes.
        if filter_adjacent:
            note_amps.sort(key = operator.itemgetter(1))
            all_amps = [0] * 12
            for note, a in note_amps:
                all_amps[note] = a

            notes = []
            for note, a in note_amps:
                if all_amps[(note - 1) % 12] < a and all_amps[(note + 1) % 12] < a:
                    notes.append(note)

        else:
            notes = map(operator.itemgetter(0), note_amps)

        return Anyklang(notes, n)

class Key(object):
    '''
    Base class for major and minor keys.
    '''

    def __init__(self, root):
        self.root = root

    def __hash__(self):
        return self.root

    def __eq__(self, other):
        return type(self) == type(other) and hash(self) == hash(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    @staticmethod
    def from_repr(string):
        match = re.search(r'<(Major|Minor)Key: ([A-G]#?)>', string)
        if not match:
            return None
        root = note_names.index(match.group(2))
        if match.group(1) == 'Major':
            return MajorKey(root)
        else:
            return MinorKey(root)

class MajorKey(Key):

    def __repr__(self):
        return '%s major' % note_names[self.root]

    def mirex_repr(self):
        return '%s\tmajor' % note_names[self.root]
    
class MinorKey(Key):

    def __repr__(self):
        return '%s minor' % note_names[self.root]

    def mirex_repr(self):
        return '%s\tminor' % note_names[self.root]


class Nklang(object):
    '''
    "Abstract" base class for all types of nklang.
    '''

    def get_number(self):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

class Nullklang(Nklang):
    '''
    Used for silent sections.
    '''

    def __init__(self):
        pass

    def get_name(self):
        return '-'

    def get_number(self):
        return -1

    def transpose(self, _):
        return Nullklang()

    def __repr__(self):
        return '<Nullklang>'

class Anyklang(object):
    '''
    An nklang, where n > 0.
    Numerically represented as sum_{i = 0}^{n - 1} k_i * 12^i, where
    k_i is the i:th note in the klang.
    '''

    def __init__(self, notes, n):
        self.original_notes = copy(notes)
        self.notes = notes
        if len(notes) < n:
            self.notes += [self.notes[-1]] * (n - len(self.notes))

    def get_name(self):
        return ', '.join(map(lambda n: note_names[n], self.original_notes))

    def get_number(self):
        return np.dot(np.array(self.notes), (12 ** np.arange(len(self.notes))))

    def get_profile(self):
        p = Profile(12 ** len(self.notes))
        p.increment(self)
        return p

    def transpose(self, delta):
        transposed_notes = map(lambda n: (n + delta) % 12, self.original_notes)
        return Anyklang(transposed_notes, len(self.notes))

    def get_n(self):
        return len(self.original_notes)

    def __repr__(self):
        return '<%d-klang: %s>' % (self.get_n(), self.get_name())


class Profile:

    def __init__(self, length = None):
        if length is None:
            self.length = 0
            self.values = None
        else:
            self.length = length
            self.values = np.zeros(length)

    @staticmethod
    def from_values(values):
        profile = Profile()
        profile.values = values
        profile.length = len(values)
        return profile

    def increment(self, klang):
        self.values[klang.get_number()] += 1

    def transpose_key(self, delta):
        values = np.roll(self.values, delta % self.length, 0)
        return Profile.from_values(values)

    def add(self, other):
        if self.length != other.length:
            raise Exception('Cannot add profiles of different shapes')
        for i in range(self.length):
            self.values[i] += other.values[i]

    def add_constant(self, k):
        for i in range(self.length):
            self.values[i] += k
        
    def multiply_constant(self, k):
        for i in range(self.length):
            self.values[i] *= k
        
    def similarity(self, other):
        return np.dot(self.values, other.values)

    def normalise(self):
        sum = np.sum(self.values)
        if sum > 0:
            self.values /= sum

    def get_n(self):
        return int(math.log(len(self.values), 12))

    def __repr__(self):
        return '<Profile length %s, sum %f>' % (self.length, np.sum(self.values))


def get_test_profile(audio_filename, time_limit=None, n=2):
    '''
    Returns a single profile profile from an mp3/wav filename.
    '''

    klangs = get_klangs(audio_filename, time_limit=time_limit, n=n)
    profile = Profile(12 ** n)
    for t, klang in klangs:
        if klang is not None and \
                not isinstance(klang, Nullklang):
            profile.increment(klang)

    return profile

def get_profile_similarities(klangs, model):
    sims = np.zeros((len(model), len(klangs)))
    for i, profile in enumerate(model):
        for j, (t, klang) in enumerate(klangs):
            sims[i, j] = profile.similarity(klang.get_profile())
    return sims

def time_keys(klangs, model):
    scores = np.zeros((len(klangs), len(klangs), len(model)))
    for p, profile in enumerate(model):
        for i, (t, klang) in enumerate(klangs):
            inner_profile = Profile(12 ** 2)
            for j in range(i, len(klangs)):
                inner_profile.increment(klangs[j][1])
                scores[i, j, p] = profile.similarity(inner_profile) / (j - i + 1)
    return scores

def moving_average(klangs, model, w=20):
    scores = np.zeros((len(model), len(klangs) - w))
    for p, profile in enumerate(model):
        for i, (t, klang) in enumerate(klangs[:-w]):
            inner_profile = Profile(12 ** 2)
            for j in range(i, i + w):
                inner_profile.increment(klangs[j][1])
            scores[p, i] = profile.similarity(inner_profile) / w
    return scores

def find_best_path(values, change_cost=1.007):
    values = np.max(values) - values
    costs = np.zeros(values.shape)
    prev = np.zeros(values.shape)

    costs[:, 0] = values[:, 0]

    for i in range(1, costs.shape[1]):
        for p1 in range(costs.shape[0]):

            min_cost = float('+inf')

            for p2 in range(costs.shape[0]):
                cost = costs[p2, i - 1] + values[p1, i]
                if p1 != p2:
                    cost *= change_cost
                if cost < min_cost:
                    min_cost = cost
                    min_p = p2

            costs[p1, i] = min_cost
            prev[p1, i] = min_p

    path = [0] * values.shape[1]
    path[-1] = np.argmin(costs[:, -1])

    for i in reversed(range(len(path) - 1)):
        path[i] = prev[path[i + 1], i]

    return path

def get_keys_for_path(path, klangs):
    keys = []
    for i in range(len(path)):
        n = int(path[i])
        # TODO: this is a hack because the first key was always c (bug)
        if i != 0 and path[i] != path[i - 1]:
            if n > 12:
                key = MinorKey(n - 12)
            else:
                key = MajorKey(n)
            keys.append((klangs[i][0], key))
    return keys

def postprocess_keytimes(keytimes, window):
    fs = 11025.0
    w = 4096.0

    postprocessed = {}
    for i, (time, key) in enumerate(keytimes):
        if len(postprocessed) == 0:
            time = 0
        else:
            time = int(round(time + (window / 2) * (w / fs)))
        postprocessed[time] = key

    postprocessed = sorted([tuple(x) for x in postprocessed.items()])

    keytimes = []
    for i in range(len(postprocessed) - 1):
        if postprocessed[i][1] != postprocessed[i + 1][1]:
            keytimes.append(postprocessed[i])

    keytimes.append(postprocessed[-1])

    return keytimes

def get_key_transitions(audio_filename, model, time_limit=None, n=2):
    klangs = get_klangs(audio_filename, time_limit=time_limit, n=n)
    window = 15
    scores = moving_average(klangs, model, window)
    path = find_best_path(scores)
    keytimes = get_keys_for_path(path, klangs)
    keytimes = np.array(keytimes)
    keytimes = postprocess_keytimes(keytimes, window)
    return keytimes

def normalise_model(model, smoothing=True):
    for profile in model:
        msum = np.sum(profile.values)
        if smoothing:
            profile.add_constant(1) # laplace smoothing
        if msum > 0: # normalise with sum from before smoothing, so that the smoothing constant is indeed constant
            profile.values /= msum
    return model

def get_key(model, test_profile):
    '''
    Computes the key based on a trained model and
    a test profile.
    '''
    argmax = -1
    maxsim = 0
    for i, profile in enumerate(model):

        sim = profile.similarity(test_profile)

        if sim > maxsim:
            maxsim = sim
            argmax = i

    argmax = argmax % 24
    if argmax < 12:
        return MajorKey(argmax)
    else:
        return MinorKey(argmax - 12)

def load_model(model_filename):
    with open(model_filename, 'rb') as f:
        values_list = pickle.load(f)
    model = []
    for values in values_list:
        model.append(Profile.from_values(values))
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'MIREX-formatted key detection')
    parser.add_argument('-i', '--input', required = True, help = 'Wav filename')
    parser.add_argument('-o', '--output', required = True, help = 'Output filename (or - for stdout)')
    parser.add_argument('-m', '--model', default = 'model.pkl', help = 'The trained, pickled model (defaults to model.pkl)')
    parser.add_argument('-t', '--transitions', action='store_true', default=False)
    parser.add_argument('-v', '--verbose', action = 'store_true', default = False, help = 'Enable verbose logging')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level = logging.DEBUG)

    model = load_model(args.model)

    if args.transitions:
        keytimes = get_key_transitions(args.input, model)
        if args.output == '-':
            args.output = '/dev/stdout'
        np.savetxt(args.output, keytimes, fmt='%d,%s')

    else:
        try:
            test_profile = get_test_profile(args.input, time_limit = 30)

            if np.sum(test_profile.values) == 0:
                logging.warning('Silent audio file: %s' % (args.input))
                sys.exit(2)

            key = get_key(model, test_profile)
        except Exception as e:
            logging.warn('Failed to get key for %s: %s' % (args.input, e))
            sys.exit(1)
        line = '%s\n' % key.mirex_repr()

        if args.output == '-':
            print line

        else:
            with open(args.output, 'w') as f:
                f.write(line)
