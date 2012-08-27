import numpy as np
from copy import copy
from cache import *
from keylab import *
from nklang import *
from audio import *
import logging

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
        print len(values)
        profile.length = len(values)
        return profile

    def increment(self, klang):
        self.values[klang.get_number()] += 1

    def transpose_key(self, delta):
        values = np.roll(self.values, delta % self.length, 0)
        return Profile.from_values(values)

    def add(self, other):
        if self.length != other.length:
            raise Error('Cannot add profiles of different shapes')
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

    def print_summary(self, max_lines = 20):
        seq = copy(self.values)
        seq.sort()
        seq = np.unique(seq)[::-1]
        i = 0
        lines = 0
        n = self.get_n()
        while(seq[i] > 0 and i < len(seq)):
            where = np.where(seq[i] == self.values)[0]
            for index in where:
                print '%-6s: %.3f' % (klang_number_to_name(index, n), seq[i])
                lines += 1
                if lines > max_lines:
                    return
            i += 1

    @staticmethod
    def aggregate_multiple(profiles_list):
        aggregate_profiles = [Profile(len(profiles_list[0][0].values))
                                      for i in range(12 * 2)]
        for profiles in profiles_list:
            if profiles is not None:
                for i in range(24):
                    aggregate_profiles[i].add(profiles[i])

        return aggregate_profiles


def get_trained_model(filenames, n = 2):
    '''
    Helper function that takes a list of (mp3_filename, keylab_filename) tuples
    and returns a trained model consisting of 24 profiles.
    '''
    aggregates = [Profile(12 ** n) for i in range(12 * 2)]
    profiles_list = []
    for mp3, keylab_file in filenames:
        logging.info('Analysing %s' % mp3)
        profiles = get_training_profiles(mp3, keylab_file, n)
        if profiles:
            logging.debug('Appending profiles')
            profiles_list.append(profiles)
        else:
            logging.debug('No profiles to append')

    logging.debug('Aggregating profiles')
    aggregates = Profile.aggregate_multiple(profiles_list)

    return aggregates


def get_training_profiles(mp3, keylab_file, n = 2):
    cache = Cache('training', '%s:%s' % (mp3, keylab_file))
    if cache.exists():
        profiles = cache.get()
    else:
        logging.debug('Getting key lab')
        keylab = get_key_lab(keylab_file)

        # no point doing lots of work if it won't
        # give any results
        if not keylab.is_valid():
            return None

        try:
            logging.debug('About to get klangs')
            klangs = get_klangs(mp3, n = n)
        except Exception, e:
            logging.warning('Failed to analyse %s: %s' % (mp3, e))
            return None

        logging.debug('About to get profiles')


        # 12 major and 12 minor profiles
        profiles = [Profile(12 ** n) for i in range(12 * 2)]
        for t, klang in klangs:
            key = keylab.key_at(t)

            if key is not None and \
                    not isinstance(klang, Nullklang):

                for i in range(12):
                    root = key.root
                    if isinstance(key, MajorKey):
                        profiles[i].increment(klang.transpose(-root + i))
                    elif isinstance(key, MinorKey):
                        profiles[i + 12].increment(klang.transpose(-root + i))

        logging.debug('About to set cache')
        cache.set(profiles)

    return profiles

def get_test_profile(mp3, time_limit = None, n = 2):
    '''
    Returns a single profile profile from an mp3.
    '''

    if time_limit:
        cache = Cache('test_%d' % time_limit, mp3)
    else:
        cache = Cache('test', mp3)
    if cache.exists():
        return cache.get()

    klangs = get_klangs(mp3, time_limit = time_limit, n = n)
    profile = Profile(12 ** n)
    for t, klang in klangs:
        if klang is not None and \
                not isinstance(klang, Nullklang):
            profile.increment(klang)

    cache.set(profile)

    return profile
