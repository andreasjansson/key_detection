import numpy as np
from copy import copy
from cache import *
from keylab import *
from nklang import *
from audio import *
import logging

class MarkovMatrix:

    def __init__(self, width = None):
        if width is None:
            self.width = 0
            self.m = None
        else:
            self.width = width
            self.m = np.zeros(shape = (width, width))

    @staticmethod
    def from_matrix(m):
        markov = MarkovMatrix()
        markov.m = m
        markov.width = np.shape(m)[0]
        return markov

    def increment(self, klang1, klang2):
        x = klang1.get_number()
        y = klang2.get_number()

        self.m[x][y] += 1

    # "transpose" in the musical sense, not matrix transposition
    def transpose_key(self, delta):
        m = np.roll(self.m, delta % self.width, 0)
        m = np.roll(m, delta % self.width, 1)
        return MarkovMatrix.from_matrix(m)

    # mutable for performance
    def add(self, other):
        if np.shape(self.m) != np.shape(other.m):
            raise Error('Cannot add markov matrices of different shapes')
        for i in range(self.width):
            for j in range(self.width):
                self.m[i][j] += other.m[i][j]

    def similarity(self, other):
        return np.dot(self.m.ravel(), other.m.ravel())

    def similarity_unmarkov(self, other):
        return np.dot(self.get_unmarkov_array(), other.get_unmarkov_array())

    def normalise(self):
        sum = np.sum(self.m)
        if sum > 0:
            self.m /= sum

    # add up all columns to get the relative frequencies of the "to"
    # values in the klang analysis. known bug: first klang is discarded.
    def get_unmarkov_array(self):
        return self.m.sum(axis = 0)

    def __repr__(self):
        s = np.shape(self.m)
        return '<Matrix %dx%d sum %f>' % (s[0], s[1], np.sum(self.m))

    def print_summary(self, max_lines = 20):
        # in numpy most things are mutable. side effects everywhere,
        # so need to copy all along.
        m = copy(self.m)
        seq = m.reshape(-1)
        seq.sort()
        seq = np.unique(seq)[::-1]
        i = 0
        lines = 0
        while(seq[i] > 0 and i < len(seq)):
            where = np.where(seq[i] == self.m)
            for fr0m, to in zip(where[0], where[1]):
                print '%6s => %-6s: %.3f' % (klang_number_to_name(fr0m), klang_number_to_name(to), seq[i])
                lines += 1
                if lines > max_lines:
                    return
            i += 1

    def multiply(self, factor):
        for i in range(self.width):
            for j in range(self.width):
                self.m[i][j] *= factor

def aggregate_matrices(matrices_list):
    aggregate_matrices = [MarkovMatrix(12 * 12) for i in range(12 * 2)]
    for matrices in matrices_list:
        if matrices is not None:
            for i in range(24):
                aggregate_matrices[i].add(matrices[i])
                

    for i, matrix in enumerate(aggregate_matrices):
        matrix.normalise()

        if i >= 12:
            matrix.multiply(.5)

    return aggregate_matrices

def get_aggregate_markov_matrices(filenames):
    aggregates = [MarkovMatrix(12 * 12) for i in range(12 * 2)]
    n = 1
    matrices_list = []
    for mp3, keylab_file in filenames:
        logging.info('Analysing %s' % mp3)
        matrices = get_training_matrices(mp3, keylab_file)
        if matrices:
            logging.debug('Appending matrices')
            matrices_list.append(matrices)
        else:
            logging.debug('No matrices to append')

    logging.debug('Aggregating matrices')
    aggregates = aggregate_matrices(matrices_list)

    return aggregates

def get_markov_matrices(keylab, klangs):
    '''
    Return one or two matrices in a dict
    keyed by mode.
    '''
    mwidth = 12 * 12

    # 12 major and 12 minor matrices
    matrices = [MarkovMatrix(mwidth) for i in range(12 * 2)]
    prev_klang = None
    prev_key = None
    for t, klang in klangs:
        key = keylab.key_at(t)

        if key is not None and \
                prev_klang is not None and \
                prev_key == key and \
                not isinstance(klang, Nullklang) and \
                not isinstance(prev_klang, Nullklang):

            for i in range(12):

                def t(klang):
                    root = key.root
                    return klang.transpose(-root + i)

                if isinstance(key, MajorKey):
                    matrices[i].increment(t(prev_klang), t(klang))
                elif isinstance(key, MinorKey):
                    matrices[i + 12].increment(t(prev_klang), t(klang))

        prev_klang = klang
        prev_key = key

    return matrices


def get_training_matrices(mp3, keylab_file):
    cache = Cache('training', '%s:%s' % (mp3, keylab_file))
    if cache.exists():
        matrices = cache.get()
    else:
        logging.debug('Getting key lab')
        keylab = get_key_lab(keylab_file)

        # no point doing lots of work if it won't
        # give any results
        if not keylab.is_valid():
            return None

        try:
            logging.debug('About to get klangs')
            klangs = get_klangs(mp3)
        except Exception, e:
            logging.warning('Failed to analyse %s: %s' % (mp3, e))
            return None

        logging.debug('About to get matrices')
        matrices = get_markov_matrices(keylab, klangs)
        logging.debug('About to set cache')
        cache.set(matrices)

    return matrices

def get_test_matrix(mp3):

    cache = Cache('test', mp3)
    if cache.exists():
        return cache.get()

    klangs = get_klangs(mp3)
    matrix = MarkovMatrix(12 * 12)
    prev_klang = None
    for t, klang in klangs:
        if prev_klang is not None and \
                not isinstance(klang, Nullklang) and \
                not isinstance(prev_klang, Nullklang):
            matrix.increment(prev_klang, klang)
        prev_klang = klang

    cache.set(matrix)

    return matrix

def get_key(training_matrices, test_matrix, unmarkov = False):
    argmax = -1
    maxsim = 0
    for i, matrix in enumerate(training_matrices):

        if unmarkov:
            sim = matrix.similarity_unmarkov(test_matrix)
        else:
            sim = matrix.similarity(test_matrix)

        if sim > maxsim:
            maxsim = sim
            argmax = i

    argmax = argmax % 24
    if argmax < 12:
        return MajorKey(argmax)
    else:
        return MinorKey(argmax - 12)

def get_pitch_class_model():
    major_matrix = MarkovMatrix(12 * 12)
    minor_matrix = MarkovMatrix(12 * 12)

    temperley_major = [0.748, 0.060, 0.488, 0.082, 0.670, 0.460,
                       0.096, 0.715, 0.104, 0.366, 0.057, 0.400]

    temperley_minor = [0.712, 0.084, 0.474, 0.618, 0.049, 0.460,
                       0.105, 0.747, 0.404, 0.067, 0.133, 0.330]

    for from1 in range(12):
        for from2 in range(12):
            for to1 in range(12):
                for to2 in range(12):

                    major_matrix.m[from1 * 12 + from2, to1 * 12 + to2] = \
                        math.sqrt(temperley_major[from1] ** 2 + temperley_major[from2] ** 2) * \
                        math.sqrt(temperley_major[to1] ** 2 + temperley_major[to2] ** 2)
                    minor_matrix.m[from1 * 12 + from2, to1 * 12 + to2] = \
                        math.sqrt(temperley_minor[from1] ** 2 + temperley_minor[from2] ** 2) * \
                        math.sqrt(temperley_minor[to1] ** 2 + temperley_minor[to2] ** 2)


    model = [None] * 24
    for i in range(12):
        model[i] = major_matrix.transpose_key(i)
        model[i + 12] = minor_matrix.transpose_key(i)

    for m in model:
        m.normalise()

    return model
