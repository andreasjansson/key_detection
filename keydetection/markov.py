import numpy as np
from copy import copy
from cache import *
from keylab import *
from nklang import *
from audio import *
import logging

class MarkovMatrix:
    '''
    An width x width Markov matrix, where the rows are the "from" states,
    and the columns are the "to" states.
    '''

    def __init__(self, width = None):
        if width is None:
            self.width = 0
            self.m = None
        else:
            self.width = width
            self.m = np.zeros(shape = (width, width))

    @staticmethod
    def from_matrix(m):
        '''
        Create a new MarkovMatrix from a raw matrix.
        '''
        markov = MarkovMatrix()
        markov.m = m
        markov.width = np.shape(m)[0]
        return markov

    def increment(self, klang1, klang2):
        '''
        Increment the transition count from klang1 to klang2.
        '''
        x = klang1.get_number()
        y = klang2.get_number()

        self.m[x][y] += 1

    def transpose_key(self, delta):
        '''
        Transpose the MarkovMatrix from one key to another, by delta semitones.
        '''
        m = np.roll(self.m, delta % self.width, 0)
        m = np.roll(m, delta % self.width, 1)
        return MarkovMatrix.from_matrix(m)

    def add(self, other):
        '''
        Add the other matrix to this matrix, cell by cell.
        '''
        if np.shape(self.m) != np.shape(other.m):
            raise Error('Cannot add markov matrices of different shapes')
        for i in range(self.width):
            for j in range(self.width):
                self.m[i][j] += other.m[i][j]

    def add_constant(self, k):
        '''
        Add a constant value to each cell of the matrix. Useful in
        Laplace smoothing.
        '''
        for i in range(self.width):
            for j in range(self.width):
                self.m[i][j] += k

    def multiply(self, factor):
        '''
        Multiply each cell of the matrix with a constant.
        '''
        for i in range(self.width):
            for j in range(self.width):
                self.m[i][j] *= factor

    def similarity(self, other):
        '''
        Compute the similarity of two matrices by taking the sum of
        the product of the cells of the two matrices.
        '''
        return np.dot(self.m.ravel(), other.m.ravel())

    def similarity_unmarkov(self, other):
        '''
        Compute the similarity as the dot product of the column sums
        of two matrices.
        '''
        return np.dot(self.get_unmarkov_array(), other.get_unmarkov_array())

        #products = 1000 * self.get_unmarkov_array() * other.get_unmarkov_array()
        #return np.prod(filter(lambda x: x > 0, products))

    def normalise(self):
        '''
        Normalise the matrix so that the sum of the values of cells add up
        to 1.
        '''
        sum = np.sum(self.m)
        if sum > 0:
            self.m /= sum

    def get_unmarkov_array(self):
        '''
        Return a vector of the column sums of the matrix. This effectively
        returns the number of occurances of each of the 144 Zweiklangs, discarding
        the first one.
        '''
        return self.m.sum(axis = 0)

    def get_density(self):
        return len(np.where(self.m == 0)[0]) / float(self.width ** 2)

    def __repr__(self):
        s = np.shape(self.m)
        return '<Matrix %dx%d sum %f>' % (s[0], s[1], np.sum(self.m))

    def print_summary(self, max_lines = 20):
        '''
        Print a dump of the most common transitions.
        '''
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

    def print_summary_unmarkov(self, max_lines = 20):
        '''
        Print a dump of the most common klangs.
        '''
        unmarkov = self.get_unmarkov_array()
        unmarkov = zip(range(self.width ** 2), unmarkov)
        unmarkov.sort(key = operator.itemgetter(1))
        for klang_number, score in unmarkov[:max_lines]:
            print '%6s: %.3f' % (klang_number_to_name(klang_number), score)


def aggregate_matrices(matrices_list):
    '''
    Add a list of markov matrices.
    '''
    aggregate_matrices = [MarkovMatrix(12 * 12) for i in range(12 * 2)]
    for matrices in matrices_list:
        if matrices is not None:
            for i in range(24):
                aggregate_matrices[i].add(matrices[i])

    return aggregate_matrices

def get_trained_model(filenames):
    '''
    Helper function that takes a list of (mp3_filename, keylab_filename) tuples
    and returns a trained model consisting of 24 144x144 matrices.
    '''
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
    Helper function that returns 12 major and 12 minor markov matrices
    learned from the keylab and klangs.
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
    '''
    Helper that wraps extracts klangs from mp3 and keylab from keylab_file
    and passes them to get_markov_matrices.
    '''
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
    '''
    Returns a single markov matrix from an mp3.
    '''
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
