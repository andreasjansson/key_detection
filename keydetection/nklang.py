import math
from copy import copy
import numpy as np

from util import *

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

    def transpose(self, delta):
        transposed_notes = map(lambda n: (n + delta) % 12, self.original_notes)
        return Anyklang(transposed_notes, len(self.notes))

    def get_n(self):
        return len(self.original_notes)

    def __repr__(self):
        return '<%d-klang: %s>' % (self.get_n(), self.get_name())



def klang_number_to_name(number):
    '''
    Helper that returns the name of a Zweiklang.
    '''
    if number == -1:
        return 'Silence'
    else:
        first = number % 12
        second = int(math.floor(number / 12))
        if first == second:
            return note_names[first]
        else:
            return note_names[first] + ', ' + note_names[second]

