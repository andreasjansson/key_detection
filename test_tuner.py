import unittest
from util import *
import os
import os.path as path
import sqlite3 as sqlite
import sys
import random
import string

class TunerTest(unittest.TestCase):

    def tet_get_max_bin(self):
        tuner = Tuner(3, 1, 4)
        row = [0, 1, .3, .4, 1, .3, .1, .5, .1, .2, .3, .2]
        self.assertEquals(1, tuner.get_max_bin(row))
        row = [1, .3, .4, 1, .3, .1, .5, .1, .2, .3, .2, 0]
        self.assertEquals(0, tuner.get_max_bin(row))
        row = [.3, .4, 1, .3, .1, .5, .1, .2, .3, .2, 0, 1]
        self.assertEquals(2, tuner.get_max_bin(row))

    def test_roll_chroma(self):
        tuner = Tuner(3, 1, 2)
        r = [0, 1, 2, 3, 4, 5]
        row = [0, 1, 2, 3, 4, 5]
        self.assertEquals(row, tuner.roll_chroma(r, 1))
        row = [5, 0, 1, 2, 3, 4]
        self.assertEquals(row, tuner.roll_chroma(r, 0))
        row = [4, 5, 0, 1, 2, 3]
        self.assertEquals(row, tuner.roll_chroma(r, 2))

    def test_tune_single_band_1(self):
        tuner = Tuner(3, 1, 4)
        rows = [[0, 1, .3, .4, 1, .3, .1, .5, .1, .2, .3, .2]]
        tuned_rows = tuner.tune(rows)
        self.assertEquals([[1.3, 1.7, .7, .7]], tuned_rows)

    def test_tune_single_band_2(self):
        tuner = Tuner(3, 1, 4)
        rows = [[1, .3, .4, 1, .3, .1, .5, .1, .2, .3, .2, 0]]
        tuned_rows = tuner.tune(rows)
        self.assertEquals([[1.3, 1.7, .7, .7]], tuned_rows)

    def test_tune_single_band_3(self):
        tuner = Tuner(3, 1, 4)
        rows = [[.3, .4, 1, .3, .1, .5, .1, .2, .3, .2, 0, 1]]
        tuned_rows = tuner.tune(rows)
        self.assertEquals([[1.3, 1.7, .7, .7]], tuned_rows)

    def test_tune_multi_band(self):
        tuner = Tuner(3, 2, 3)
        rows = [[1, 0, 0,  0, 1, .3,  .5, 0, 1,   0, 0, 0,  1, 0, 0,  1, 0, .1]]
        tuned_rows = tuner.tune(rows)
        self.assertEquals([[2, 1, .8, .1, 1, 1]], tuned_rows)

    def test_tune_mutli_rows(self):
        tuner = Tuner(3, 1, 3)
        rows = [
            [1, 0, 0,   0, 1, 0,  0, 0, 1],
            [0, 1, 0,  .1, 0, 1,  1, 0, 0]]
        tuned_rows = tuner.tune(rows)
        self.assertEquals([[2, 1, 0], [1, .1, 2]], tuned_rows)

if __name__ == '__main__':
    unittest.main()
