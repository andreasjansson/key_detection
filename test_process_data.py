import unittest
from util import *
from process_data import *
import os
import os.path as path
import sqlite3 as sqlite
import sys
import random
import string
import matplotlib.pylab as pylab

class ProcessDataTest(unittest.TestCase):

    def test_roll_bands(self):
        values = [1,2,3,4]
        self.assertEquals([2,3,4,1], roll_bands(values, 1, 1, 4))
        values = [1,2,3,4, 5,6,7,8, 9,10,11,12]
        self.assertEquals([4,1,2,3, 8,5,6,7, 12,9,10,11], roll_bands(values, 3, 3, 4))

    def test_set_implicit_keys(self):
        keymap = {'c': 0, 'c#': 1, 'd': 2, 'd#': 3, 'a:minor': 0}
        totals = {0: [0,1,2,3]}
        all_totals = {0: [0,1,2,3], 1: [1,2,3,0], 2: [2,3,0,1], 3: [3,0,1,2]}
        self.assertEquals(all_totals, set_implicit_keys(totals, keymap))

    def test_get_chroma_mean(self):
        processor = Processor()
        chroma_mean = processor.get_chroma_mean()

        # TODO: fix, do some exploratory analysis of tuned data
        pylab.bar(range(12 * 3), chroma_mean[0])
        pylab.show()
        pylab.bar(range(12 * 3), chroma_mean[1])
        pylab.show()
        pylab.bar(range(12 * 3), chroma_mean[2])
        pylab.show()

    # def test_markov(self):
    #     processor = Processor()
    #     processor.get_markov_matrix()

    # def test_rows_by_track(self):
    #     processor = Processor()
    #     for rows in processor.rows_by_track():
    #         print(len(rows))

if __name__ == '__main__':
    unittest.main()
