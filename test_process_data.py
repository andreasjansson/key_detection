import unittest
from util import *
from process_data import *
import os
import os.path as path
import sqlite3 as sqlite
import sys
import random
import string

class ProcessDataTest(unittest.TestCase):

    def test_markov(self):
        processor = Processor()
        processor.get_markov_matrix()

    def test_rows_by_track(self):
        processor = Processor()
        for rows in processor.rows_by_track():
            print(len(rows))

if __name__ == '__main__':
    unittest.main()
