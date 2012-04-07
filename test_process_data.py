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
        print(processor.get_markov_matrix())

if __name__ == '__main__':
    unittest.main()
