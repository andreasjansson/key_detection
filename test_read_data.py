import unittest
from util import *
from read_data import *
import os
import os.path as path
import sqlite3 as sqlite
import sys
import random
import string

class ReadDataTest(unittest.TestCase):

    def setUp(self):
        self.db = ''.join(random.sample(string.ascii_lowercase, 10)) + ".db"

    def tearDown(self):
        if path.exists(self.db):
            os.remove(self.db)

if __name__ == '__main__':
    unittest.main()

