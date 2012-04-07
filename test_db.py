from util import *
from evaluate import Evaluator
from scipy import fft
import matplotlib.pylab as pylab
import os.path as path
import os
import unittest

class TestDb(unittest.TestCase):

    def setUp(self):
        self.db_file = 'test.db'

    def test_write(self):
        if path.exists(self.db_file):
            os.unlink(self.db_file)
        db = Db(self.db_file, 'test', ['col1 INTEGER', 'col2 VARCHAR(100)'])
        db.create_table()
        db.insert([123, 'abc'])
        db.insert([456, 'def'])

    def test_read(self):
        db = Db(self.db_file, 'test', ['col1 INTEGER', 'col2 VARCHAR(100)'])
        rows = db.select(['col1'])
        self.assertEquals({'col1': 123}, rows[0])
        self.assertEquals({'col1': 456}, rows[1])

if __name__ == '__main__':
    unittest.main()
