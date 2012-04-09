from util import *
from evaluate import Evaluator
from scipy import fft
import matplotlib.pylab as pylab
import os.path as path
import os
import unittest

class TestTable(unittest.TestCase):

    def setUp(self):
        self.db_file = 'test.db'
        if path.exists(self.db_file):
            os.unlink(self.db_file)
        self.db = Table(self.db_file, 'test', [('col1', 'INTEGER'), ('col2', 'VARCHAR(100)')])
        self.db.create_table()
        self.db.insert([123, 'abc'])
        self.db.insert([456, 'def'])

    def test_read(self):
        rows = self.db.select(['col1'])
        self.assertEquals({'col1': 123}, rows[0])
        self.assertEquals({'col1': 456}, rows[1])

    def test_read_star(self):
        rows = self.db.select(['*'])
        self.assertEquals({'id': 1, 'col1': 123, 'col2': u'abc'}, rows[0])
        self.assertEquals({'id': 2, 'col1': 456, 'col2': u'def'}, rows[1])

if __name__ == '__main__':
    unittest.main()
