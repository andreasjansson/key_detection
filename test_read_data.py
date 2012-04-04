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
        os.remove(self.db)

    def test_writer_create_db(self):
        writer = SqliteDataWriter(self.db)
        self.assertTrue(path.exists(self.db))

    def test_writer_create_table(self):
        writer = SqliteDataWriter(self.db)
        writer.create_table()        

    def test_writer_write_row(self):
        writer = SqliteDataWriter(self.db, 3, 2)
        writer.create_table()        

        row = range(8)
        self.assertRaises(Exception, writer.write_row, row)
        row = range(10)
        self.assertRaises(Exception, writer.write_row, row)
        row = range(9)
        writer.write_row(row)
        writer.write_row(row)
        row2 = random.sample(row, 9)
        self.assertFalse(row == row2)
        writer.write_row(row2)

if __name__ == '__main__':
    unittest.main()

